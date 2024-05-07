"""Script used for computing guidance during inference."""

from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import cv2
# from scipy.ndimage import binary_dilation
from omegaconf import DictConfig
import heapq
# import kaolin
# from utils.smplx_utils import convert_smplx_parameters_format
from models.optimizer.optimizer import Optimizer
from models.networks import OPTIMIZER
from models.loss.oriented_iou_loss import cal_iou_3d
from kaolin.ops.mesh import check_sign
from utils.utils import get_textured_objects
from utils.overlap import bbox_overlap, voxel_grid_from_mesh
import open3d as o3d
from datasets.gapartnet_dataset import MapThreedfuture2gparnet
import random
from tqdm import trange

def check_articulate( class_label, classes):
    query_label = classes[class_label.argmax(-1)]  #TODO
    if query_label=='start' or query_label == 'end' or query_label == 'empty':
        print("error")

    if query_label in MapThreedfuture2gparnet:    
        isArticulated = True
    else:
        isArticulated = False
    return isArticulated 

def draw_2d_gaussian(center, size, angle, image_size = 256):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    covariance_matrix = np.array([
        [size[0]**2, 0],
        [0, size[1]**2]
    ])
    rotation_convariance_matrix = rotation_matrix @ covariance_matrix @ rotation_matrix.T

    x = np.arange(0,image_size)
    y = np.arange(0,image_size)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx.ravel(), yy.ravel()]).T -center
    try:
        z = np.sum((xy @ np.linalg.inv(rotation_convariance_matrix)) * xy, axis=1)
    except:
        breakpoint()
    gaussian = np.exp(-0.5 * z)
    gaussian = gaussian.reshape(xx.shape)
    return gaussian

def heuristic_distance(node1, node2):
    # return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def find_shortest_path(matrix, start, end):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    open_set = [(0, start)]
    parent_map = {}
    g_cost = {node: float('inf') for node in np.ndindex(matrix.shape)}
    g_cost[start] = 0
    count = 0
    while open_set and count<50000:
        count+=1
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in parent_map:
                path.append(current)
                current = parent_map[current]
            path.append(start)
            return path[::-1]

        for direction in directions:
            new_node = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= new_node[0] < matrix.shape[0] and 0 <= new_node[1] < matrix.shape[1]:
                tentative_g_cost = g_cost[current] + matrix[new_node]
                if tentative_g_cost < g_cost[new_node]:
                    parent_map[new_node] = current
                    g_cost[new_node] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic_distance(new_node, end) * 0.01
                    heapq.heappush(open_set, (f_cost, new_node))

    return None

@OPTIMIZER.register()
class CollisionOptimizer(Optimizer):

    def __init__(self, cfg: DictConfig,  *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type
        
        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.clip_grad_by_value = cfg.clip_grad_by_value
        self.collision_type = cfg.collision_type
        self.guidance = cfg.guidance

        self.d_class = 0
        self.d_bbox = 0
        self.d_shape = 0
        self.dataset = None


    def get_mesh_points_and_faces(self,mesh,device):
        try:
            points,faces = mesh.to_points_and_faces()
        except:
            mesh_cnt = len(mesh.renderables)
            points = []
            faces = []
            point_cnt = 0
            for s in range(mesh_cnt):
                p,f = mesh.renderables[s].to_points_and_faces()
                points.append(p)
                faces.append(f+point_cnt)
                point_cnt += p.shape[0]
            points = np.concatenate(points,axis=0)
            faces = np.concatenate(faces,axis=0)

        verts = torch.tensor(points,device = device).unsqueeze(0)
        faces = torch.tensor(faces,device = device).long()
        return verts,faces
    
    def calc_occupancy(self, bbox_cur, index, device):
        #B,N,C
        bbox_cur_cnt = bbox_cur["class_labels"].shape[1] 
        occupancy_map = torch.ones([bbox_cur_cnt,bbox_cur_cnt],device = device)
        object_flag = torch.ones(bbox_cur_cnt)
        # get mesh
        boxes = bbox_cur
        bbox_params = np.concatenate([
                boxes["class_labels"].detach().cpu().numpy(),
                boxes["translations"].detach().cpu().numpy(),
                boxes["sizes"].detach().cpu().numpy(),
                boxes["angles"].detach().cpu().numpy(),
                boxes["objfeats_32"] #add 
            ], axis=-1)[index:index+1,:,:]
        classes = np.array(self.dataset.class_labels)
        renderables, _,_, renderables_remesh,_ = get_textured_objects(
            bbox_params, self.objects_dataset, self.gapartnet_dataset, classes, self.cfg
        )
        #get empty object
        class_num = self.cfg.task.dataset.class_num
        for j in range(bbox_params.shape[1]):
            query_label = classes[bbox_params[0, j, :class_num].argmax(-1)] 
            if query_label=='start' or query_label == 'end' or query_label == 'empty' :
                occupancy_map[j,:] = 0
                occupancy_map[:,j] = 0
                object_flag[j] = 0
          
        #get voxel
        voxels = []
        for mesh in renderables_remesh:
            voxel = voxel_grid_from_mesh(mesh,device,gridsize = 0.01) #1cm
            voxels.append(voxel)
        
        for i in range(bbox_cur_cnt):
            if object_flag[i] == 0:
                continue
            for j in range(bbox_cur_cnt):
                if object_flag[j] == 0:
                    continue
                if i==j:
                    occupancy_map[i,j] = 0
                    continue
                #valid mesh index
                a = int(torch.sum(object_flag[:i+1]))-1
                b = int(torch.sum(object_flag[:j+1]))-1
                
                if not bbox_overlap(renderables_remesh[a],renderables[b]):
                    occupancy_map[i,j] = 0
                    continue
                else:
                    points = voxels[a][None,:,:]
                    pointscuda = torch.tensor(points,device = device)
                    verts,faces = self.get_mesh_points_and_faces(renderables_remesh[b],device)

                    #visualize
                    visualize = False
                    if visualize:
                        mesh = o3d.geometry.TriangleMesh()
                        mesh.vertices = o3d.utility.Vector3dVector(verts[0].cpu().numpy())
                        mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
                        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                        pcd0 = o3d.geometry.PointCloud()
                        pcd0.points = o3d.utility.Vector3dVector(points[0])
                        pcd0.colors = o3d.utility.Vector3dVector([[0,1,0]])
                        from scripts.eval.calc_ckl import draw_box_label
                        draw_box_label(pcd0,None,None, mesh)

                    occupancy = check_sign(verts,faces,pointscuda)
                    #no collision
                    if occupancy.max()<=0:
                        occupancy_map[i,j] = 0
                        continue
                    else:
                        col_area = occupancy.sum()
                        # this is not iou,
                        # (A&B)/A, not (A&B)/(AUB)
                        occupancy_map[i,j] = col_area/points.shape[1] 

        return occupancy_map, object_flag
    
    def calc_open_bbox(self,bbox_target,class_label_target):
        # bbox_target shape: #1,7

        classes = np.array(self.dataset.class_labels)
        isArticulated = check_articulate(class_label_target,classes)
        if not isArticulated:
            return bbox_target
        
        box = bbox_target[0]
        centerxy = box[:2]
        angle = box[-1]
        angle = angle.detach().cpu().numpy()
        size = box[3:6]
        h = size[1]  
        # here we take h/3 as the open ratio
        open_size = h/3
        
        center_bias = np.array([[0,open_size.cpu().data/2]])  
        R = np.zeros((2, 2))
        R[0, 0] = np.cos(angle)
        R[0, 1] = -np.sin(angle)
        R[1, 0] = np.sin(angle)
        R[1, 1] = np.cos(angle)
        center_bias = center_bias.dot(R) 
        center_bias=torch.tensor(center_bias).cuda()

        
        bias = torch.zeros_like(bbox_target).cuda()
        bias[:,:2] = center_bias
        bias[:,4] = open_size
        bbox_target_new = bbox_target + bias
        
        return bbox_target_new

    def collision_loss(self,boxes,bbox,objectness,class_labels):
        print("Calculating Collision Guidance ...")
        device = bbox.device
        loss_collision = 0.0
         
        # [1] bbox IOU
        if self.collision_type == "bbox_IOU":
            # B,N,C, # batch,obj_num,featdim
            for j in trange(len(bbox)):
                
                bbox_cur = bbox[j:j+1,:,:]
                class_labels_cur = class_labels[j:j+1,:,:]
                objectness_cur = objectness[j:j+1,:,:]

                bbox_cur = bbox_cur[:,objectness_cur[0,:,0],:]
                class_labels_cur = class_labels_cur[:,objectness_cur[0,:,0],:]

                bbox_cur_cnt = bbox_cur.shape[1] 
                for i in range(bbox_cur_cnt):    
                    bbox_target = bbox_cur[:,i,:]  #1,7
                    class_label_target = class_labels_cur[:,i,:] 
                    #calc opened bbox
                    if self.guidance.open:
                        bbox_target = self.calc_open_bbox(bbox_target,class_label_target)
                    bbox_target = torch.tile(bbox_target[:,None,:],[1,bbox_cur_cnt,1])   #1,12,7
                    loss_iter = cal_iou_3d(bbox_cur,bbox_target) #obj_num,obj_num
                    valid_pair = torch.ones_like(loss_iter).int()
                    valid_pair[:,i] = 0
                    loss_iter = loss_iter*valid_pair  
                    loss_collision += loss_iter.sum()/bbox_cur_cnt/len(bbox)
            loss_collision = loss_collision*0.075
        

        # [2] mesh occupancy
        elif self.collision_type == "mesh_occupancy":
            # #B,N,C, # 128,12,featdim
            obj_cnt = bbox.shape[1]
            for j in trange(len(bbox)):
                bbox_cur = bbox[j:j+1,:,:]
                # occupancy_map = torch.ones([obj_cnt,obj_cnt],device = device).int()
                # object_flag = torch.ones(obj_cnt)
                occupancy_map, object_flag = self.calc_occupancy(boxes,j,device)
                obj_valid_cnt = torch.sum(object_flag)
                for i in range(obj_cnt):    
                    bbox_target = bbox_cur[:,i,:]  #1,7
                    bbox_target = torch.tile(bbox_target[:,None,:],[1,obj_cnt,1])   #1,12,7
                    loss_iter = cal_iou_3d(bbox_cur,bbox_target) #128,12
                    valid_pair = torch.ones_like(loss_iter)
                    valid_pair[:,:] = occupancy_map[i:i+1,:]
                    # valid_pair[valid_pair>0] = 1
                    # valid_pair1[:,i] = 0
                    loss_iter = loss_iter*valid_pair
                    loss_collision += loss_iter.sum()/obj_valid_cnt/len(bbox)
            loss_collision = loss_collision*5
            
        else:
            print("error in loading collision_type : ", self.collision_type)
            assert( self.collision_type not in ["bbox_IOU","mesh_occupancy"])
        
        return loss_collision


    def room_outer_loss(self, bbox, room_outer_box, objectness):
        print("Calculating Room-layout Guidance ...")
        loss_room_outer = 0.0
        #xyz
        bbox_outer = room_outer_box
        bbox_cnt_room = bbox_outer.shape[1]

        for j in trange(len(bbox)):
            bbox_cur = bbox[j:j+1,:,:]
            objectness_cur = objectness[j:j+1,:,:]
            bbox_cur = bbox_cur[:,objectness_cur[0,:,0],:]
            bbox_cur_cnt = bbox_cur.shape[1] 
            bbox_outer_cur = bbox_outer[j:j+1,:,:]
            for i in range(bbox_cur_cnt):
                bbox_target = bbox_cur[:,i,:]  #1,7
                bbox_target = torch.tile(bbox_target[:,None,:],[1,bbox_cnt_room,1])   #1,12,7
                loss_room_outer += cal_iou_3d(bbox_outer_cur,bbox_target).sum()/len(bbox)/bbox_cur_cnt #1,12
        return loss_room_outer

    def optimize(self, x: torch.Tensor, data, room_outer_box=None, doors=None, floor_plan=None, floor_plan_centroid=None, objectness=None) -> torch.Tensor:

        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data
        
        Return:
            The optimizer objective value of current step
        """
        
        ### iou loss
        boxes = self.post_process(x) #post processed data

        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]*2
        sizes[sizes<0] = 0
        angles = boxes["angles"]
        
        bbox = torch.cat([translations,sizes,angles],dim=-1)
        # bbox[invalid_bbox[:,:,:7]==1] = 0

        # permute (x,z,y,w,l,h,alpha)->(x,y,z,w,h,l,alpha)
        bbox[:,:,1] = translations[:,:,2]
        bbox[:,:,2] = translations[:,:,1]
        bbox[:,:,4] = sizes[:,:,2]  #z
        bbox[:,:,5] = sizes[:,:,1]  #y

        #collision loss
        loss_collision = 0
        if self.guidance.collision:
            loss_collision = self.collision_loss(boxes,bbox,objectness,class_labels)

        #room_outer_box loss
        loss_room_outer = 0.
        if self.guidance.room_layout:
            loss_room_outer = self.room_outer_loss(bbox, room_outer_box, objectness)

        # walkable loss
        loss_walkable = 0.
        if self.guidance.reachability:
            loss_walkable = self.walkable_loss(bbox,objectness,floor_plan,floor_plan_centroid,class_labels)

        loss = loss_collision*self.guidance.weight_coll + \
                loss_room_outer*self.guidance.weight_room + \
                loss_walkable*self.guidance.weight_reach 
        
        return (-1.0) * loss

    def walkable_loss(self,bbox,objectness,floor_plan,floor_plan_centroid,class_labels):
        print("Calculating Reachability Guidance ...")
        loss_walkable = 0.0
        robot_width_real = self.guidance.robot_width_real
        robot_hight_real = self.guidance.robot_hight_real
        
        # wakable loss can not be parallelized. 
        # larger batch size will cause longer time.
        for i in trange(len(bbox)):
            bbox_cur = bbox[i:i+1,:,:]
            objectness_cur = objectness[i:i+1,:,:]
            bbox_cur = bbox_cur[:,objectness_cur[0,:,0],:]
            class_labels_cur = class_labels[i:i+1,objectness_cur[0,:,0],:]
            bbox_cur_cnt = bbox_cur.shape[1] 

            vertices, faces = floor_plan[i]
            vertices = vertices - floor_plan_centroid[i]
            vertices = vertices[:, 0::2]
            scale = np.abs(vertices).max()+0.2
            #remove objects in a high level, such as ceiling lamp
            bbox_floor = bbox_cur[0, bbox_cur[0, :, 2] < robot_hight_real]  
            class_labels_cur = class_labels_cur[0, bbox_cur[0, :, 2] < robot_hight_real]

            image_size = 256
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

            robot_width = int(robot_width_real / scale * image_size/2)

            def map_to_image_coordinate(point):
                x, y = point
                x_image = int(x / scale * image_size/2)+image_size/2
                y_image = int(y / scale * image_size/2)+image_size/2
                return x_image, y_image
            
            def image_to_map_coordinate(point):
                x, y = point
                x_map = (x - image_size/2) * 2 / image_size *scale
                y_map = (y - image_size/2) * 2 / image_size *scale
                return x_map, y_map

            # draw floor plan
            for face in faces:
                face_vertices = vertices[face]
                face_vertices_image = [map_to_image_coordinate(v) for v in face_vertices]
                pts = np.array(face_vertices_image, np.int32)
                pts = pts.reshape(-1, 1, 2)
                color = (255, 0, 0)  # Blue (BGR)
                cv2.fillPoly(image, [pts], color)

            kernel = np.ones((robot_width, robot_width))
            image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
            # draw bboxes
            floor_plan_mask = image[:, :, 0]==255

            box_heat_map = np.zeros((image_size, image_size), dtype=np.uint8)
            handle_lst = []
            for box,class_label in zip(bbox_floor,class_labels_cur):
                classes = np.array(self.dataset.class_labels)
                isArticulated = check_articulate(class_label,classes)
                box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                box = box.cpu().detach().numpy()
                center = map_to_image_coordinate(box[:2])
                # full size
                size = (int(box[3] / scale * image_size / 2),
                        int(box[4] / scale * image_size / 2))
                angle = box[-1]

            
                # calculate box vertices
                if not isArticulated:
                    # # [1] #original box
                    box_points = cv2.boxPoints(
                        ((center[0], center[1]), size, -angle/np.pi*180))
                else:
                    # [2] #open box
                    w,h = size  #x,z, z is front
                    open_size = h/3
                    rot_center = (center[0], center[1])
                    handle = np.array([0,h/2+open_size+robot_width/2+1])
                    bbox_expand = np.array([[-w/2,-h/2],  #left bottom
                                    [-w/2,h/2+open_size],  #left top  -open
                                    [w/2,h/2+open_size],  #right top -open
                                    [w/2,-h/2]])  #right down
                    R = np.zeros((2, 2))
                    R[0, 0] = np.cos(angle)
                    R[0, 1] = -np.sin(angle)
                    R[1, 0] = np.sin(angle)
                    R[1, 1] = np.cos(angle)
                    box_points = bbox_expand.dot(R) + rot_center
                    # here we simply take the front-center point as the handle point 
                    handle_point = handle.dot(R) + rot_center
                    handle_point = [int(handle_point[0]),int(handle_point[1])]
                    if 0<=handle_point[1]<image_size and 0<=handle_point[0]<image_size:
                        handle_lst.append(handle_point)
                

                box_points = np.intp(box_points)

                cv2.drawContours(image, [box_points], 0,
                                (0, 255, 0), robot_width)  # Green (BGR)
                
                cv2.fillPoly(image, [box_points], (0, 255, 0))
                cv2.drawContours(box_mask, [box_points], 0,
                                (0, 255, 0), robot_width)  # Green (BGR)
                
                cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
                box_mask = box_mask[:,:,1]==255
                if min(size)!=0:
                    gaussian = draw_2d_gaussian((int(center[0]), int(center[1])), size, -angle, image_size)
                    box_heat_map = box_heat_map + gaussian*box_mask
                # box_heat_map = box_heat_map + gaussian
                
            box_heat_map = floor_plan_mask * box_heat_map

            # add wall boundary 
            wall_dist_transform = cv2.distanceTransform(floor_plan_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
            wall_dist_transform[wall_dist_transform!=0] = 1./ wall_dist_transform[wall_dist_transform!=0]
            box_wall_heat_map = box_heat_map #+ wall_dist_transform
            box_wall_heat_map = box_wall_heat_map + (1-floor_plan_mask)*box_heat_map.max()
            
            # visual
            box_wall_heat_map_image = cv2.normalize(box_wall_heat_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            box_wall_heat_map_image = cv2.applyColorMap(box_wall_heat_map_image,cv2.COLORMAP_JET)
            # cv2.imwrite("mesh_image_with_boxes2.png", box_wall_heat_map_image)
            # cv2.imwrite("mesh_image_with_boxes1.png", image)

            walkable_map = image[:, :, 0].copy()
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                walkable_map, connectivity=8)
            
     
            # [1] region reachable,
            # find shortest path from region_1 to region_2
            if num_labels>2:
                area_1 = np.zeros_like(walkable_map)
                area_2 = np.zeros_like(walkable_map)
                for label in range(1, num_labels):
                    mask = np.zeros_like(walkable_map)
                    mask[labels == label] = 1
                    if mask.sum()>area_2.sum():
                        area_2 = mask.copy()
                    if area_2.sum()>area_1.sum():
                        area_2, area_1 = area_1.copy(), area_2.copy()
                if area_2.sum()>100:
                    minimum_area_1 = np.argmax(cv2.distanceTransform(area_1.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5))
                    minimum_area_1_position = np.unravel_index(minimum_area_1, area_1.shape)
                    minimum_area_2 = np.argmax(cv2.distanceTransform(area_2.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5))
                    minimum_area_2_position = np.unravel_index(minimum_area_2, area_2.shape)
                    
                    shortest_path_image = box_wall_heat_map_image.copy()
                    #draw circle for start and end point
                    cv2.circle(shortest_path_image, [minimum_area_1_position[1],minimum_area_1_position[0]], 2, (255, 255, 255), -1)
                    cv2.circle(shortest_path_image, [minimum_area_2_position[1],minimum_area_2_position[0]], 2, (255, 255, 255), -1)
                    # cv2.imwrite('walkable_map.png', shortest_path_image)
                    shortest_path = find_shortest_path(box_wall_heat_map, (minimum_area_1_position[0],minimum_area_1_position[1]),
                                    (minimum_area_2_position[0],minimum_area_2_position[1]),)
                    if shortest_path == None:
                        continue
                    for r, c in shortest_path:
                        shortest_path_image[r, c] = (255, 255, 255)  # white

                    # cv2.imwrite('shortest_path_image.png', shortest_path_image)
                    loss_walkable += self.calc_loss_on_path(image,shortest_path,robot_width,robot_width_real,robot_hight_real,
                                                            map_to_image_coordinate, image_to_map_coordinate,
                                                            scale,image_size,bbox,bbox_floor)

            # [2] handle reachable
            # find shortest path from region_1 to handle_1(front center point of the open bbox)
            if len(handle_lst)>0:
                shortest_path_image = box_wall_heat_map_image.copy()
                area_1 = np.zeros_like(walkable_map)
                for label in range(1, num_labels):
                    mask = np.zeros_like(walkable_map)
                    mask[labels == label] = 1
                    if mask.sum()>area_1.sum():
                        area_1 = mask.copy()
                #start point
                minimum_area_1 = np.argmax(cv2.distanceTransform(area_1.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5))
                minimum_area_1_position = np.unravel_index(minimum_area_1, area_1.shape)
                cv2.circle(shortest_path_image, [minimum_area_1_position[1],minimum_area_1_position[0]], 2, (255, 255, 255), -1)
                for handle_point in handle_lst:
                    # # [2] #
                    # handle_point = random.choice(handle_lst)
                    #find the largest area
                    
                    if area_1.sum()>100:
                        #end point
                        cv2.circle(shortest_path_image, [int(handle_point[0]),int(handle_point[1])], 2, (255, 255, 255), -1)
                        shortest_path = find_shortest_path(box_wall_heat_map, (minimum_area_1_position[0],minimum_area_1_position[1]),
                                        (int(handle_point[1]),int(handle_point[0])),)
                        if shortest_path == None:
                            continue
                        for r, c in shortest_path:
                            shortest_path_image[r, c] = (255, 255, 255)  # white
                        loss_walkable += self.calc_loss_on_path(image,shortest_path,robot_width,robot_width_real,robot_hight_real, 
                                                                map_to_image_coordinate, image_to_map_coordinate, 
                                                                scale,image_size,bbox,bbox_floor)
                # cv2.imwrite('shortest_path_image.png', shortest_path_image)

        return loss_walkable

    def calc_loss_on_path(self,image,shortest_path,robot_width,robot_width_real,robot_hight_real,
                          map_to_image_coordinate, image_to_map_coordinate,
                          scale,image_size,bbox,bbox_floor):
        loss_walkable = 0.0
        box_mask = image[:,:,1] == 255
        bbox_path = []
        path_count = 0
        for i in range(len(shortest_path)):
            if box_mask[shortest_path[i]]:
                if path_count%robot_width==0:
                    
                    center_map = image_to_map_coordinate((shortest_path[i][1], shortest_path[i][0]))
                    angle = 0.
                    box = np.array([*center_map, 0, robot_width_real, robot_width_real, robot_hight_real, angle])
                    bbox_path.append(box)
                
                path_count+=1
        if bbox_path == []:
            return loss_walkable
        for box in bbox_path:
            center = map_to_image_coordinate(box[:2])
            size = (int(box[3] / scale * image_size / 2),
                    int(box[4] / scale * image_size / 2))
            angle = box[-1]

            # calculate box vertices
            box_points = cv2.boxPoints(
                ((center[0], center[1]), size, -angle/np.pi*180))
            box_points = np.intp(box_points)

            cv2.drawContours(image, [box_points], 0,
                            (0, 255, 255), robot_width)  # Yellow
        # cv2.imwrite("shortest_path_box_image.png", image)
        bbox_path = np.expand_dims(np.stack(bbox_path, 0),0)

        #xyz
        bbox_cnt_path = bbox_path.shape[1]
        bbox_floor = bbox_floor[None,:,:]
        bbox_floor_cur_cnt = bbox_floor.shape[1]
        for bbox_cnt_idx in range(bbox_floor_cur_cnt):    
            bbox_target = bbox_floor[:,bbox_cnt_idx,:]
            bbox_target = torch.tile(bbox_target[:,None,:],[1,bbox_cnt_path,1])
            
            loss_walkable += cal_iou_3d(torch.tensor(bbox_path,device=bbox_target.device,dtype=bbox_target.dtype),bbox_target).sum()/len(bbox)/bbox_floor_cur_cnt
        return loss_walkable

    def post_process(self, pred_x):
        feat = pred_x[:,:,self.d_bbox+self.d_class:].cpu().detach().numpy()
        bbox_params = {
                        "class_labels": pred_x[:,:,self.d_bbox:self.d_bbox+self.d_class],
                        "translations": pred_x[:,:,:3],
                        "sizes": pred_x[:,:,3:6],
                        "angles":  pred_x[:,:,6:self.d_bbox],  #TODO
                        "objfeats_32": feat
                    }
        boxes = self.dataset.post_process_cuda(bbox_params)
        return boxes

    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor, room_outer_box=None, doors=None, floor_plan=None, floor_plan_centroid=None, objectness=None) -> torch.Tensor:
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step
            data: data dict that provides original data
            variance: variance at current step
        
        Return:
            Commputed gradient
        """
        
        
        #[translations, sizes, angles, class_labels, objectness]
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)  

            with torch.autograd.set_detect_anomaly(True):
                obj = self.optimize(x_in, data, room_outer_box, doors=doors, floor_plan=floor_plan, floor_plan_centroid=floor_plan_centroid, objectness=objectness)
                if obj == 0:
                    return None
                grad = torch.autograd.grad(obj, x_in)[0]
                #only keep gradient of translation x,y
                grad[:,:,1] = 0  #z
                grad[:,:,3:6] = 0  #sizes
                grad[:,:,8:] = 0 
                ## clip gradient by value
                grad = torch.clip(grad, **self.clip_grad_by_value)
                ## TODO clip gradient by norm
            
            if self.scale_type == 'normal':
                grad = self.scale * grad * variance
            elif self.scale_type == 'div_var':
                grad = self.scale * grad
            else:
                raise Exception('Unsupported scale type!')

            return grad
        
