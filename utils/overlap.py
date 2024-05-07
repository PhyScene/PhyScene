import numpy as np
import torch 
import cv2
from scipy.ndimage import binary_dilation
from scripts.eval.walkable_metric import cal_walkable_metric
from kaolin.ops.mesh import check_sign

def voxel_grid_from_mesh(mesh,device,gridsize = 0.01):
    #mesh
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

    #grid 
    #[xmin,ymin,zmin]
    #[xmax,ymax,zmax]
    box = mesh.bbox 
    xmin,ymin,zmin = box[0]
    xmax,ymax,zmax = box[1]
    points = []
    
    xcnt = round((xmax-xmin)/gridsize+1)
    ycnt = round((ymax-ymin)/gridsize+1)
    zcnt = round((zmax-zmin)/gridsize+1)
    for i in range(xcnt):
        x = xmin + (xmax-xmin)*i/(xcnt-1)
        for j in range(ycnt):
            y = ymin+(ymax-ymin)*j/(ycnt-1)
            for k in range(zcnt):
                z = zmin+(zmax-zmin)*k/(zcnt-1)
                points.append(np.array([[x,y,z]]))
    
    points = np.concatenate(points,axis=0)[None,:,:]
    pointscuda = torch.tensor(points,device = device)
    occupancy = check_sign(verts,faces,pointscuda)

    points_inside = points[0][occupancy.cpu().numpy()[0]==1]

    # #vis
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_inside)
    # app = gui.Application.instance
    # app.initialize()
    # vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    # vis.show_settings = True
    # vis.add_geometry("pcd0",pcd)
   

    # vis.reset_camera_to_default()
    # app.add_window(vis)
    # app.run()
    return points_inside

def bbox_overlap(render1,render2):
    box1 = np.array(render1.bbox)
    box2 = np.array(render2.bbox)
    if box1[0,0]>=box2[1,0] or box2[0,0]>=box1[1,0] : #xmin>xmax
        return False
    if box1[0,1]>=box2[1,1] or box2[0,1]>=box1[1,1] : #y
        return False
    if box1[0,2]>=box2[1,2] or box2[0,2]>=box1[1,2] : #z
        return False
    
    return True




def calc_overlap_rotate_bbox(synthesized_scenes):
    collision_box_count = 0
    box_count = 0
    for d in synthesized_scenes:

        valid_idx = d["objectness"][:,0]<0 #False

        
        translations = torch.from_numpy(d["translations"][valid_idx]).cuda().type(torch.float)
        sizes = torch.from_numpy(d["sizes"][valid_idx]*2).cuda().type(torch.float)
        sizes[sizes<0] = 0
        angles = torch.from_numpy(d["angles"][valid_idx]).cuda().type(torch.float)

        bbox = torch.cat([translations,sizes,angles],dim=-1)

        #(x,z,y,w,l,h,alpha)->(x,y,z,w,h,l,alpha)
        bbox[:,1] = translations[:,2]
        bbox[:,2] = translations[:,1]
        bbox[:,4] = sizes[:,2]
        bbox[:,5] = sizes[:,1]
        bbox= bbox[None,:,:]
        bbox_cnt = bbox.shape[1]
        from models.loss.oriented_iou_loss import cal_iou_3d
        for i in range(bbox_cnt):     
            bbox_target = bbox[:,i,:]  #128,7
            bbox_target = torch.tile(bbox_target[:,None,:],[1,bbox_cnt,1])   #128,12,7

            loss_iter = cal_iou_3d(bbox,bbox_target) #128,12
            valid_pair = torch.ones_like(loss_iter).int()
            valid_pair[:,i] = 0
            loss_iter = loss_iter*valid_pair  
            collision_box_count += loss_iter.sum()>0
        box_count +=bbox_cnt
    box_box_rate = collision_box_count/box_count
    print('box_box_collision_rate:', (box_box_rate).item())
    return box_box_rate


def calc_overlap_rotate_bbox_doors(synthesized_scenes,door_boxes):
    collision_box_count = 0
    box_count = 0
    for d in synthesized_scenes:

        valid_idx = d["objectness"][:,0]<0 #False

        
        translations = torch.from_numpy(d["translations"][valid_idx]).cuda().type(torch.float)
        sizes = torch.from_numpy(d["sizes"][valid_idx]*2).cuda().type(torch.float)
        sizes[sizes<0] = 0
        angles = torch.from_numpy(d["angles"][valid_idx]).cuda().type(torch.float)

        bbox = torch.cat([translations,sizes,angles],dim=-1)

        #(x,z,y,w,l,h,alpha)->(x,y,z,w,h,l,alpha)
        bbox[:,1] = translations[:,2]
        bbox[:,2] = translations[:,1]
        bbox[:,4] = sizes[:,2]
        bbox[:,5] = sizes[:,1]
        bbox= bbox[None,:,:]
        bbox_cnt = bbox.shape[1]
        from models.loss.oriented_iou_loss import cal_iou_3d
        for i in range(door_boxes.shape[1]):     
            bbox_target = door_boxes[:,i,:]  #128,7
            bbox_target = torch.tile(bbox_target[:,None,:],[1,bbox_cnt,1])   #128,12,7
            loss_iter = cal_iou_3d(bbox,bbox_target) #128,12
            collision_box_count += loss_iter.sum()>0
        box_count +=bbox_cnt
    box_box_rate = collision_box_count/box_count
    print('door_box_collision_rate:', (box_box_rate).item())
    return box_box_rate

def calc_bbox_masks(bbox,class_labels,image,image_size,scale,robot_width,floor_plan_mask, box_wall_count,classes): 
    """
    type: ["bbox","front_line", "front_center"]
    """

    box_masks = []
    handle_points = []
    for box,class_label in zip(bbox,class_labels):
        box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        box_wall_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]

        # calculate box vertices
        from models.optimizer.collision import check_articulate
        isArticulated = check_articulate(class_label,classes)
        if not isArticulated:
            box_points = cv2.boxPoints(
                ((center[0], center[1]), size, -angle/np.pi*180))
        else:
            w, h = size
            # open_size = h/3
            open_size = 0
            rot_center = center
            handle = np.array([0, h/2+open_size+robot_width/2+1])
            bbox = np.array([[-w/2, -h/2],
                            [-w/2, h/2+open_size],
                            [w/2, h/2+open_size],
                            [w/2, -h/2],
                            ])
            
            R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

            box_points = bbox.dot(R) + rot_center
            handle_point = handle.dot(R) + rot_center
            handle_points.append(handle_point)
        box_points = np.intp(box_points)
        # box wall collision
        cv2.fillPoly(box_wall_mask, [box_points], (0, 255, 0))
        # cv2.imwrite("debug1.png", box_wall_mask)
        box_wall_mask = box_wall_mask[:,:,1]==255
        if (box_wall_mask*(1-floor_plan_mask)).sum()>0:
            box_wall_count+=1
        # cv2.imwrite("debug2.png", floor_plan_mask)

        #image connected region
        cv2.drawContours(image, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))
        
        # per box mask
        cv2.drawContours(box_mask, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
        st_element = np.ones((3, 3), dtype=bool)
        box_mask = binary_dilation((box_mask[:, :, 1].copy()==255).astype(image.dtype), st_element)
        box_masks.append(box_mask)
        # cv2.imwrite("debug.png", box_mask)
    return box_masks, handle_points, box_wall_count, image

def map_to_image_coordinate(point, scale, image_size):
    x, y = point
    x_image = int(x / scale * image_size/2)+image_size/2
    y_image = int(y / scale * image_size/2)+image_size/2
    return x_image, y_image

def calc_wall_overlap(synthesized_scenes, floor_plan_lst, floor_plan_centroid_list, cfg, robot_real_width=0.3,calc_object_area=False,classes=None):
    
    box_wall_count = 0
    accessable_count = 0
    box_count = 0
    walkable_metric_list = []
    accessable_rate_list = []
    accessable_handle_rate_list = []
    # from tqdm import tqdm

    # for i in tqdm(range(len(synthesized_scenes))):
    for i in range(len(synthesized_scenes)):
        d = synthesized_scenes[i]
        floor_plan = floor_plan_lst[i]
        floor_plan_centroid = floor_plan_centroid_list[i]
        valid_idx = d["objectness"][:,0]<0 #False
        if cfg.task.dataset.use_feature:
            class_labels = d["class_labels"]
            bbox = np.concatenate([
                        # d["class_labels"],
                        d["translations"][valid_idx],
                        d["sizes"][valid_idx],
                        d["angles"][valid_idx],
                        # d["objfeats_32"]
                    ],axis=-1)
        else:
            class_labels = d["class_labels"]
            bbox = np.concatenate([
                        # d["class_labels"],
                        d["translations"],
                        d["sizes"],
                        d["angles"]
                    ],axis=-1)
            
        vertices, faces = floor_plan
        vertices = vertices - floor_plan_centroid
        # vertices = vertices[:, 0::2]
        vertices = vertices[:, :2]
        scale = np.abs(vertices).max()+0.2

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        robot_width = int(robot_real_width / scale * image_size/2)

        
        
        # draw face
        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [
                map_to_image_coordinate(v,scale, image_size) for v in face_vertices]

            pts = np.array(face_vertices_image, np.int32)
            pts = pts.reshape(-1, 1, 2)
            color = (255, 0, 0)  # Blue (BGR)
            cv2.fillPoly(image, [pts], color)
        
        floor_plan_mask = (image[:,:,0]==255)*255
        # cv2.imwrite("debug_floor.png", floor_plan_mask)
        # 缩小墙边界，机器人行动范围
        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        
        box_masks, handle_points, box_wall_count, image = calc_bbox_masks(bbox,class_labels,image,image_size,scale,robot_width,floor_plan_mask, box_wall_count,classes=classes)
        # cv2.imwrite("debug.png", image)
        # breakpoint()
        walkable_map = image[:, :, 0].copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walkable_map, connectivity=8)
        # 遍历每个连通域

        accessable_rate = 0
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            mask[labels == label] = 1
            accessable_count = 0
            for box_mask in box_masks:
                if (box_mask*mask).sum()>0:
                    accessable_count += 1
            accessable_rate += accessable_count/len(box_masks)*mask.sum()/(labels!=0).sum()
        accessable_rate_list.append(accessable_rate)
        box_count += len(box_masks)
        
        accessable_handle_rate = 0
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            mask[labels == label] = 1
            accessable_handle_count = 0
            for handle_point in handle_points:
                # breakpoint()
                handle_point_int = np.round(handle_point).astype(int)
                if 0<=handle_point_int[1]<image_size and 0<=handle_point_int[0]<image_size and mask[handle_point_int[0], handle_point_int[1]]>0:
                    accessable_handle_count+=1
            accessable_handle_rate += accessable_handle_count/(len(handle_points)*mask.sum()+0.00001)/(labels!=0).sum()
        accessable_handle_rate_list.append(accessable_handle_rate)

        #walkable map area rate
        if calc_object_area:
            walkable_rate, object_area_ratio = cal_walkable_metric(floor_plan, floor_plan_centroid, bbox, doors=None, robot_width=0.3, visual_path=None,calc_object_area=True)
        else:
            walkable_rate = cal_walkable_metric(floor_plan, floor_plan_centroid, bbox, doors=None, robot_width=0.3, visual_path=None,)
        walkable_metric_list.append(walkable_rate)

        # breakpoint()
    # print('walkable_metric_list:', walkable_metric_list)
    walkable_average_rate = sum(walkable_metric_list)/len(walkable_metric_list)
    accessable_rate = sum(accessable_rate_list)/len(accessable_rate_list)
    accessable_handle_rate = sum(accessable_handle_rate_list)/len(accessable_handle_rate_list)
    box_wall_rate = box_wall_count/box_count
    
    print('walkable_average_rate:', walkable_average_rate)
    print('accessable_rate:', accessable_rate)
    print('accessable_handle_rate:', accessable_handle_rate)
    print('box_wall_rate:', box_wall_rate)
    if calc_object_area:
        print('object_area_ratio:', object_area_ratio)
        return walkable_average_rate, accessable_rate,box_wall_rate, object_area_ratio
    else:
        return walkable_average_rate, accessable_rate,box_wall_rate

