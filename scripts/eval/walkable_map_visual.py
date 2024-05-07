import numpy as np
import cv2
"""Script for visualizing walkable map. """
# robot_width_real = 0.1
robot_hight_real = 1.5
IMAGE_SIZE = 1024
def walkable_map_visual(bbox, floor_plan, floor_plan_centroid, scale, floor_render, robot_width_real = 0.3,path_to_walk=None):

    bbox = bbox
    vertices, faces = floor_plan
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max() / scale
    bbox_floor = bbox[bbox[:, 1] < robot_hight_real]

    image_size = IMAGE_SIZE
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
    # cv2.imwrite(path_to_walk, image)

    # floor_plan_mask = image[:, :, 0]==255

    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    # draw bboxes
    floor_plan_mask = image[:, :, 0]==255

    # box_heat_map = np.zeros((image_size, image_size), dtype=np.uint8)
    box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    box_mask_erode = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for box in bbox_floor:
        center = map_to_image_coordinate(box[:3][0::2])
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(
            ((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)

        cv2.drawContours(box_mask_erode, [box_points], 0,
                        (255, 255, 255), 2)  
        
        cv2.fillPoly(box_mask_erode, [box_points], (128, 128, 128))
        cv2.drawContours(box_mask, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        
        cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
        cv2.drawContours(image, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        
        cv2.fillPoly(image, [box_points], (0, 255, 0))
        # box_mask = box_mask[:,:,1]==255
        # gaussian = draw_2d_gaussian((int(center[0]), int(center[1])), size, -angle, image_size)
        # breakpoint()
        # box_heat_map = box_heat_map + gaussian*box_mask
        # # box_heat_map = box_heat_map + gaussian
        
    # box_heat_map = floor_plan_mask * box_heat_map
    # cv2.imwrite(path_to_walk, image)


    walkable_map = image[:, :, 0].copy()
    # cv2.imwrite(path_to_walk, walkable_map)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8)
    # walkable_map_visual = np.zeros((image_size, image_size))
    # for label in range(1, num_labels):
    #     mask = np.zeros_like(walkable_map)
    #     mask[labels == label] = 1
        
    #     # add wall boundary 
    #     wall_dist_transform = cv2.distanceTransform(mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)

    #     walkable_map_visual[wall_dist_transform!=0] = wall_dist_transform[wall_dist_transform!=0]/wall_dist_transform.max()

    def random_move(x, y, room_mask):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1),(1, 1), (-1, 1), (-1, -1), (1, -1)]
        while True:
            dx, dy = directions[np.random.choice(4)]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < room_mask.shape[0] and 0 <= new_y < room_mask.shape[1] and room_mask[new_x, new_y] != 0:
                return new_x, new_y

    robot_trajectories = []
    walkable_map_visual = np.zeros_like(walkable_map).astype(np.uint16)
    # cv2.imwrite(path_to_walk, walkable_map_visual)
    mask = np.zeros_like(walkable_map)
    for label in range(1, num_labels):
        mask_cur = np.zeros_like(walkable_map)
        mask_cur[labels == label] = 1
        if mask.sum()<mask_cur.sum():
            mask = mask_cur.copy()
        

    start_position_idx = np.random.choice(np.arange(0,mask.sum()),int(mask.sum()//100),replace=False)
    for pos_idx in start_position_idx:
        x, y = np.where(mask==1)[0][int(pos_idx)], np.where(mask==1)[1][int(pos_idx)]
        trajectory = [(x, y)]
        # steps = int(mask.sum()/10)
        steps = 1000
        for _ in range(steps):
        # for _ in range(1000):
            x, y = random_move(x, y, mask)
            trajectory.append((x, y))
            walkable_map_visual[x][y]+=1
        robot_trajectories.append(trajectory)
        
    walkable_map_visual_image = cv2.normalize(np.sqrt(walkable_map_visual), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    walkable_map_visual_image = cv2.applyColorMap(walkable_map_visual_image,cv2.COLORMAP_JET)
    # cv2.imwrite(path_to_walk, walkable_map_visual_image)
    
    # box_mask_visual = box_mask_erode.sum(-1) != 0
    # walkable_map_visual_image[box_mask_visual==True] = box_mask_erode[box_mask_visual==True]
    alpha = 0.5
    floor_render = np.stack([floor_render[:,:,2], floor_render[:,:,1], floor_render[:,:,0]], -1)
    floor_render = cv2.resize(floor_render, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    walkable_map_visual_image[(box_mask[:,:,1]==255)] = floor_render[:, :, :3][(box_mask[:,:,1]==255)]
    walkable_map_visual_image[mask==False] = floor_render[:, :, :3][mask==False]
    final_visual = cv2.addWeighted(walkable_map_visual_image, alpha, floor_render[:, :, :3], 1-alpha, 0)
    print("saving walkable map in ",path_to_walk)
    cv2.imwrite(path_to_walk, final_visual)
    

def walkable_map_empty(image_size=IMAGE_SIZE,path_to_walk="debug.png"):

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8)

    def random_move(x, y, room_mask):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1),(1, 1), (-1, 1), (-1, -1), (1, -1)]
        while True:
            dx, dy = directions[np.random.choice(4)]
            new_x, new_y = x + dx, y + dy
            # return new_x, new_y
            if 0 <= new_x < room_mask.shape[0] and 0 <= new_y < room_mask.shape[1]:
                return new_x, new_y

    walkable_map_visual = np.zeros_like(walkable_map).astype(np.uint16)
    mask = np.zeros_like(walkable_map)
    for label in range(1, num_labels):
        mask_cur = np.zeros_like(walkable_map)
        mask_cur[labels == label] = 1
        if mask.sum()<mask_cur.sum():
            mask = mask_cur.copy()
        

    start_position_idx = np.random.choice(np.arange(0,mask.sum()),int(mask.sum()//100),replace=False)
    for x in range(image_size):
        for y in range(image_size):
            new_x = x
            new_y = y
            # steps = int(mask.sum()/10)
            steps = 2000
            for _ in range(steps):
            # for _ in range(1000):
                new_x, new_y = random_move(new_x, new_y, mask)
                if new_x<0 or new_y<0 or new_x>=image_size or new_y>=image_size:
                    break
                walkable_map_visual[new_x][new_y]+=1
    # walkable_map_visual = walkable_map_visual[1:-1,1:-1]
        
    walkable_map_visual_image = cv2.normalize(np.sqrt(walkable_map_visual), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    walkable_map_visual_image = cv2.applyColorMap(walkable_map_visual_image,cv2.COLORMAP_JET)
    
    # box_mask_visual = box_mask_erode.sum(-1) != 0
    # walkable_map_visual_image[box_mask_visual==True] = box_mask_erode[box_mask_visual==True]
    # alpha = 0.5

    # walkable_map_visual_image[mask==False] = floor_render[:, :, :3][mask==False]
    # final_visual = cv2.addWeighted(walkable_map_visual_image, alpha, floor_render[:, :, :3], 1-alpha, 0)
    cv2.imwrite(path_to_walk, walkable_map_visual_image)
    

if __name__ == "__main__":
    walkable_map_empty()