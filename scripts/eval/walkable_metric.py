import cv2
import numpy as np
import math
import trimesh

"""Script for calculating walkable metric. """

def cal_walkable_metric(floor_plan, floor_plan_centroid, bboxes, doors, robot_width=0.01, visual_path=None, calc_object_area=False):

    vertices, faces = floor_plan
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max()+0.2
    bboxes = bboxes[bboxes[:, 1] < 1.5]

    # door
    if doors is not None:

        doors_position = [door[0][door[0][:, 1] == door[0][:, 1].min()].mean(0) for door in doors]

    image_size = 256
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    robot_width = int(robot_width / scale * image_size/2)

    def map_to_image_coordinate(point):
        x, y = point
        x_image = int(x / scale * image_size/2)+image_size/2
        y_image = int(y / scale * image_size/2)+image_size/2
        return x_image, y_image

    # draw face
    for face in faces:
        face_vertices = vertices[face]
        face_vertices_image = [
            map_to_image_coordinate(v) for v in face_vertices]

        pts = np.array(face_vertices_image, np.int32)
        pts = pts.reshape(-1, 1, 2)
        color = (255, 0, 0)  # Blue (BGR)
        cv2.fillPoly(image, [pts], color)

    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    # draw bboxes
    # cv2.imwrite("image.png", image)
    for box in bboxes:
        center = map_to_image_coordinate(box[:3][0::2])
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(
            ((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)

        cv2.drawContours(image, [box_points], 0,
                         (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

    # cv2.imwrite("image.png", image)

    if calc_object_area:
        green_cnt = 0
        blue_cnt = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if list(image[i][j]) == [0, 255, 0]:
                    green_cnt += 1
                elif list(image[i][j]) == [255, 0, 0]:
                    blue_cnt += 1
        object_area_ratio = green_cnt/(blue_cnt+green_cnt)
        
    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8)

    if doors is not None:
        walkable_rate_list = []
        for door_position in doors_position:
            distance_to_door = np.inf
            door_position = door_position - floor_plan_centroid
            door_position = map_to_image_coordinate(
                np.array(door_position)[0::2])
            door_position = np.array(door_position, np.int32)
            cv2.circle(image, door_position, 5, (255, 255, 255), 5)
            if visual_path is not None:
                cv2.imwrite(visual_path, image)
            
            walkable_map_max = np.zeros_like(walkable_map)
            walkable_map_door = np.zeros_like(walkable_map)
            for label in range(1, num_labels):
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 255

                dist_transform = cv2.distanceTransform(
                    (255-mask).T, distanceType=cv2.DIST_L2, maskSize=5)
                distance_mask_to_door = dist_transform[door_position[0], door_position[1]]
                
                if distance_mask_to_door < distance_to_door and distance_mask_to_door <= robot_width + 1:
                    # room connected component with door
                    distance_to_door = distance_mask_to_door
                    walkable_map_door = mask.copy()
            walkable_rate = walkable_map_door.sum()/walkable_map.sum()
            walkable_rate_list.append(walkable_rate)
        # print("walkable_rate:", np.mean(walkable_rate_list))
        if calc_object_area:
            return np.mean(walkable_rate_list),object_area_ratio
        else:
            return np.mean(walkable_rate_list)
    else:
        walkable_map_max = np.zeros_like(walkable_map)
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            walkable_map_door = np.zeros_like(walkable_map)
            mask[labels == label] = 255

            if mask.sum() > walkable_map_max.sum():
                # room connected component with door
                walkable_map_max = mask.copy()

            # print("walkable_rate:", walkable_map_max.sum()/walkable_map.sum())
            if calc_object_area:
                return walkable_map_max.sum()/walkable_map.sum(), object_area_ratio
            else:
                return walkable_map_max.sum()/walkable_map.sum()
        if calc_object_area:    
            return 0.,object_area_ratio
        else:
            return 0.


if __name__ == "__main__":
    vertices = np.array([[-4.6405,  0., -3.4067],
                         [-5.8161,  0.,  0.0457],
                         [-5.8161,  0., -3.4067],
                         [-5.8161,  0.,  0.0457],
                         [-4.6405,  0., -3.4067],
                         [-5.8161,  0.,  2.149],
                         [-5.8161,  0.,  2.149],
                         [-4.6405,  0., -3.4067],
                         [-4.806,  0.,  0.9523],
                         [-4.806,  0.,  0.9523],
                         [-4.6405,  0., -3.4067],
                         [-1.6766,  0.,  0.9523],
                         [-1.6766,  0.,  0.9523],
                         [-4.6405,  0., -3.4067],
                         [-1.6766,  0., -3.4067],
                         [-4.806,  0.,  2.149],
                         [-5.8161,  0.,  2.149],
                         [-4.806,  0.,  0.9523]])
    faces = np.array([[0,  2,  1],
                      [3,  5,  4],
                      [6,  8,  7],
                      [9, 11, 10],
                      [12, 14, 13],
                      [15, 17, 16]])
    floor_plan_centroid = np.array([-3.74635,  0., -0.62885])

    bboxes = np.array([[-0.4438298,  2.48225976, -1.43605851,  0.363684,  0.11344881,
                        0.36719965, -3.12878394],
                       [1.43159928,  0.47924918,  0.20020839,  0.26718597,  0.48059947,
                        0.29016826, -1.57097435],
                       [1.81570315,  1.24139991,  0.52753237,  1.61088674,  1.23749601,
                        0.18798582, -1.58269131],
                       [-0.98245618,  1.52104479,  0.52209809,  0.09574268,  0.45265616,
                        0.09491291, -3.14084959],
                       [1.46061269,  0.47784704, -0.42971056,  0.26653308,  0.48063201,
                        0.28948942, -1.56536341],
                       [-1.28833527,  1.51192221,  0.4921519,  0.09354646,  0.45193829,
                        0.09264546,  3.13901949],
                       [0.63996411,  0.3789686, -0.15421736,  0.80494057,  0.37606669,
                        0.45039837,  1.56570542],
                       [0.0053806,  0.47375414,  0.20516208,  0.26344236,  0.47474433,
                        0.2883519,  1.57637453],
                       [-0.01188186,  0.47401467, -0.47031119,  0.26275272,  0.47505132,
                        0.28851427,  1.57725668]])

    floor_plan = [vertices, faces]
    bboxes = bboxes
    # door_position = [-4.6405,  0.    , -3.4067]
    door_position = None
    # door_position = [-3.6405,  0.    , 2.2067]
    # np.array([-3.74635,  0.     , -0.62885])
    robot_width = 0.3
    from time import time
    tt = 0
    for i in range(100):
        t1 = time()
        walkable_metric = cal_walkable_metric(
            floor_plan, floor_plan_centroid, bboxes,  door_position, robot_width)
        t2 = time()
        tt += (t1-t2)
    print(tt/100)
