"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]  #x,z,y
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, np.radians(gt_boxes[6]) + 1e-10, 0])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, names = None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        if names is not None:
            corners = box3d.get_box_points()
            vis.add_3d_label(corners[5], '%s' % names[i])


        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud

def high_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", points)
    for idx in range(0, len(points.points)):
        vis.add_3d_label(points.points[idx], "{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()

def draw_box_label(gt_boxes, names = None, mesh_floor_plan=None):
    #gt_boxes: xcenter,ycenter,zcenter,xl,yl,zl,degree
    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True

    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        vis.add_geometry(str(i),line_set)
        # vis.add_geometry(names[i]+str(i),line_set)

        if names is not None:
            corners = box3d.get_box_points()
            vis.add_3d_label(corners[5], '%s' % names[i]+str(i))

    if mesh_floor_plan is not None:
            vis.add_geometry("floorplan",mesh_floor_plan)
    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()
    app.quit()

    return vis




if __name__ == "__main__":

   
    names = ["nightstand","double_bed","wardrobe","nightstand2","ceiling_lamp"]
    gt_boxes = [[14,20,14,17,20,37,90],
                [67,33,61,146,33,176,-90],
                [90,99,25,170,99,27,0],
                [14,20,14,238,20,38,-90],
                [23,5,23,120,204,138,0]]


    gt_boxes = [[i[3]+i[0]/2,i[5]+i[2]/2,i[4]-i[1]/2,
                 i[0],i[1],i[2],i[6]] for i in gt_boxes]
    gt_boxes = np.array(gt_boxes)

    # vis = open3d.visualization.Visualizer()
    # vis.create_window()

    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.zeros(3)
    # gt_boxes = np.array([[0,0,0,3,4,5,0],[10,10,10,3,4,5,0]])
    # names = ["debug","aaa"]
    vis = draw_box_label(gt_boxes, (0, 0, 1), names = names)
    # vis.run()
    # vis.destroy_window()
    # high_level()
