"""Script used for pickling the 3D Future dataset in order to be subsequently
used by our scripts.
"""
import os
import sys
sys.path.insert(0,sys.path[0]+"/../../")

from datasets.base import filter_function
from datasets.threed_front import ThreedFront
from datasets.threed_future_dataset import ThreedFutureDataset
import trimesh
import numpy as np
import open3d as o3d
from datasets.utils_io import export_pointcloud, export_pointcloud_with_rgb
import open3d.visualization.gui as gui
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../../configs", config_name="pickle_data")
def main(cfg: DictConfig):
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(cfg.output_directory):
        os.makedirs(cfg.output_directory)

    with open(cfg.task.dataset.path_to_invalid_scene_ids, "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    with open(cfg.task.dataset.path_to_invalid_bbox_jids, "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    config = {
        "filter_fn":                 cfg.task.dataset.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": cfg.task.dataset.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": cfg.task.dataset.path_to_invalid_bbox_jids,
        "annotation_file":           cfg.task.dataset.annotation_file
    }

    for split in ["train", "val", "test"]:

        # Initially, we only consider the train split to compute the dataset
        # statistics, e.g the translations, sizes and angles bounds
        scenes_dataset = ThreedFront.from_dataset_directory(
            dataset_directory=cfg.dataset.path_to_3d_front_dataset_directory,
            path_to_model_info=cfg.dataset.path_to_model_info,
            path_to_models=cfg.dataset.path_to_3d_future_dataset_directory,
            path_to_room_masks_dir=cfg.task.dataset.path_to_room_masks_dir,
            filter_fn=filter_function(config, [split], cfg.task.dataset.without_lamps)
        )
        print("Loading dataset with {} rooms".format(len(scenes_dataset)))

        # Collect the set of objects in the scenes
        objects = {}
        for scene in scenes_dataset:
            for obj in scene.bboxes:
                objects[obj.model_jid] = obj
        objects = [vi for vi in objects.values()]

        objects_dataset = ThreedFutureDataset(objects)
        room_type = cfg.task.dataset.dataset_filtering.split("_")[-1]
        output_directory = "{}/threed_future_pointcloud_{}".format(
            cfg.output_directory,
            room_type
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        model_names = []
        #read the ThreedFutureDataset
        for idx in range(len(objects_dataset)):
            obj = objects_dataset[idx]
            model_uid = obj.model_uid
            model_jid = obj.model_jid
            raw_model_path = obj.raw_model_path
            model_names.append(model_jid)

            filename = "/".join(raw_model_path.split("/")[:-1]) + "/raw_model_norm_pc.npz"
            if os.path.exists(filename):
                continue

            # read mesh
            mesh = trimesh.load(
                raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,   #False for color
                skip_texture=True
            )
           
            
            bbox = mesh.bounding_box.bounds

            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max() / (1 - cfg.bbox_padding)

            # Transform input mesh
            mesh.apply_translation(-loc)
            mesh.apply_scale(1 / scale)


            # sample point clouds with normals
            points, face_idx = mesh.sample(cfg.pointcloud_size, return_index=True)
            normals = mesh.face_normals[face_idx]
            #add colors
            vertex_colors = mesh.visual.to_color().vertex_colors
            face_colors = trimesh.visual.color.vertex_to_face_color(vertex_colors, mesh.faces) 
            # face_colors = mesh.visual.to_color()
            # face_colors = face_colors._get_colors('face')
            # vertex_colors = mesh.visual.to_color().vertex_colors
            face_colors = face_colors[face_idx]

            # #visualize PCD
            # o3d_mesh = o3d.geometry.TriangleMesh()
            # o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            # o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            # o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
            # pcd = o3d.t.geometry.PointCloud(
            #             np.array(points, dtype=np.float32))
            # draw_box_label(o3d_mesh,pcd)

            # Compress
            dtype = np.float16
            #dtype = np.float32
            points = points.astype(dtype)
            normals = normals.astype(dtype)
            # face_colors = face_colors.astype(dtype)

            # save_npz or ply
            #filename = "{}/{}.npz".format(output_directory, model_jid)
            filename = raw_model_path[:-4] + "_norm_pc.npz"
            print('Writing pointcloud: %s' % filename)
            np.savez(filename, points=points, face_colors = face_colors, normals=normals, loc=loc, scale=scale)

            # filename = "{}/{}_color.ply".format(output_directory, model_jid)
            filename = "{}/{}.ply".format(output_directory, model_jid)
            export_pointcloud(points, filename)
            # export_pointcloud_with_rgb(points, filename, face_colors)
            print('Writing pointcloud: %s' % filename)

        # write train/val/test split
        split_lst = "{}/threed_future_pointcloud_{}/{}.lst".format(
            cfg.output_directory,
            room_type,
            split
        )
        open(split_lst, 'w').writelines([ name +'\n' for name in model_names])
            
def draw_box_label(mesh,pointcloud):
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.show_settings = True
        # vis.add_geometry("mesh",mesh)
        vis.add_geometry("pointcloud",pointcloud)

        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

        return vis
    

if __name__ == "__main__":
    main()