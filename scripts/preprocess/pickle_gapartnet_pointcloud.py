"""Script used for pickling the GAPartNet dataset in order to be subsequently
used by our scripts.
"""
import os
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.insert(0,sys.path[0]+"/../../")
from datasets.threed_future_dataset import ThreedFutureDataset

import trimesh
import numpy as np
import open3d as o3d
from datasets.utils_io import export_pointcloud, export_pointcloud_with_rgb
from datasets.gapartnet_dataset import GAPartNetDataset
import open3d.visualization.gui as gui

def update_value(basedata,new_data):
    if basedata is None:
        return new_data
    else:
        data = np.concatenate((basedata,new_data),axis=0)
        return data
        
@hydra.main(version_base=None, config_path="../../configs", config_name="pickle_data")
def main(cfg: DictConfig):
    
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR    

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(cfg.output_directory):
        os.makedirs(cfg.output_directory)
    
    # Build the dataset of Garpartnet
    pickled_GPN_dir = cfg.GAPartNet.pickled_GPN_dir
    pickled_GPN_path = "{}/gapartnet_model.pkl".format(pickled_GPN_dir)
    if os.path.exists(pickled_GPN_path):
        gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
    else:
        gapartnet_dataset = GAPartNetDataset(cfg)
        with open(pickled_GPN_path, "wb") as f:
            pickle.dump(gapartnet_dataset, f)
    
    print("Loaded {} Gapartnet models".format(len(gapartnet_dataset.objects)))

    objects_dataset = ThreedFutureDataset(gapartnet_dataset.objects)
    # room_type = args.dataset_filtering.split("_")[-1]
    output_directory = "{}/gapartnet_pointcloud".format(
        cfg.output_directory 
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    model_names = []
    #read the ThreedFutureDataset
    for idx in range(len(objects_dataset)):
        
        # idx = 4
        obj = objects_dataset[idx]
        model_uid = obj.model_uid
        model_jid = obj.model_jid
        model_names.append(model_jid)
        
        raw_model_path = obj.raw_model_path
        try:
            filename = "/".join(raw_model_path.split("/")[:-1]) + "/raw_model_norm_pc.npz"
        except:
            filename = "/".join(raw_model_path[0].split("/")[:-1]) + "/raw_model_norm_pc.npz"

        # try:
        #     filename = "/".join(raw_model_path.split("/")[:-1]) + "/raw_model_norm_pc_open.npz"
        # except:
        #     filename = "/".join(raw_model_path[0].split("/")[:-1]) + "/raw_model_norm_pc_open.npz"

        if os.path.exists(filename):
            continue
        raw_model_path = obj.raw_model_path_close
        # raw_model_path = obj.raw_model_path_open
        mesh = trimesh.load(
            raw_model_path,
            process=False,
            force="mesh",
            skip_materials=False,
            skip_texture=False
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
        # vertex_colors = mesh.visual.to_color().vertex_colors
        # face_colors = trimesh.visual.color.vertex_to_face_color(vertex_colors, mesh.faces) 
        # face_colors = mesh.visual.to_color()
        # face_colors = face_colors._get_colors('face')
        # vertex_colors = mesh.visual.to_color().vertex_colors
        # face_colors = face_colors[face_idx]

        # pcd = o3d.t.geometry.PointCloud(
        #             np.array(points, dtype=np.float32))
        # draw_box_label(None,pcd)

        # Compress
        dtype = np.float16
        #dtype = np.float32
        points = points.astype(dtype)
        normals = normals.astype(dtype)
        # face_colors = face_colors.astype(dtype)

        # save_npz or ply
        print('Writing pointcloud: %s' % filename)
        np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)

        filename = "{}/{}.ply".format(output_directory, model_jid)
        # filename = "{}/{}_open.ply".format(output_directory, model_jid)
        export_pointcloud(points, filename)
        # export_pointcloud_with_rgb(points, filename, face_colors)
        print('Writing pointcloud: %s' % filename)

    # write train/val/test split
    split = "train"
    split_lst = "{}/gapartnet_pointcloud/{}.lst".format(
        cfg.output_directory,
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