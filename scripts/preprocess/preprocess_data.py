"""Script used for parsing the 3D-FRONT data scenes into numpy files in order
to be able to avoid I/O overhead when training our model.
"""
import argparse
import logging
import json
import os
import sys
sys.path.insert(0,sys.path[0]+"/../../")
import hydra

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
# from scripts.utils import get_colored_objects_in_scene
from omegaconf import DictConfig
from utils.utils_preprocess import DirLock, ensure_parent_directory_exists, \
    floor_plan_renderable, floor_plan_from_scene, \
    get_textured_objects_in_scene, scene_from_cfg, render, \
    get_colored_objects_in_scene

from datasets.base import filter_function
from datasets.threed_front import ThreedFront
from datasets.threed_front_dataset import \
    dataset_encoding_factory
import seaborn as sns
from datasets.threed_future_dataset import ThreedFutureNormPCDataset
 
@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess_data")
def main(cfg: DictConfig):
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES   
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    logging.getLogger("trimesh").setLevel(logging.ERROR)

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

    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    dataset = ThreedFront.from_dataset_directory(
        dataset_directory=cfg.dataset.path_to_3d_front_dataset_directory,
        path_to_model_info=cfg.dataset.path_to_model_info,
        path_to_models=cfg.dataset.path_to_3d_future_dataset_directory,
        path_to_room_masks_dir=cfg.task.dataset.path_to_room_masks_dir,
        filter_fn=filter_function(config, ["train", "val"], cfg.task.dataset.without_lamps)
    )
    print("Loading dataset with {} rooms".format(len(dataset)))

    # Compute the bounds for the translations, sizes and angles in the dataset.
    # This will then be used to properly align rooms.
    tr_bounds = dataset.bounds["translations"]
    si_bounds = dataset.bounds["sizes"]
    an_bounds = dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
        "class_labels": dataset.class_labels,
        "object_types": dataset.object_types,
        "class_frequencies": dataset.class_frequencies,
        "class_order": dataset.class_order,
        "count_furniture": dataset.count_furniture
    }

    if cfg.add_objfeats:

        of_bounds_32 = dataset.bounds["objfeats_32"]
        print([of_bounds_32[0], of_bounds_32[1], of_bounds_32[2]], type(of_bounds_32[0]), of_bounds_32[0].shape)
        dataset_stats["bounds_objfeats_32"] = of_bounds_32[0].tolist() + of_bounds_32[1].tolist() + of_bounds_32[2].tolist()
        print(of_bounds_32[0].tolist() + of_bounds_32[1].tolist() + of_bounds_32[2].tolist())
        print("add objfeats_32 statistics: std {}, min {}, max {}".format(of_bounds_32[0], of_bounds_32[1], of_bounds_32[2]))

    path_to_json = os.path.join(cfg.output_directory, "dataset_stats.txt")
    with open(path_to_json, "w") as f:
        json.dump(dataset_stats, f)
    print(
        "Saving training statistics for dataset with bounds: {} to {}".format(
            dataset.bounds, path_to_json
        )
    )

    dataset = ThreedFront.from_dataset_directory(
        dataset_directory=cfg.dataset.path_to_3d_front_dataset_directory,
        path_to_model_info=cfg.dataset.path_to_model_info,
        path_to_models=cfg.dataset.path_to_3d_future_dataset_directory,
        filter_fn=filter_function(
            config, ["train", "val", "test"], cfg.task.dataset.without_lamps
        )
    )
    print(dataset.bounds)
    print("Loading dataset with {} rooms".format(len(dataset)))

    encoded_dataset = dataset_encoding_factory(
        "basic", dataset, augmentations=None, box_ordering=None
    )

    # Create the scene and the behaviour list for simple-3dviz
    scene_black = scene_from_cfg(cfg,background=[0,0,0,1])
    scene_gray = scene_from_cfg(cfg)

    cnt = len(encoded_dataset)
    for i in range(cnt):
        es = encoded_dataset[i]
        ss = dataset[i]
        print(i)

        # Create a separate folder for each room
        room_directory = os.path.join(cfg.output_directory, ss.uid)
        print(room_directory)
        # Check if room_directory exists and if it doesn't create it
        
        if os.path.exists(room_directory):
            continue


        # Make sure we are the only ones creating this file
        with DirLock(room_directory + ".lock") as lock:
            if not lock.is_acquired:
                continue
            if os.path.exists(room_directory):
                print("Folder already exists")
                continue
                
            ensure_parent_directory_exists(room_directory)

            uids = [bi.model_uid for bi in ss.bboxes]
            jids = [bi.model_jid for bi in ss.bboxes]

            floor_plan_vertices, floor_plan_faces = ss.floor_plan

            # Render and save the room mask as an image
            room_mask = render(
                scene_black,
                [floor_plan_renderable(ss)],
                (1.0, 1.0, 1.0),
                "flat",
                os.path.join(room_directory, "room_mask.png")
            )[:, :, 0:1]

            if cfg.add_objfeats:
                np.savez_compressed(
                    os.path.join(room_directory, "boxes"),
                    uids=uids,
                    jids=jids,
                    scene_id=ss.scene_id,
                    scene_uid=ss.uid,
                    scene_type=ss.scene_type,
                    json_path=ss.json_path,
                    room_layout=room_mask,
                    floor_plan_vertices=floor_plan_vertices,
                    floor_plan_faces=floor_plan_faces,
                    floor_plan_centroid=ss.floor_plan_centroid,
                    class_labels=es["class_labels"],
                    translations=es["translations"],
                    sizes=es["sizes"],
                    angles=es["angles"],
                    objfeats_32=es["objfeats_32"],
                    objfeats_pc_ulips=es["objfeats_pc_ulip"],
                )
                    # objfeats=es["objfeats"],
            else:
                np.savez_compressed(
                    os.path.join(room_directory, "boxes"),
                    uids=uids,
                    jids=jids,
                    scene_id=ss.scene_id,
                    scene_uid=ss.uid,
                    scene_type=ss.scene_type,
                    json_path=ss.json_path,
                    room_layout=room_mask,
                    floor_plan_vertices=floor_plan_vertices,
                    floor_plan_faces=floor_plan_faces,
                    floor_plan_centroid=ss.floor_plan_centroid,
                    class_labels=es["class_labels"],
                    translations=es["translations"],
                    sizes=es["sizes"],
                    angles=es["angles"]
                )


            if cfg.no_texture:
                # Render a top-down orthographic projection of the room at a
                # specific pixel resolutin
                path_to_image = "{}/rendered_scene_notexture_{}.png".format(
                    room_directory, cfg.visualizer.window_size[0]
                )
                if os.path.exists(path_to_image):
                    continue
                
                floor_plan, _, _ = floor_plan_from_scene(
                    ss, cfg.path_to_floor_plan_textures, without_room_mask=True, no_texture=True,
                )
                # read class labels and get the color map of each class
                class_labels = es["class_labels"]
                color_palette = np.array(sns.color_palette('hls', class_labels.shape[1]-1))
                class_index = class_labels.argmax(axis=1)
                cc = color_palette[class_index, :]
                print('class_labels :', class_labels.shape)
                renderables = get_colored_objects_in_scene(
                    ss, cc, ignore_lamps=cfg.task.dataset.without_lamps
                )
            else:
                # Render a top-down orthographic projection of the room at a
                # specific pixel resolutin
                path_to_image = "{}/rendered_scene_{}.png".format(
                    room_directory, cfg.visualizer.window_size[0]
                )
                if os.path.exists(path_to_image):
                    continue

                # Get a simple_3dviz Mesh of the floor plan to be rendered
                floor_plan, _, _ = floor_plan_from_scene(
                    ss, cfg.path_to_floor_plan_textures, without_room_mask=True, no_texture=False,
                )
                renderables = get_textured_objects_in_scene(
                    ss, ignore_lamps=cfg.task.dataset.without_lamps
                )

            if cfg.without_floor:
                render(
                    scene_gray,
                    renderables,
                    color=None,
                    mode="shading",
                    frame_path=path_to_image
                )
            else:
                render(
                    scene_gray,
                    renderables + floor_plan,
                    color=None,
                    mode="shading",
                    frame_path=path_to_image
                )



if __name__ == "__main__":
    main()