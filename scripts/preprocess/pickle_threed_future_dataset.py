# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for pickling the 3D Future dataset in order to be subsequently
used by our scripts.
"""
import argparse
import os
import sys
sys.path.insert(0,sys.path[0]+"/../../")

import pickle
import hydra
from omegaconf import DictConfig
from datasets.base import filter_function
from datasets.threed_front import ThreedFront
from datasets.threed_front_dataset import \
    dataset_encoding_factory
from datasets.threed_future_dataset import ThreedFutureDataset

@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess_data")
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

    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    scenes_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=cfg.dataset.path_to_3d_front_dataset_directory,
        path_to_model_info=cfg.dataset.path_to_model_info,
        path_to_models=cfg.dataset.path_to_3d_future_dataset_directory,
        path_to_room_masks_dir=cfg.task.dataset.path_to_room_masks_dir,
        filter_fn=filter_function(config, ["train", "val"], cfg.task.dataset.without_lamps)
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
    output_path = "{}/threed_future_model_{}.pkl".format(
        cfg.output_directory,
        room_type
    )
    with open(output_path, "wb") as f:
        pickle.dump(objects_dataset, f)
    print("finish saving ",output_path )


if __name__ == "__main__":
    main()
