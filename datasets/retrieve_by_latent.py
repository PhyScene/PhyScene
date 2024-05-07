"""Script used to train a ATISS."""
import argparse
import os
import sys
from numpy.linalg import norm

import numpy as np

import pickle
from scripts.train.training_utils import load_config

from datasets.base import filter_function

from datasets.threed_front import ThreedFront
from datasets.gapartnet_dataset import GAPartNetDataset
import hydra
from omegaconf import DictConfig, OmegaConf

def load_objects(config):
    config_task = config.task
    split = ["train", "val", "test"]

     ## add bed rooms
    config1 = {
        "filter_fn":                 "threed_front_bedroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config_task["dataset"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config_task["dataset"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "configs/data/bedroom_threed_front_splits.csv"
    }

    scenes_train_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["dataset"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["dataset"]["path_to_model_info"],
        path_to_models=config["dataset"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config1, split, config_task["dataset"]["without_lamps"])
    )
    print("Loading train dataset with {} rooms".format(len(scenes_train_dataset)))

    # add dining rooms
    config2 = {
        "filter_fn":                 "threed_front_diningroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config_task["dataset"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config_task["dataset"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "configs/data/diningroom_threed_front_splits.csv"
    }
    scenes_train_dataset2 = ThreedFront.from_dataset_directory(
        dataset_directory=config["dataset"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["dataset"]["path_to_model_info"],
        path_to_models=config["dataset"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config2, split, config_task["dataset"]["without_lamps"])
    )
    print("Loading train dataset 2 with {} rooms".format(len(scenes_train_dataset2)))

    ## add living rooms
    config3 = {
        "filter_fn":                 "threed_front_livingroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config_task["dataset"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config_task["dataset"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "configs/data/livingroom_threed_front_splits.csv"
    }
    scenes_train_dataset3 = ThreedFront.from_dataset_directory(
        dataset_directory=config["dataset"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["dataset"]["path_to_model_info"],
        path_to_models=config["dataset"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config3, split, config_task["dataset"]["without_lamps"])
    )
    print("Loading train dataset 3 with {} rooms".format(len(scenes_train_dataset3)))

    # Build the dataset of Garpartnet
    pickled_GPN_dir = config.GAPartNet.pickled_GPN_dir
    pickled_GPN_path = "{}/gapartnet_model.pkl".format(pickled_GPN_dir)
    if os.path.exists(pickled_GPN_path):
        gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
    else:
        gapartnet_dataset = GAPartNetDataset()
        with open(pickled_GPN_path, "wb") as f:
            pickle.dump(gapartnet_dataset, f)

    # Collect the set of objects in the scenes
    retrevie_objects_3dfuture = {}
    retrevie_objects_GPN = {}
    # Collect the set of objects in the scenes
    for scene in scenes_train_dataset:
        for obj in scene.bboxes:
            retrevie_objects_3dfuture[obj.model_jid] = obj
    # diningroom
    for scene in scenes_train_dataset2:
        for obj in scene.bboxes:
            retrevie_objects_3dfuture[obj.model_jid] = obj
    # livingroom
    for scene in scenes_train_dataset3:
        for obj in scene.bboxes:
            retrevie_objects_3dfuture[obj.model_jid] = obj
    #GPN
    for obj in gapartnet_dataset.objects:
        # if obj.label == "StorageFurniture":
        if True:
            retrevie_objects_GPN[obj.model_jid] = obj
    
    retrevie_objects_3dfuture = [vi for vi in retrevie_objects_3dfuture.values()]
    retrevie_objects_GPN = [vi for vi in retrevie_objects_GPN.values()]
    return retrevie_objects_3dfuture, retrevie_objects_GPN
    
    

def load_all_latent(retrevie_objects):
    latent_dict = dict()
    # name = "0aab9dbc-ada7-439f-8ebe-742d076ce617"
    for obj in retrevie_objects:
        #lat32
        # filename = obj.raw_model_norm_pc_lat32_path
        #ulip
        try:
            filename = obj.raw_model_path[:-4]+"_norm_pc_lat_ulip.npz"
        except:
            filename = "/".join(obj.raw_model_path[0].split("/")[:-1]) + "/raw_model_norm_pc_lat_ulip.npz"
        if os.path.exists(filename):
            data = np.load(filename)['latent']
            latent_dict[filename] = data
    return latent_dict

def retrieve_by_name(latent_dict_3dfuture,latent_dict_GPN, name):
    if name in latent_dict_3dfuture:
        latent = latent_dict_3dfuture[name]
    else:
        latent = latent_dict_GPN[name]
    #retrieve 3d future
    scores = {}
    for filename in latent_dict_3dfuture:
        feature = latent_dict_3dfuture[filename]
        # compute cosine similarity
        cosine = np.dot(latent,feature)/(norm(latent)*norm(feature))
        scores[filename] = cosine
    sorted_scores = [[k,v] for k, v in sorted(scores.items(), key=lambda x:x[1],reverse=True)]
    sorted_ids_3dfuture = [k for k, v in sorted(scores.items(), key=lambda x:x[1],reverse=True)]

    #retrieve gpn
    scores = {}
    for filename in latent_dict_GPN:
        feature = latent_dict_GPN[filename]
        # compute cosine similarity
        cosine = np.dot(latent,feature)/(norm(latent)*norm(feature))
        scores[filename] = cosine
    sorted_scores = [[k,v] for k, v in sorted(scores.items(), key=lambda x:x[1],reverse=True)]
    sorted_ids_GPN = [k for k, v in sorted(scores.items(), key=lambda x:x[1],reverse=True)]
    return sorted_ids_3dfuture, sorted_ids_GPN

@hydra.main(version_base=None, config_path="./configs", config_name="pickle_data")
def main(cfg: DictConfig):
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    #load object list
    retrevie_objects_3dfuture, retrevie_objects_GPN = load_objects(cfg)
    #build dict={filename:latent feature}
    latent_dict_3dfuture = load_all_latent(retrevie_objects_3dfuture)
    latent_dict_GPN = load_all_latent(retrevie_objects_GPN)
    #sort by name and retrieve top 10 
    # a sample of table
    name = os.path.join(cfg.dataset.path_to_3d_future_dataset_directory,"2b9e8a81-a96c-4dd5-8fe5-9d9949b09362/raw_model_norm_pc_lat_ulip.npz")
    sorted_ids_3dfuture, sorted_ids_GPN = retrieve_by_name(latent_dict_3dfuture,latent_dict_GPN, name)
    print(sorted_ids_3dfuture[:10])
    print(sorted_ids_GPN[:10])



if __name__ == "__main__":
    main()
    