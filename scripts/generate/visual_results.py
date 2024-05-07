# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for retrieve object scenes with previously saved bbox scenes."""
import logging
import numpy as np
import torch
import json
import os
import hydra
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
import pickle
import sys
sys.path.insert(0,sys.path[0]+"/../../")
from simple_3dviz.renderables.mesh import Mesh
from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh

from datasets.gapartnet_dataset import GAPartNetDataset
from datasets.base import filter_function, get_dataset_raw_and_encoded
from datasets.threed_future_dataset import ThreedFutureDataset
from models.networks import build_network
from utils.utils_preprocess import render as render_top2down
from utils.utils_preprocess import room_outer_box_from_obj
from utils.utils import get_textured_objects, get_bbox_objects, get_textured_objects_from_bbox
from scripts.eval.calc_ckl import show_renderables
from utils.overlap import calc_wall_overlap,calc_overlap_rotate_bbox,calc_overlap_rotate_bbox_doors
from scripts.eval.calc_ckl import init_render_scene, calc_overlap
import json



@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    
    global config
    config = cfg
    cfg.task.dataset = {**cfg.task.dataset,**cfg.dataset}
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config.task["dataset"],
        filter_fn=filter_function(
            config.task["dataset"],
            split=["train","val","test"]
        ),
        split=["train","val","test"]
    )

    classes = np.array(dataset.class_labels)
    vis = True
    evaluate(None,dataset,config,raw_dataset,dataset,device,vis=vis)
    return 

def evaluate(network,dataset,cfg,raw_dataset,ground_truth_scenes,device,vis:True):
    classes = np.array(dataset.class_labels)
    print('class labels:', classes, len(classes))

    if vis:
        # Build the dataset of 3D models
        objects_dataset = ThreedFutureDataset.from_pickled_dataset(
            cfg.task.path_to_pickled_3d_futute_models
        )
        print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    path_to_center_info = cfg.ai2thor.path_to_center_info
    with open(path_to_center_info,"r") as f:
        center_info = json.load(f)

    # Build the dataset of Garpartnet
    # pickled_GPN_dir = cfg.GAPartNet.pickled_GPN_dir
    # pickled_GPN_path = "{}/gapartnet_model.pkl".format(pickled_GPN_dir)
    # gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
    gapartnet_dataset = None
    # pickled_GPN_dir = cfg.GAPartNet.pickled_GPN_dir
    # pickled_GPN_path = "{}/gapartnet_model_good_idx.pkl".format(pickled_GPN_dir)
    # if os.path.exists(pickled_GPN_path):
    #     gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
    # else:
    #     gapartnet_dataset = GAPartNetDataset(cfg, remove_largeopen=True)
    #     with open(pickled_GPN_path, "wb") as f:
    #         pickle.dump(gapartnet_dataset, f)
    # print("Loaded {} Gapartnet models".format(len(gapartnet_dataset.objects)))

    #### limit room lst
    boxdir = "/home/yandan/workspace/PhyScene/ai2thor/generate_bbox"
    room_dict = dict()
    for d in os.listdir(boxdir):
        housedir = os.path.join(boxdir,d)
        for file in os.listdir(housedir):
            if file.endswith("_bbox.json"):
                roomdir = os.path.join(housedir,file)
                obj_name = "House_"+file.split("_")[1]+"/room_"+file.split("_")[3]+".obj"
                room_dict[obj_name] = roomdir

    # for i in range(268,len(center_info)//batch_size+1):
    for i in range(len(room_dict.keys())):
        
        floor_plan_mask_list = [] 
        floor_plan_centroid_list = []        
        # Get a floor plan
        room_lst = []
        floor_plan_lst = []
        room_outer_box_lst = []
        mask_name_lst = []
        for  j in range(1): 
            obj_name = list(room_dict.keys())[i]
            obj_name = "House_51/room_3.obj"
            roomdir = room_dict[obj_name]

            roomType = center_info[obj_name]["roomType"]
            filter_fn = cfg.task.dataset.filter_fn
            if filter_fn =="threed_front_diningroom":
                filter_fn = "threed_front_livingroom"
            if roomType.lower() not in filter_fn:
            # if roomType.lower() != "livingroom": 
                continue
            
            obj_path = os.path.join(cfg.ai2thor.path_to_ai2thor,obj_name)
            if not os.path.exists(obj_path):
                continue

            center = center_info[obj_name]["center"]

            mask_name = "_".join(obj_name[:-4].split("/"))+".png"
            mask_name_lst.append(mask_name)
            mask_name = os.path.join(cfg.ai2thor.path_to_mask,mask_name)

            room_mask = Image.open(mask_name).convert("RGB")
            room_mask = np.asarray(room_mask).astype(np.float32) / np.float32(255)
            room_mask = torch.from_numpy(
                np.transpose(room_mask[None, :, :, 0:1], (0, 3, 1, 2))
            )
            
            floor_plan =  Mesh.from_file(obj_path)
            center = (floor_plan.bbox[0]+floor_plan.bbox[1])/2
            vertices, faces = floor_plan.to_points_and_faces()
            
            floor_plan_mask_list.append([vertices.copy(), faces.copy()])
            floor_plan_centroid_list.append(center)
            # Center the floor
            vertices -= center
            
        
            #texture floor
            uv = np.copy(vertices[:, [0, 2]])
            uv -= uv.min(axis=0)
            uv /= 0.3  # repeat every 30cm
            floor_plan = TexturedMesh.from_faces(
                vertices=vertices,
                uv=uv,
                faces=faces,
                material=Material.with_texture_image("demo/floor_plan_texture_images/floor_00003.jpg")
            )

            room_outer_box = room_outer_box_from_obj(vertices, faces)
            room_outer_box = torch.Tensor(room_outer_box[None,:,:]).to(device)           

            room_outer_box_lst.append(room_outer_box)
            room_lst.append(room_mask)
            floor_plan_lst.append(floor_plan)
        if room_lst == []:
            continue
        room_mask = torch.concat(room_lst)
        room_outer_box = torch.concat(room_outer_box_lst)

        with open(roomdir,"r") as f:
            bboxes = json.load(f)["objects"]

        # show_scene(box_scene,objects_dataset,dataset,cfg,floor_plan_lst[s:s+1],None,room_outer_box[s:s+1],gapartnet_dataset,mask_name_lst[s:s+1])
        show_scene(bboxes,objects_dataset,dataset,cfg,floor_plan_lst[:],None,None,gapartnet_dataset,mask_name_lst[:])
    return 


def show_scene(bboxes,objects_dataset,dataset,cfg,floor_plan_lst,tr_floor,room_outer_box=None,gapartnet_dataset=None,mask_name_lst=[]):
    
    
    scene_top2down = init_render_scene(cfg,gray=True,room_side=6.2,size=(512,512))

    classes = np.array(dataset.class_labels)
    for i in range(1):

        houseinfo = mask_name_lst[i][:-4].split("_")
        House_number = "_".join(houseinfo[:2])

        # [1] retrieve objects
        renderables, trimesh_meshes, _,_ , scene_info = get_textured_objects_from_bbox(
            bboxes, objects_dataset, gapartnet_dataset, classes, cfg
        )
        roomtype = cfg.task.dataset.filter_fn.split("_")[-1]
        path_to_json = "{}/{}/{}".format(
            cfg.ai2thor.path_to_result,
            House_number,
            mask_name_lst[i][:-4]+"_"+roomtype+".json"
        )
        


        # if not without_floor:
        floor_plan = floor_plan_lst[i]
        import math
        theta = math.pi*3/2
        R = np.zeros((3, 3))
        R[2, 2] = np.cos(theta)
        R[2, 1] = -np.sin(theta)
        R[1, 2] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        R[0, 0] = 1.
        # Apply the transformations in order to correctly position the mesh
        floor_plan.affine_transform(R=R)

        if True:
            renderables += [floor_plan]
            # trimesh_meshes += tr_floor

        # if render2img:
        if True:
            path_to_image = "{}/{}/{}".format(
                cfg.ai2thor.path_to_result,
                House_number,
                mask_name_lst[i][:-4]+"_"+roomtype+".png"
            )
            render_top2down(
                scene_top2down,
                renderables,
                color=None,
                mode="shading",
                frame_path=path_to_image,
            )
            print("saving image into  ",path_to_image)
        else:
            show_renderables(renderables)
        show_renderables(renderables)

        
        save_mesh = False
        if save_mesh:
            if trimesh_meshes is not None:
                # Create a trimesh scene and export it
               
                path_to_scene = "debug.obj"
                # whole_scene_mesh = merge_meshes( trimesh_meshes )
                whole_scene_mesh = trimesh.util.concatenate(trimesh_meshes)
                whole_scene_mesh.export(path_to_scene)
    return



if __name__ == "__main__":
    import random
    import numpy as np
    seed = 2022
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()