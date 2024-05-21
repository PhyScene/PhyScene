# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating bboxes scenes (do not rely on object retrieval or other datasets)
     with procthor floor plan."""
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
from utils.utils import get_textured_objects, get_bbox_objects
from scripts.eval.calc_ckl import show_renderables
from utils.overlap import calc_wall_overlap,calc_overlap_rotate_bbox,calc_overlap_rotate_bbox_doors
from scripts.eval.calc_ckl import init_render_scene, calc_overlap
import json

def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()

def calc_door_box(doors,center):
    door_boxes = []
    for door in doors:
        doorType = door["doorType"]
        holePolygon = door["holePolygon"]
        x0,y0,z0 = holePolygon[0]
        x1,y1,z1 = holePolygon[1]
        y_mean = (y0+y1)/2-center[1]
        y_size = max(y1-y0,y0-y1)
        x_mean = (x0+x1)/2-center[0]
        x_size = max(x1-x0,x0-x1)
        z_mean = (z0+z1)/2-center[2]
        z_size = max(z1-z0,z0-z1)
        if x_size<z_size:
            x_size = z_size if doorType=="single" else z_size/2
        else:
            z_size = x_size if doorType=="single" else x_size/2
        #(x,z,y,w,l,h,alpha)->(x,y,z,w,h,l,alpha)
        
        door_boxes.append([x_mean,z_mean,y_mean,x_size,z_size,y_size,0])

    return torch.tensor(door_boxes)

@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    
    global config
    config = cfg
    cfg.task.dataset = {**cfg.task.dataset,**cfg.dataset}
    weight_file = cfg.task.evaluator.weight_file
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
        path_to_bounds=config.task.dataset.path_to_bounds,
        split=["train","val","test"]
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, weight_file, dataset, device=device
    )
    network.eval()

    vis = False
    evaluate(network,dataset,config,device,vis=vis)
    return 

def evaluate(network,dataset,cfg,device,vis:True):
    classes = np.array(dataset.class_labels)
    print('class labels:', classes, len(classes))
    synthesized_scenes = []
    batch_size = 128
    if vis:
        # Build the dataset of 3D models
        objects_dataset = ThreedFutureDataset.from_pickled_dataset(
            cfg.task.path_to_pickled_3d_futute_models
        )
        print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    path_to_center_info = cfg.ai2thor.path_to_center_info
    with open(path_to_center_info,"r") as f:
        center_info = json.load(f)

    pickled_GPN_dir = cfg.GAPartNet.pickled_GPN_dir
    pickled_GPN_path = "{}/gapartnet_model_good_idx.pkl".format(pickled_GPN_dir)
    if os.path.exists(pickled_GPN_path):
        gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
    else:
        gapartnet_dataset = GAPartNetDataset(cfg, remove_largeopen=True)
        with open(pickled_GPN_path, "wb") as f:
            pickle.dump(gapartnet_dataset, f)
    print("Loaded {} Gapartnet models".format(len(gapartnet_dataset.objects)))

   
    for i in range(len(center_info)//batch_size+1):
        print("{} / {}:".format(
            i, len(center_info)//batch_size+1)
        )
        floor_plan_mask_list = [] 
        floor_plan_centroid_list = []
        synthesized_scenes = []
        
        # Get a floor plan
        room_lst = []
        floor_plan_lst = []
        room_outer_box_lst = []
        mask_name_lst = []
        door_boxes_lst = []
        if i<=194:
            continue

        for  j in range(batch_size): 
            idx = i*batch_size + j
            # if idx<=19783:
            #     continue

            if idx<len(center_info.keys()):
                obj_name = list(center_info.keys())[idx]
                print(obj_name)
            else:
                break
            # obj_name = "House_8236/room_8.obj"
            roomType = center_info[obj_name]["roomType"]
            filter_fn = cfg.task.dataset.filter_fn
            if filter_fn =="threed_front_diningroom":
                filter_fn = "threed_front_livingroom"
            if roomType.lower() not in filter_fn:
                continue
            
            obj_path = os.path.join(cfg.ai2thor.path_to_ai2thor,obj_name)
            if not os.path.exists(obj_path):
                continue

            center = center_info[obj_name]["center"]
            doors = center_info[obj_name]["door"]
            door_boxes = calc_door_box(doors,center)
            door_boxes_lst.append(door_boxes)

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
            
            #gray floor
            # colors = np.ones((len(vertices), 3))*[0.9, 0.9, 0.9]
            # floor_plan = Mesh.from_faces(vertices, faces, colors=colors)

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

            # import trimesh
            # tr_floor = trimesh.Trimesh(
            #     np.copy(vertices), np.copy(faces), process=False
            # )
            # uv = np.copy(vertices[:, [0, 2]])
            # uv -= uv.min(axis=0)
            # uv /= 0.3  # repeat every 30cm
            # tr_floor.visual = trimesh.visual.TextureVisuals(
            #     uv=np.copy(uv),
            #     material=trimesh.visual.material.SimpleMaterial(
            #         image=Image.open("demo/floor_plan_texture_images/floor_00003.jpg")
            #     )
            # )

            room_outer_box = room_outer_box_from_obj(vertices, faces)
            room_outer_box = torch.Tensor(room_outer_box[None,:,:]).to(device)           

            room_outer_box_lst.append(room_outer_box)
            room_lst.append(room_mask)
            floor_plan_lst.append(floor_plan)
        if room_lst == []:
            continue
        room_mask = torch.concat(room_lst)
        room_outer_box = torch.concat(room_outer_box_lst)

        bbox_params = network.generate_layout(
            room_mask=room_mask.to(device),
            room_outer_box=room_outer_box,
            batch_size = len(room_lst),
            num_points=cfg.task["network"]["sample_num_points"],
            point_dim=cfg.task["network"]["point_dim"],
            # text=torch.from_numpy(samples['desc_emb'])[None, :].to(device) if 'desc_emb' in samples.keys() else None,
            #text=samples['description'] if 'description' in samples.keys() else None,
            device=device,
            clip_denoised=cfg.clip_denoised,
            batch_seeds=torch.arange(i, i+1),
            keep_empty = True
        )

        boxes = dataset.post_process(bbox_params)
        # gapartnet_dataset = None

        for j in range(len(boxes["class_labels"])):
            synthesized_scenes.append({
                k: np.array(v[j])
                for k, v in boxes.items()
            })
        
        for s in range(len(synthesized_scenes)):
            
            box_scene = { k: v[s][None,:,:] for k, v in boxes.items() }

            try:
                door_boxes = door_boxes_lst[s][None,:,:].to(device)   
                door_overlap_ratio = calc_overlap_rotate_bbox_doors(synthesized_scenes[s:s+1],door_boxes)
            except:
                door_overlap_ratio = 0
                door_boxes = None

            if cfg.evaluation.overlap_type == "rotated_bbox":
                overlap_ratio = calc_overlap_rotate_bbox(synthesized_scenes[s:s+1])
            else:    
                overlap_ratio = calc_overlap(synthesized_scenes[s:s+1],classes,cfg)
            ratio = calc_wall_overlap(synthesized_scenes[s:s+1], floor_plan_mask_list[s:s+1], floor_plan_centroid_list[s:s+1], 
                                      cfg, robot_real_width=0.3, calc_object_area=True, classes=classes)
            walkable_average_rate, accessable_rate,box_wall_rate, object_area_ratio = ratio
           
            # if door_overlap_ratio>0.01 or overlap_ratio > 0 or box_wall_rate>0 or object_area_ratio<0.25:
            #     continue

            if overlap_ratio > 0 : #or box_wall_rate>0 or object_area_ratio<0.25:
                continue

            # print("overlap_ratio ",overlap_ratio)
            # print("box_wall_rate ",box_wall_rate)
            # print("walkable_average_rate ",walkable_average_rate)
            # print("accessable_rate ",accessable_rate)
            scene_info_bbox = export_scene_info(boxes,dataset,cfg,center_info, mask_name_lst, s)
            if vis:
                # show_scene(box_scene,objects_dataset,dataset,cfg,floor_plan_lst[s:s+1],None,room_outer_box[s:s+1],gapartnet_dataset,mask_name_lst[s:s+1])
                show_scene_bbox(scene_info_bbox,floor_plan_lst[s:s+1])
    return 

def export_scene_info(boxes,dataset,cfg,center_info, mask_name_lst=[],idx=0):
    bbox_params_t = torch.cat([
        boxes["class_labels"],
        boxes["translations"],
        boxes["sizes"],
        boxes["angles"],
        boxes["objfeats_32"]  #add 
    ], dim=-1).cpu().numpy()
    
    classes = np.array(dataset.class_labels)
    roomtype = cfg.task.dataset.filter_fn.split("_")[-1]
    
    i = idx
    bbox_param = bbox_params_t[i:i+1]
    # # [1] retrieve objects
    # renderables, trimesh_meshes, _,_ , scene_info = get_textured_objects(
    #     bbox_param, objects_dataset, gapartnet_dataset, classes, cfg
    # )
    # path_to_json = "{}/{}".format(
    #     cfg.ai2thor.path_to_result,
    #     mask_name_lst[i][:-4]+"_"+roomtype+".json"
    # )
    # with open(path_to_json,"w") as f:
    #     json.dump(scene_info,f,indent=4)

    # [2] only compute bbox
    houseinfo = mask_name_lst[i][:-4].split("_")
    House_number = "_".join(houseinfo[:2])
    obj = "_".join(houseinfo[2:])+".obj"
    obj_name = f"{House_number}/{obj}"

    scene_info_bbox = get_bbox_objects(bbox_param, classes, cfg)

    center = center_info[obj_name]["center"]
    scene_info_bbox["floor_center"] = center

    doors = center_info[obj_name]["door"]
    door_boxes = calc_door_box(doors,center)
    scene_info_bbox["door_box"] = door_boxes.numpy().tolist()
    
    newpath = "{}/{}".format(cfg.ai2thor.path_to_result,House_number)
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    os.system(f"cp {cfg.ai2thor.path_to_ai2thor}/{obj_name[:-4]}* {cfg.ai2thor.path_to_result}/{House_number}/")

    
    path_to_json = "{}/{}/{}".format(
        cfg.ai2thor.path_to_result,
        House_number,
        mask_name_lst[i][:-4]+"_"+roomtype+"_bbox.json"
    )
    with open(path_to_json,"w") as f:
        json.dump(scene_info_bbox,f,indent=4)
    
    return scene_info_bbox


def show_scene_bbox(scene_info_bbox,floor_plan_lst=None):
    
    import math
    import open3d as o3d
    if floor_plan_lst:
        theta = math.pi*3/2
        R = np.zeros((3, 3))
        R[2, 2] = np.cos(theta)
        R[2, 1] = -np.sin(theta)
        R[1, 2] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        R[0, 0] = 1.
        #floor plan mesh
        floor_plan = floor_plan_lst[0]
        floor_plan.affine_transform(R=R)
        points,faces = floor_plan.to_points_and_faces()
        mesh_floor_plan = o3d.geometry.TriangleMesh()
        mesh_floor_plan.vertices = o3d.utility.Vector3dVector(points)
        mesh_floor_plan.triangles = o3d.utility.Vector3iVector(faces)
        mesh_floor_plan = o3d.t.geometry.TriangleMesh.from_legacy(mesh_floor_plan)
    else:
        mesh_floor_plan = None

    #visualize bboxes
    bboxes = []
    names = []
    for obj in scene_info_bbox["objects"]:
        bbox = obj["position"]+[s*2 for s in obj["scale"]]+[obj["rot_degree"]]
        bboxes.append(bbox)
        names.append(obj["label"])
    bboxes = np.array(bboxes)

    from utils.open3d_vis_utils import draw_box_label
    draw_box_label(bboxes,names,mesh_floor_plan)
        
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