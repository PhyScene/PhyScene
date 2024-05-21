# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating scenes with procthor floor plan."""
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

    classes = np.array(dataset.class_labels)
    vis = False
    evaluate(network,dataset,config,raw_dataset,dataset,device,vis=vis)
    return 

def evaluate(network,dataset,cfg,raw_dataset,ground_truth_scenes,device,vis:True):
    classes = np.array(dataset.class_labels)
    print('class labels:', classes, len(classes))
    synthesized_scenes = []
    batch_size = 128
    
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

    #### limit room lst
    last_dir = "/home/yandan/workspace/PhyScene/ai2thor/generate"
    next_dir = "/home/yandan/workspace/PhyScene/ai2thor/generate_filterGPN"
    file_lst = os.listdir(last_dir)
    # exist_lst = os.listdir(next_dir)
    # good_room = []
    # for file in file_lst:
    #     if file in exist_lst:
    #         continue
    #     if file[-5:]==".json":
    #         roomname = file.split("_")[:-1]
    #         roomname = roomname[0]+"_"+roomname[1]+"/"+roomname[2]+"_"+roomname[3]+".obj"
    #         good_room.append(roomname)

    # for i in range(268,len(center_info)//batch_size+1):
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

        for  j in range(batch_size): 
            idx = i*batch_size + j

            if idx<len(center_info.keys()):
                obj_name = list(center_info.keys())[idx]
            # if idx<len(good_room):
            #     obj_name = good_room[idx]
            else:
                break
            # obj_name = "House_35/room_7.obj"
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
            if overlap_ratio > 0 or box_wall_rate>0 or object_area_ratio<0.25:
                continue

            # print("overlap_ratio ",overlap_ratio)
            # print("box_wall_rate ",box_wall_rate)
            # print("walkable_average_rate ",walkable_average_rate)
            # print("accessable_rate ",accessable_rate)
            
            export_scene_info(boxes,dataset,cfg,center_info, mask_name_lst, floor_plan_lst, s, objects_dataset, gapartnet_dataset,retrieve=True,vis=vis)
           
            
    return 

def export_scene_info(boxes,dataset,cfg,center_info, mask_name_lst=[],floor_plan_lst=[],idx=0, objects_dataset=None, gapartnet_dataset=None, retrieve=False,vis=False):
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

    houseinfo = mask_name_lst[i][:-4].split("_")
    House_number = "_".join(houseinfo[:2])

    if retrieve:
        # [1] retrieve objects
        renderables, trimesh_meshes, _,_ , scene_info = get_textured_objects(
            bbox_param, objects_dataset, gapartnet_dataset, classes, cfg
        )
        path_to_json = "{}/{}".format(
            cfg.ai2thor.path_to_result,
            mask_name_lst[i][:-4]+"_"+roomtype+".json"
        )
        print(path_to_json)
        with open(path_to_json,"w") as f:
            json.dump(scene_info,f,indent=4)

        if vis:
            floor_plan = floor_plan_lst[i]
            import math
            theta = math.pi*3/2
            R = np.zeros((3, 3))
            R[2, 2] = np.cos(theta)
            R[2, 1] = -np.sin(theta)
            R[1, 2] = np.sin(theta)
            R[1, 1] = np.cos(theta)
            R[0, 0] = 1.
            floor_plan.affine_transform(R=R)
            renderables += [floor_plan]
            from scripts.eval.calc_ckl import show_renderables
            show_renderables(renderables)

    else:
        # [2] only compute bbox
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

        if vis:
            from generate_bbox import show_scene_bbox
            show_scene_bbox(scene_info_bbox,floor_plan_lst[i:i+1])

def show_scene(boxes,objects_dataset,dataset,cfg,floor_plan_lst,tr_floor, return_bbox=True,
               without_floor=False, room_outer_box=None,gapartnet_dataset=None,mask_name_lst=[]):
    bbox_params_t = torch.cat([
        boxes["class_labels"],
        boxes["translations"],
        boxes["sizes"],
        boxes["angles"],
        boxes["objfeats_32"]  #add 
    ], dim=-1).cpu().numpy()
    
    scene_top2down = init_render_scene(cfg,gray=True,room_side=6.2,size=(512,512))

    classes = np.array(dataset.class_labels)
    for i in range(bbox_params_t.shape[0]):
        bbox_param = bbox_params_t[i:i+1]

        houseinfo = mask_name_lst[i][:-4].split("_")
        House_number = "_".join(houseinfo[:2])
        roomtype = cfg.task.dataset.filter_fn.split("_")[-1]

        # # [1] retrieve objects
        if not return_bbox:
            renderables, trimesh_meshes, _,_ , scene_info = get_textured_objects(
                bbox_param, objects_dataset, gapartnet_dataset, classes, cfg
            )
           

        # [2] only compute bbox
        else:
            scene_info_bbox = get_bbox_objects(bbox_params_t, classes, cfg)
            path_to_json = "{}/{}".format(
                cfg.ai2thor.path_to_result,
                mask_name_lst[i][:-4]+"_"+roomtype+"_bbox.json"
            )
            with open(path_to_json,"w") as f:
                json.dump(scene_info_bbox,f,indent=4)

        
        if room_outer_box != None:
            render_boxes = []
            # gt_boxes = []
            for j in range(room_outer_box.shape[1]):
                # if i==1 or i==5: continue
            
                box = room_outer_box[0][j].cpu().numpy().copy() 
                # xyz->xzy
                box[1] = room_outer_box[0][j][2]
                box[2] = room_outer_box[0][j][1]
                box[4] = room_outer_box[0][j][5]
                box[5] = room_outer_box[0][j][4]
                # box[4] = 0.1
                box[3:6] = box[3:6]/2

                # gt_boxes.append([box[0],box[1],box[2],box[3],box[4],box[5],0])
                boxmesh = Mesh.from_boxes(box[:3][None,:],box[3:6][None,:],(1,1,1))
                render_boxes.append(boxmesh)
            # gt_boxes = np.array(gt_boxes)
            # from utils.open3d_vis_utils import draw_box_label
            # vis = draw_box_label(gt_boxes, (0, 0, 1))
            # show_scene(render_boxes)
            renderables += render_boxes
        

        
        import math
        theta = math.pi*3/2
        R = np.zeros((3, 3))
        R[2, 2] = np.cos(theta)
        R[2, 1] = -np.sin(theta)
        R[1, 2] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        R[0, 0] = 1.
        # Apply the transformations in order to correctly position the mesh
        

        if not without_floor:
            # if not without_floor:
            floor_plan = floor_plan_lst[i]
            floor_plan.affine_transform(R=R)
            renderables += [floor_plan]
            # trimesh_meshes += tr_floor

        if cfg.evaluation.render2img:
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
        # show_renderables(renderables)

        
        save_mesh = False
        if (not return_bbox) and save_mesh:
            if trimesh_meshes is not None:
                # Create a trimesh scene and export it
               
                path_to_scene = "debug.obj"
                # whole_scene_mesh = merge_meshes( trimesh_meshes )
                whole_scene_mesh = trimesh.util.concatenate(trimesh_meshes)
                whole_scene_mesh.export(path_to_scene)
    return

def sample_class_labels(class_labels):
    # Extract the sizes in local variables for convenience
    L, C = class_labels.shape
    # Sample the class
    sampled_classes = np.argmax(class_labels,axis=-1)
    return np.eye(C)[sampled_classes]

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