# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys
from scipy.ndimage import binary_dilation
import cv2
import numpy as np
import torch
import json
sys.path.insert(0,sys.path[0]+"/../../")
from utils.utils_preprocess import floor_plan_from_scene, room_outer_box_from_scene

from datasets.base import filter_function, get_dataset_raw_and_encoded
from datasets.threed_front import ThreedFront
from datasets.threed_future_dataset import ThreedFutureDataset

from models.networks import build_network

from simple_3dviz import Scene
import pickle
from datasets.gapartnet_dataset import GAPartNetDataset
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger

import numpy as np
from models.networks import build_network, optimizer_factory, schedule_factory, adjust_learning_rate
from datasets.base import filter_function, get_dataset_raw_and_encoded

import hydra 
from omegaconf import DictConfig, OmegaConf
# from generate_diffusion import show_scene
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from utils.utils import get_textured_objects
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.window import show
from utils.utils_preprocess import render as render_top2down
from pyrr import Matrix44
from simple_3dviz.renderables.mesh import Mesh
from kaolin.ops.mesh import check_sign, index_vertices_by_faces,face_normals
import open3d as o3d
import open3d.visualization.gui as gui
from datasets.gapartnet_dataset import GAPartNetDataset
from utils.utils_preprocess import merge_meshes
import trimesh

from utils.overlap import bbox_overlap, calc_overlap_rotate_bbox, calc_wall_overlap, voxel_grid_from_mesh

from scripts.eval.walkable_metric import cal_walkable_metric
from scripts.eval.walkable_map_visual import walkable_map_visual

IMAGE_SIZE = 1024
def map_scene_id(raw_dataset):
    mapping = dict()
    for i in range(len(raw_dataset)):
        mapping[str(raw_dataset[i].scene_id)] = i
    return mapping


def get_objects(cfg):
    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        cfg.task.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    
    # Build the dataset of Garpartnet
    if cfg.evaluation.gapartnet:
        pickled_GPN_dir = cfg.GAPartNet.pickled_GPN_dir
        pickled_GPN_path = "{}/gapartnet_model.pkl".format(pickled_GPN_dir)
        
        if os.path.exists(pickled_GPN_path):
            gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
        else:
            gapartnet_dataset = GAPartNetDataset(cfg,remove_largeopen=True)
            with open(pickled_GPN_path, "wb") as f:
                pickle.dump(gapartnet_dataset, f)
        print("Loaded {} Gapartnet models".format(len(gapartnet_dataset.objects)))
    else:
        gapartnet_dataset = None
    
    return objects_dataset, gapartnet_dataset

def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def show_renderables(renderables):
        window_size="256,256"
        camera_position="0,15,0"
        camera_target="0.0,0.0,0.0"
        up_vector="0,0,-1" 
        background= "0.5,0.5,0.5,0.5"
        window_size=tuple(map(int, window_size.split(",")))
        camera_position=tuple(map(float, camera_position.split(",")))
        camera_target=tuple(map(float, camera_target.split(",")))
        up_vector=tuple(map(float, up_vector.split(",")))
        background=tuple(map(float, background.split(",")))

        show(
            renderables,
            behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
            size=window_size,
            camera_position=camera_position,
            camera_target=camera_target,
            up_vector=up_vector,
            background=background,
            title="Generated Scene"
        )
        return


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

    if cfg.evaluation.generate_result_json:
        cfg.evaluation.load_result = False
        cfg.evaluation.save_result = True
        # cfg.evaluation.visual = False
    else:
        cfg.evaluation.load_result = True
        cfg.evaluation.save_result = False
        if "nomask" in cfg.evaluation.jsonname or "no_mask" in cfg.evaluation.jsonname or "diffuscene" in cfg.evaluation.jsonname:
            cfg.evaluation.without_floor=True

    #train+test
    path_to_bounds = cfg.task.dataset.path_to_bounds
    raw_dataset, ground_truth_scenes = get_dataset_raw_and_encoded(
        config.task["dataset"],
        filter_fn=filter_function(
            config.task["dataset"],
            split=["train", "val","test"]
        ),
        split=["train", "val","test"],
        path_to_bounds=path_to_bounds
    )

    #train+test
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config.task["dataset"],
        filter_fn=filter_function(
            config.task["dataset"],
            split=["train", "val","test"]
        ),
        split=["train", "val","test"],
        path_to_bounds=path_to_bounds
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    #object dataset
    global objects_dataset
    global gapartnet_dataset
    objects_dataset, gapartnet_dataset = get_objects(cfg)

    if not cfg.evaluation.load_result:
        network, _, _ = build_network(
            dataset.feature_size, dataset.n_classes,
            config, weight_file, dataset, cfg, 
            objects_dataset, gapartnet_dataset, device=device
        )
        network.eval()
    else:
        network=None

    classes = np.array(dataset.class_labels)

    
    
    eva_result = evaluate(network,dataset,config,raw_dataset,ground_truth_scenes,device)
    CKL, overlap_ratio, walkable_average_rate, accessable_rate,box_wall_rate = eva_result
    # print('CKL ',CKL)
    # print('overlap_ratio ',overlap_ratio)
    # print(weight_file)

    return

def evaluate(network,dataset,cfg,raw_dataset, ground_truth_scenes, device):
    classes = np.array(dataset.class_labels)
    
    print('class labels:', classes, len(classes))
    synthesized_scenes = []
    floor_plan_mask_list = []
    floor_plan_centroid_list = []
    batch_size = cfg.task.test.batch_size
    mapping = map_scene_id(raw_dataset)

    # if cfg.evaluation.visual:
    #     objects_dataset, gapartnet_dataset = get_objects(cfg)
    
    if cfg.evaluation.save_result:
        boxes_save = []
        scene_id_save = []

    if not cfg.evaluation.load_result:
        idx = 0
        for i in range(cfg.task.evaluator.n_synthesized_scenes//batch_size+1):
            
            print("{} / {}:".format(
                i, cfg.task.evaluator.n_synthesized_scenes//batch_size+1)
            )
            # Get a floor plan
            room_lst = []
            floor_plan_lst = []
            floor_plan_mask_batch_list = []
            floor_plan_centroid_batch_list = []
            scene_idx_lst = []
            scene_id_lst = []
            room_outer_box_lst = []
            tr_floor_lst = []
            for  j in range(batch_size):
                scene_idx = np.random.choice(len(dataset))
                # scene_idx = 525  #50 #525  #1610   #3454 #3921
                print(j,scene_idx)
                scene_idx_lst.append(scene_idx)
                samples = dataset[scene_idx]
                # print("scene id ",scene_idx)

                current_scene = raw_dataset[scene_idx]
                scene_id_lst.append(current_scene.scene_id)

                floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                    current_scene, cfg.path_to_floor_plan_textures, no_texture=False
                )

                room_outer_box = room_outer_box_from_scene(current_scene)
                room_outer_box = torch.Tensor(room_outer_box[None,:,:]).to(device)
                
                room_outer_box_lst.append(room_outer_box)

                room_lst.append(room_mask)
                floor_plan_lst.append(floor_plan)
                floor_plan_mask_list.append(current_scene.floor_plan)
                floor_plan_mask_batch_list.append(current_scene.floor_plan)
                floor_plan_centroid_list.append(current_scene.floor_plan_centroid)
                floor_plan_centroid_batch_list.append(current_scene.floor_plan_centroid)
                tr_floor_lst.append(tr_floor)
            room_mask = torch.concat(room_lst)
            room_outer_box = torch.concat(room_outer_box_lst)

            bbox_params = network.generate_layout(
                room_mask=room_mask.to(device),
                room_outer_box=room_outer_box,
                floor_plan=floor_plan_mask_batch_list, 
                floor_plan_centroid=floor_plan_centroid_batch_list,
                batch_size = batch_size,
                num_points=cfg.task["network"]["sample_num_points"],
                point_dim=cfg.task["network"]["point_dim"],
                text=torch.from_numpy(samples['desc_emb'])[None, :].to(device) if 'desc_emb' in samples.keys() else None,
                device=device,
                clip_denoised=cfg.clip_denoised,
                batch_seeds=torch.arange(i, i+1),
                keep_empty = True
            )

            boxes = dataset.post_process(bbox_params)
            if cfg.evaluation.save_result:
                boxes_save.append(boxes)
                scene_id_save +=  scene_id_lst
            if cfg.evaluation.visual:
                # show_scene(boxes,objects_dataset,dataset,gapartnet_dataset, cfg,floor_plan_lst,tr_floor_lst, room_outer_box_lst,scene_id_lst,scene_idx_lst,idx,render2img=render2img)
                render_scene_all(boxes,objects_dataset,dataset,gapartnet_dataset,cfg,room_lst,floor_plan_lst,floor_plan_mask_batch_list,floor_plan_centroid_batch_list,
                        tr_floor_lst,room_outer_box_lst,scene_id_lst,scene_idx_lst)
            for j in range(boxes["class_labels"].shape[0]):
                synthesized_scenes.append({
                    k: v[j].cpu().numpy()
                    for k, v in boxes.items()
                })
            idx += batch_size

        if cfg.evaluation.save_result:
            result = dict()
            result["class_labels"] = np.concatenate([boxes["class_labels"] for boxes in boxes_save],axis = 0).tolist()
            result["translations"] = np.concatenate([boxes["translations"] for boxes in boxes_save],axis = 0).tolist()
            result["sizes"] = np.concatenate([boxes["sizes"] for boxes in boxes_save],axis = 0).tolist()
            result["angles"] = np.concatenate([boxes["angles"] for boxes in boxes_save],axis = 0).tolist()
            result["objfeats_32"] = np.concatenate([boxes["objfeats_32"] for boxes in boxes_save],axis = 0).tolist()
            result["objectness"] = np.concatenate([boxes["objectness"] for boxes in boxes_save],axis = 0).tolist()
            result["scene_ids"] = scene_id_save
            out_file = open(cfg.evaluation.jsonname, "w")         
            json.dump(result, out_file, indent = 4)
            out_file.close()
            # sys.exit()
    else:
        print("loading json ",cfg.evaluation.jsonname )
        f = open(cfg.evaluation.jsonname,"r")
        boxes = json.load(f)
        for k in boxes.keys():
            boxes[k] = np.array(boxes[k])

        synthesized_scenes = []
        floor_plan_mask_list = []
        floor_plan_centroid_list = []
        room_lst = []
        floor_plan_lst = []
        floor_plan_mask_batch_list = []
        floor_plan_centroid_batch_list = []
        scene_idx_lst = []
        scene_id_lst = []
        room_outer_box_lst = []
        tr_floor_lst = []
        for j in range(len(boxes["class_labels"])):
            scene_idx_lst.append(j)
            synthesized_scenes.append({
                k: np.array(v[j])
                for k, v in boxes.items()
            })
            scene_id = boxes["scene_ids"][j]
            data_idx = mapping[scene_id]
            current_scene = raw_dataset[data_idx]
            if current_scene.scene_id != scene_id:
                print('error')
                breakpoint()
            scene_idx_lst.append(data_idx)
            scene_id_lst.append(scene_id)
            floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                current_scene, cfg.path_to_floor_plan_textures, no_texture=False
            )
            floor_plan_lst.append(floor_plan)
            tr_floor_lst.append(tr_floor)
            room_outer_box = room_outer_box_from_scene(current_scene)
            room_outer_box = torch.Tensor(room_outer_box[None,:,:]).to(device)
            
            room_outer_box_lst.append(room_outer_box)
            room_lst.append(room_mask)
            floor_plan_mask_list.append(current_scene.floor_plan)
            floor_plan_centroid_list.append(current_scene.floor_plan_centroid)

            if "atiss" in cfg.evaluation.jsonname or "3dfront" in cfg.evaluation.jsonname:
                cfg.task.dataset.use_feature = False

        
    #generate image
    if cfg.evaluation.visual:
        render_scene_all(boxes,objects_dataset,dataset,gapartnet_dataset,cfg,room_lst,floor_plan_lst,floor_plan_mask_list,floor_plan_centroid_list,
                        tr_floor_lst,room_outer_box_lst,scene_id_lst,scene_idx_lst)
    #collision rate
    print("collision type", cfg.evaluation.overlap_type)
    if cfg.evaluation.overlap_type == "rotated_bbox":
        overlap_ratio = calc_overlap_rotate_bbox(synthesized_scenes)
    else:    
        #"bbox_no_direction", "mesh", "grid_for_depth"
        overlap_ratio = calc_overlap(synthesized_scenes,classes,cfg)
    print(cfg.evaluation.jsonname)
    # print(overlap_ratio)

    CKL, gt_class_labels, syn_class_labels = calc_CKL(synthesized_scenes,ground_truth_scenes,classes)
    print(cfg.evaluation.jsonname)
    print(CKL)

    walkable_average_rate, accessable_rate,box_wall_rate = calc_wall_overlap(synthesized_scenes, floor_plan_mask_list, floor_plan_centroid_list, cfg, robot_real_width=0.3,classes=classes)

    
    return CKL, overlap_ratio, walkable_average_rate, accessable_rate,box_wall_rate




def calc_overlap(synthesized_scenes,classes,cfg,visualize_overlap = False):
    device = 'cuda'
    class_num = classes.shape[0]-1
    a = 0
    overlap_cnt_total = 0
    obj_cnt_total = 0
    overlap_scene = 0
    scene_cnt = 0

    overlap_area = 0
    overlap_area_max = 0
    obj_overlap_cnt = 0
    
    # objects_dataset, gapartnet_dataset = get_objects(cfg)

    cnt = min(100,len(synthesized_scenes))
    for d in synthesized_scenes[:cnt]:
        a += 1
        # if a>3:
        #     break
        # if a!=3:
        #     continue
        max_classes = np.argmax(d["class_labels"],axis=-1)
        valid_idx = max_classes!=class_num

        #visualize
        print("visualizing scene ", a ,"/",cnt)
        boxes = d
        if not cfg.task.dataset.use_feature:  #atiss
            bbox_params = np.concatenate([
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"]
            ], axis=-1)[None,:,:]
        elif boxes["class_labels"].shape[1] == 21 or boxes["class_labels"].shape[1] ==24:  #diffuscene
            bbox_params = np.concatenate([
                boxes["class_labels"],
                boxes["objectness"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"],
                boxes["objfeats_32"]  #add 
            ], axis=-1)[None,:,:]
        else:  #ours
            bbox_params = np.concatenate([
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"],
                boxes["objfeats_32"]  #add 
            ], axis=-1)[None,:,:]

        renderables, _,_, renderables_remesh,_ = get_textured_objects(
            bbox_params, objects_dataset, gapartnet_dataset, classes, cfg
        )
        obj_cnt = len(renderables)
        
        overlap_flag = np.zeros(obj_cnt)

        if cfg.evaluation.overlap_type == "grid_for_depth":
            voxels = []
            gridsize = 0.01  #1cm
            for mesh in renderables_remesh:
                voxel = voxel_grid_from_mesh(mesh,device,gridsize)
                voxels.append(voxel)
        overlap_depths = np.zeros(obj_cnt)
        for i in range(obj_cnt):
            #obj1 use remesh
            if cfg.evaluation.overlap_type == "bbox_no_direction":
                ## collision metric 0 fast: bbox_no_direction
                for j in range(obj_cnt):
                    if i==j:
                        continue
                    if not bbox_overlap(renderables_remesh[i],renderables[j]):
                        continue
                    else:
                        overlap_flag[i] = 1
                        overlap_flag[j] = 1
            else:
                try:
                    points,faces = renderables_remesh[i].to_points_and_faces()
                except:
                    mesh_cnt = len(renderables_remesh[i].renderables)
                    points = []
                    faces = []
                    point_cnt = 0

                
                    for s in range(mesh_cnt):
                        p,f = renderables_remesh[i].renderables[s].to_points_and_faces()
                        points.append(p)
                        faces.append(f+point_cnt)
                        point_cnt += p.shape[0]
                    points = np.concatenate(points,axis=0)
                    faces = np.concatenate(faces,axis=0)
            
                if visualize_overlap:
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(points)
                    mesh.triangles = o3d.utility.Vector3iVector(faces)
                    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                    pcd0 = o3d.geometry.PointCloud()
                    pcd0.points = o3d.utility.Vector3dVector(points)
                    pcd0.colors = o3d.utility.Vector3dVector([[0,1,0]])


                verts = torch.tensor(points,device = device).unsqueeze(0)
                faces = torch.tensor(faces,device = device).long()
                face_verts = index_vertices_by_faces(verts,faces)
                normals = face_normals(face_verts,unit=True)

                if cfg.evaluation.overlap_type == "mesh":
                    ## collison metric 1 : original mesh and pcd
                    for j in range(i+1,obj_cnt):
                        if overlap_flag[i] and overlap_flag[j]:
                            continue    
                        #obj2
                        if not bbox_overlap(renderables_remesh[i],renderables[j]):
                            continue
                        try:
                            points = renderables[j].to_points_and_faces()[0][None,:,:]
                        except:
                            mesh_cnt = len(renderables[j].renderables)
                            points = [renderables[j].renderables[s].to_points_and_faces()[0] for s in range(mesh_cnt)]
                            points = np.concatenate(points,axis=0)[None,:,:]
                        pointscuda = torch.tensor(points,device = device)
                        occupancy = check_sign(verts,faces,pointscuda)
                        
                        if occupancy.max()>0:
                            # print((occupancy.sum()/occupancy.shape[1]).item())
                            overlap_flag[i] = 1
                            overlap_flag[j] = 1
                            
                            # if visualize_overlap:
                            #     pcd = o3d.geometry.PointCloud()
                            #     pcd.points = o3d.utility.Vector3dVector(points[0])
                            #     p = points[0][occupancy.cpu().numpy()[0]==1]
                            #     pcdinside = o3d.geometry.PointCloud()
                            #     pcdinside.points = o3d.utility.Vector3dVector(p)
                            #     # pcdinside.colors = o3d.utility.Vector3dVector([[1,0,0]])
                            #     pcdinside.paint_uniform_color([1, 0, 0])
                            #     draw_box_label(pcd0,pcd,pcdinside)
                elif cfg.evaluation.overlap_type == "grid_for_depth":
                    ## collision metric 2: grid & mesh, calc collision depth
                    for j in range(obj_cnt):
                        if i==j:
                            continue
                        if not bbox_overlap(renderables_remesh[i],renderables[j]):
                            continue
                        points = voxels[j][None,:,:]
                        pointscuda = torch.tensor(points,device = device)
                        occupancy = check_sign(verts,faces,pointscuda)
                        if occupancy.max()>0:
                            overlap_flag[i] = 1
                            overlap_flag[j] = 1

                            # col_rate = occupancy.sum()/occupancy.shape[1]
                            col_area = occupancy.sum()
                            overlap_depths[j] = max(overlap_depths[j],col_area)

                            # visualize_overlap=True
                            # if visualize_overlap:
                            #     pcd = o3d.geometry.PointCloud()
                            #     pcd.points = o3d.utility.Vector3dVector(points[0])
                            #     p = points[0][occupancy.cpu().numpy()[0]==1]
                            #     pcdinside = o3d.geometry.PointCloud()
                            #     pcdinside.points = o3d.utility.Vector3dVector(p)
                            #     # pcdinside.colors = o3d.utility.Vector3dVector([[1,0,0]])
                            #     pcdinside.paint_uniform_color([1, 0, 0])
                            #     draw_box_label(pcd0,pcd,pcdinside)

        print(overlap_flag)
        print(overlap_depths)
        overlap_cnt_total += overlap_flag.sum()
        obj_cnt_total += obj_cnt

        overlap_scene += overlap_flag.sum()>0
        scene_cnt += 1

        overlap_area += overlap_depths.sum()
        overlap_area_max += overlap_depths.max()
        
        obj_overlap_cnt += sum(overlap_depths>0)
        
        # show_renderables(renderables)
    overlap_ratio = overlap_cnt_total/obj_cnt_total 
    print("overlap object: ",overlap_ratio, "cnt ",overlap_cnt_total,"/",obj_cnt_total)  
    overlap_scene_rate = overlap_scene/scene_cnt  
    print( "overlap scene rate: ",overlap_scene_rate)
    print( "overlap_area_mean: ",overlap_area/obj_cnt_total)
    print( "overlap_area_max : ",overlap_area_max/scene_cnt)
    print( "overlap_area_mean_only_overlaped : ",overlap_area/obj_overlap_cnt)
   
    return overlap_ratio,overlap_scene_rate





def draw_box_label( pcd0, pointcloud=None, pcdinside=None,mesh=None):
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.show_settings = True
        if mesh is not None:
            vis.add_geometry("mesh",mesh)
        vis.add_geometry("pcd0",pcd0)
        if pointcloud is not None:
            vis.add_geometry("pointcloud",pointcloud)
        if pcdinside is not None:
            vis.add_geometry("pcdinside",pcdinside)

        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

        return vis

def calc_CKL(synthesized_scenes,ground_truth_scenes,classes): 
    # Firstly compute the frequencies of the class labels
    class_num = classes.shape[0]-1
    valid_cnt_total = 0
    
    label_cnt_total = np.zeros(class_num)
    for i in range(len(ground_truth_scenes)):
        d = ground_truth_scenes[i]
        valid_idx = d["class_labels"][:,-1]!=1
        valid_cnt = sum(valid_idx)
        valid_cnt_total += valid_cnt
        d["class_labels"] = (d["class_labels"]+1)/2
        label_cnt = d["class_labels"][valid_idx][:,:-1].sum(0) 
        label_cnt_total += label_cnt
    gt_class_labels = label_cnt_total/valid_cnt_total

    valid_cnt_total = 0
    label_cnt_total = np.zeros(class_num)
    for d in synthesized_scenes:
        max_classes = np.argmax(d["class_labels"],axis=-1)
        valid_idx = d["objectness"][:,0]<0 #False
        valid_cnt = sum(valid_idx)
        valid_cnt_total += valid_cnt
        if d["class_labels"].shape[1]==24 or d["class_labels"].shape[1]==21:
            class_labels = sample_class_labels(d["class_labels"][valid_idx])
        else:
            class_labels = sample_class_labels(d["class_labels"][valid_idx][:,:-1])
        label_cnt = class_labels.sum(0) 
        label_cnt_total += label_cnt
    syn_class_labels = label_cnt_total/valid_cnt_total


    assert 0.9999 <= gt_class_labels.sum() <= 1.0001
    assert 0.9999 <= syn_class_labels.sum() <= 1.0001
    CKL = categorical_kl(gt_class_labels, syn_class_labels)
    print('CKL:',CKL)

    # for c, gt_cp, syn_cp in zip(classes[:-1], gt_class_labels, syn_class_labels):
    #     print("{}: target: {} / synth: {}".format(c, gt_cp, syn_cp))

    return CKL, gt_class_labels, syn_class_labels

def show_scene(boxes,objects_dataset,dataset,gapartnet_dataset,cfg,floor_plan_lst,tr_floor_lst,room_outer_box_lst,scene_id_lst,scene_idx_lst,idx,render2img = True):
    bbox_params_t = torch.cat([
        boxes["class_labels"],
        boxes["translations"],
        boxes["sizes"],
        boxes["angles"],
        boxes["objfeats_32"]  #add 
    ], dim=-1).cpu().numpy()
    
    scene_top2down = init_render_scene(cfg,gray=True,size=IMAGE_SIZE)
    if not os.path.exists(cfg.evaluation.render_save_path):
        os.makedirs(cfg.evaluation.render_save_path)

    classes = np.array(dataset.class_labels)
    for i in range(bbox_params_t.shape[0]):
        bbox_param = bbox_params_t[i:i+1]
        idx += 1

        renderables, trimesh_meshes, _,_,_ = get_textured_objects(
            bbox_param, objects_dataset, gapartnet_dataset, classes, cfg
        )

        # visualize room outer box
        # if room_outer_box_lst!=None and 'optimizer' in cfg:
        #     render_boxes = []
        #     # gt_boxes = []
        #     room_outer_box = room_outer_box_lst[i].cpu().numpy().copy() 
        #     for j in range(room_outer_box.shape[1]):
        #         # if i==1 or i==5: continue
            
        #         box = room_outer_box[0][j].copy() 
        #         #xyz->xzy
        #         box[1] = room_outer_box[0][j][2]
        #         box[2] = room_outer_box[0][j][1]
        #         box[4] = room_outer_box[0][j][5]
        #         box[5] = room_outer_box[0][j][4]
        #         box[4] = 0.1
        #         box[3:6] = box[3:6]/2

        #         # gt_boxes.append([box[0],box[1],box[2],box[3],box[4],box[5],0])
        #         boxmesh = Mesh.from_boxes(box[:3][None,:],box[3:6][None,:],(0.3,0.3,0.3))
        #         render_boxes.append(boxmesh)
        #     renderables += render_boxes

        if not cfg.evaluation.without_floor:
            renderables += floor_plan_lst[i]
            trimesh_meshes += tr_floor_lst[i]
        
        if render2img:
            path_to_image = "{}/{}-{}-{:04d}.png".format(
                cfg.evaluation.render_save_path,
                scene_id_lst[i],
                scene_idx_lst[i],
                idx
            )
            render_top2down(
                scene_top2down,
                renderables,
                color=None,
                mode="shading",
                frame_path=path_to_image,
            )
        else:
            show_renderables(renderables)

        
        if cfg.evaluation.save_mesh:
            if trimesh_meshes is not None:
                # Create a trimesh scene and export it
                path_to_objs = os.path.join(
                    cfg.evaluation.render_save_path,
                    "scene_mesh",
                )
                if not os.path.exists(path_to_objs):
                    os.mkdir(path_to_objs)
                path_to_scene = "debug.obj"
                # whole_scene_mesh = merge_meshes( trimesh_meshes )
                whole_scene_mesh = trimesh.util.concatenate(trimesh_meshes)
                whole_scene_mesh.export(path_to_scene)
                # o3d.io.write_triangle_mesh(path_to_scene, whole_scene_mesh)
    return

def render_scene_all(boxes,objects_dataset,dataset,gapartnet_dataset,cfg,room_lst,
                        floor_plan_lst,floor_plan_mask_list,floor_plan_centroid_list,
                        tr_floor_lst,room_outer_box_lst,scene_id_lst,scene_idx_lst):
    if not cfg.task.dataset.use_feature:  #atiss
        bbox_params_t = np.concatenate([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], axis=-1)
    elif boxes["class_labels"].shape[2] == 21 or boxes["class_labels"].shape[2] ==24:  #diffuscene
        bbox_params_t = np.concatenate([
            boxes["class_labels"],
            boxes["objectness"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"],
            boxes["objfeats_32"]  #add 
        ], axis=-1)
    else:  #ours
        bbox_params_t = np.concatenate([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"],
            boxes["objfeats_32"]  #add 
        ], axis=-1)

    bbox_params_walk = np.concatenate([
        boxes["class_labels"],
        boxes["translations"],
        boxes["sizes"],
        boxes["angles"],
    ], axis=-1)

    
    # scene_top2down = init_render_scene(cfg,gray=True)
    
    if not os.path.exists(cfg.evaluation.render_save_path):
        os.makedirs(cfg.evaluation.render_save_path)

    classes = np.array(dataset.class_labels)
    scene_top2down = init_render_scene(cfg,gray=True)
    for i in range(bbox_params_t.shape[0]):
        print(i)    
        
        bbox_param = bbox_params_t[i:i+1]

        renderables, trimesh_meshes, _,_, _  = get_textured_objects(
            bbox_param, objects_dataset, gapartnet_dataset, classes, cfg
        )

        if not cfg.evaluation.without_floor:
            renderables += floor_plan_lst[i]
            trimesh_meshes += tr_floor_lst[i]

        if cfg.evaluation.render2img:
            path_to_objs = os.path.join(
                cfg.evaluation.render_save_path,
                "render",
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            path_to_image = "{}/render/{:04d}-{}.png".format(
                cfg.evaluation.render_save_path,
                scene_idx_lst[i],
                scene_id_lst[i],
            )
            frame = render_top2down(
                scene_top2down,
                renderables,
                color=None,
                mode="shading",
                frame_path=path_to_image,
            )

            if cfg.evaluation.save_walkable_map:
                #walkable map
                path_to_objs = os.path.join(
                    cfg.evaluation.render_save_path,
                    "walk",
                )
                if not os.path.exists(path_to_objs):
                    os.mkdir(path_to_objs)
                path_to_walk = "{}/walk/{:04d}-{}.png".format(
                    cfg.evaluation.render_save_path,
                    scene_idx_lst[i],
                    scene_id_lst[i],
                    
                )
                #calc (room mask -> 256*256 floor image) resize scale 
                room_mask = room_lst[i]
                max_scale = max(np.where(room_mask[0][0]==1)[0].max(), np.where(room_mask[0][0]==1)[1].max())
                min_scale = min(np.where(room_mask[0][0]==1)[0].min(), np.where(room_mask[0][0]==1)[1].min())
                scale = max((128-min_scale)/128, (max_scale-128)/128)
                if cfg.task.dataset.filter_fn =="threed_front_bedroom":
                    scale = scale*2

                #here we simply use the predicted bbox, instead of retrieved mesh, to calculated wakable map
                walkable_map_visual(bbox_params_walk[i,:,-7:], floor_plan_mask_list[i], floor_plan_centroid_list[i], scale, frame, robot_width_real=0.2,path_to_walk=path_to_walk)
        else:
            show_renderables(renderables)

        if cfg.evaluation.save_mesh:
            if trimesh_meshes is not None:
                # Create a trimesh scene and export it
                path_to_objs = os.path.join(
                    cfg.evaluation.render_save_path,
                    "scene_mesh",
                )
                if not os.path.exists(path_to_objs):
                    os.mkdir(path_to_objs)
                path_to_scene_dir = "{}/scene_mesh/{:04d}-{}".format(
                    cfg.evaluation.render_save_path,
                    scene_idx_lst[i],
                    scene_id_lst[i],
                )
                if not os.path.exists(path_to_scene_dir):
                    os.mkdir(path_to_scene_dir)
                path_to_scene = f"{path_to_scene_dir}/mesh.obj"
                
                # path_to_scene = "save_scene/scene_1/debug.obj"
                whole_scene_mesh = trimesh.util.concatenate(trimesh_meshes)
                whole_scene_mesh.export(path_to_scene)
    return

def get_floor_plan(raw_dataset,scene_id,cfg):
    for scene_idx in range(len(raw_dataset)):
        current_scene = raw_dataset[scene_idx]
        if current_scene.scene_id == scene_id:
            floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                current_scene, cfg.path_to_floor_plan_textures, no_texture=False
            )
            return floor_plan, tr_floor, room_mask
    return None
   
def init_render_scene(cfg,without_floor=False,gray=False,room_side=None,size=None):
    if gray:
        background = [1,1,1,1]
    else:
        background=[0,0,0,1]
    if size is None:
        size=(256, 256)
    if without_floor:
            scene_top2down = Scene(size=size, background=background)
    else:
        scene_top2down = Scene(size=size, background=background)
    scene_top2down.up_vector = (0,0,-1)
    scene_top2down.camera_target = (0, 0, 0)
    scene_top2down.camera_position = (0,4,0)
    scene_top2down.light = (0,4,0)
    if room_side is None:
        if 'bedroom' in cfg.task.dataset.filter_fn:
            room_side = 3.1
        else:
            room_side = 6.2
    scene_top2down.camera_matrix = Matrix44.orthogonal_projection(
        left=-room_side, right=room_side,
        bottom=room_side, top=-room_side,
        near=0.1, far=6
    )
    return scene_top2down


def sample_class_labels(class_labels):
    # Extract the sizes in local variables for convenience
    L, C = class_labels.shape
    # Sample the class
    sampled_classes = np.argmax(class_labels,axis=-1)
    return np.eye(C)[sampled_classes]

if __name__ == "__main__":
    import random
    import numpy as np
    seed = 2027
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
