# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for retrieve object scenes using previously saved bbox scenes.
   And save each object scene into a json file. """
import logging
import numpy as np
import torch
import json
import os
import hydra
from omegaconf import DictConfig
import sys
sys.path.insert(0,sys.path[0]+"/../../")
from datasets.base import filter_function, get_dataset_raw_and_encoded
from datasets.threed_future_dataset import ThreedFutureDataset
from utils.utils_preprocess import render as render_top2down
from utils.utils import get_textured_objects
from scripts.eval.calc_ckl import show_renderables
from scripts.eval.calc_ckl import init_render_scene
from utils.utils_preprocess import floor_plan_from_scene, room_outer_box_from_scene
from scripts.eval.calc_ckl import map_scene_id, get_objects
import trimesh

def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


@hydra.main(version_base=None, config_path="./configs", config_name="default")
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

    vis = True
    evaluate(None,dataset,config,raw_dataset,dataset,device,vis=vis)
    return 

def evaluate(network,dataset,cfg,raw_dataset,ground_truth_scenes,device,vis:True):
    classes = np.array(dataset.class_labels)
    if vis:
        # Build the dataset of 3D models
        objects_dataset = ThreedFutureDataset.from_pickled_dataset(
            cfg.task.path_to_pickled_3d_futute_models
        )
        print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))
    print('class labels:', classes, len(classes))
    synthesized_scenes = []
    floor_plan_mask_list = []
    floor_plan_centroid_list = []
    batch_size = cfg.task.test.batch_size
    mapping = map_scene_id(raw_dataset)

    # if cfg.evaluation.visual:
    objects_dataset, gapartnet_dataset = get_objects(cfg)
    
    if cfg.evaluation.save_result:
        boxes_save = []
        scene_id_save = []

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
    for j in range(len(boxes["class_labels"][:200])):
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

    for s in range(len(synthesized_scenes)):
        box_scene = { k: torch.Tensor(v[s][None,:,:]) for k, v in boxes.items() if k!="scene_ids" }
        door_boxes = None
        show_scene(box_scene,objects_dataset,dataset,cfg,floor_plan_lst[s:s+1],tr_floor_lst[s:s+1],door_boxes,gapartnet_dataset,scene_id_lst[s:s+1])
    return 

def show_scene(boxes,objects_dataset,dataset,cfg,floor_plan_lst,tr_floor,room_outer_box=None,gapartnet_dataset=None,mask_name_lst=[]):
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

        # scene_info = dict()
        print("showing " ,mask_name_lst[i])

        renderables, trimesh_meshes, _,_ , scene_info = get_textured_objects(
            bbox_param, objects_dataset, gapartnet_dataset, classes, cfg
        )
        path_to_json = "{}/{}".format(
            cfg.ThreeDFRONT.path_to_result,
            mask_name_lst[i]+".json"
        )
        with open(path_to_json,"w") as f:            
            json.dump(scene_info,f,indent=4)

        # if not without_floor:
        floor_plan = floor_plan_lst[i][0]
        if True:
            renderables += [floor_plan]
            # trimesh_meshes += tr_floor
        
        #export floor mesh
        # path_to_scene = "debug.obj"
        path_to_floor = "{}/{}".format(
            cfg.ThreeDFRONT.path_to_result,
            "room_"+mask_name_lst[i]+".obj"
        )
        # tr_floor[0][0].export(path_to_floor)

        if cfg.evaluation.render2img:

            path_to_image = "{}/{}".format(
                cfg.ThreeDFRONT.path_to_result,
                mask_name_lst[i]+".png"
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

        if trimesh_meshes is not None:
                # Create a trimesh scene and export it
                path_to_scene = "{}/{}".format(
                    cfg.ThreeDFRONT.path_to_result,
                    mask_name_lst[i]+".obj"
                )
                # if not os.path.exists(path_to_objs):
                #     os.mkdir(path_to_objs)
                # path_to_scene = "debug.obj"
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