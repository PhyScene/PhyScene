"""Script used for preprocess the Procthor data 
    1) project floor plan mesh into room mask images
    2) save room info into cfg.ai2thor.path_to_center_info
"""

import os
import sys
sys.path.insert(0,sys.path[0]+"/../../")
import hydra
from omegaconf import DictConfig
import json
from utils.utils_preprocess import  scene_from_cfg, render
import numpy as np
from numpy import linalg
from simple_3dviz import Mesh, Scene
from tqdm import tqdm
@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess_data")
def main(cfg: DictConfig):
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    
    scene_black = scene_from_cfg(cfg,background=[0,0,0,1])

    ai2thor_dir = cfg.ai2thor.path_to_ai2thor
    house_lst = os.listdir(ai2thor_dir)

    center_info = dict()

    for housename in tqdm(house_lst):
        if "House" not in housename:
            continue

        jsonname = cfg.ai2thor.path_to_json+"/"+housename+".json"
        roominfo_map = dict()
        
        with open(jsonname) as f:
            j = json.load(f)
            rooms = j["rooms"]
            for room in rooms:
                room_id = "_".join(room["id"].split("|"))
                roomType = room["roomType"].lower()
                roominfo_map[room_id] = {"roomType":roomType,
                                         "door":[]}
            doors = j["doors"]
            
            walls = dict()
            for item in j["walls"]:
                walls[item["id"]] = dict()
                walls[item["id"]]["polygon"] = item["polygon"]
                walls[item["id"]]["p0"] = np.array([item["polygon"][0]["x"],item["polygon"][0]["y"],item["polygon"][0]["z"]])
                walls[item["id"]]["p1"] = np.array([item["polygon"][1]["x"],item["polygon"][1]["y"],item["polygon"][1]["z"]])
                p0p1 = np.array(walls[item["id"]]["p1"]-walls[item["id"]]["p0"])
                walls[item["id"]]["p0p1_normalize"] = p0p1/linalg.norm(p0p1)


            for door in doors:
                double_or_single = "double" if "Double" in door["assetId"] else "single"
                wall0 = walls[door["wall0"]]
                p0 = wall0["p0"]
                p0p1_normalize = wall0["p0p1_normalize"]
                holePolygon_p0 = p0 + p0p1_normalize*door["holePolygon"][0]["x"]
                holePolygon_p1 = p0 + p0p1_normalize*door["holePolygon"][1]["x"]
                # ymax
                holePolygon_p1[1] = door["holePolygon"][1]["y"]
                # x = -x
                holePolygon_p0[0] = -holePolygon_p0[0]
                holePolygon_p1[0] = -holePolygon_p1[0]
                holePolygon = [list(holePolygon_p0),list(holePolygon_p1)]
                # wall0 = list(map(float,door["wall0"].split("|")[-4:]))
                # xmin,zmin,xmax,zmax = wall0
                # ymin = 0
                # ymax = door["holePolygon"][1]["y"]
                # if zmax-zmin>xmax-xmin:
                #     zmax = zmin + door["holePolygon"][1]["x"]
                #     zmin +=  door["holePolygon"][0]["x"]
                # else:
                #     xmax = xmin + door["holePolygon"][1]["x"]
                #     xmin += door["holePolygon"][0]["x"]
                # holePolygon = [[xmin,ymin,zmin],[xmax,ymax,zmax]]
                
                room0 = "_".join(door["room0"].split("|"))
                roominfo_map[room0]["door"].append({"holePolygon":holePolygon,
                                                    "doorType": double_or_single})

                room1 = "_".join(door["room1"].split("|"))
                roominfo_map[room1]["door"].append({"holePolygon":holePolygon,
                                                    "doorType": double_or_single})

        housedir = os.path.join(ai2thor_dir,housename)
        file_lst = os.listdir(housedir)
        for room_name in file_lst:
            if room_name[:4]=="room" and room_name[-4:]==".obj":
                room_id = room_name.split(".")[0]
                roominfo = roominfo_map[room_id]
                filename = os.path.join(housedir,room_name)
                # Render and save the room mask as an image
                floor_plan =  Mesh.from_file(filename)
                center = (floor_plan.bbox[0]+floor_plan.bbox[1])/2
                center_info[housename+"/"+room_name] = {"center":center.tolist(),
                                                        "roomType":roominfo["roomType"],
                                                        "door":roominfo["door"]}         
                vertices, faces = floor_plan.to_points_and_faces()
                # Center the floor
                vertices -= center
                floor_plan = Mesh.from_faces(vertices, faces)

                outfile = os.path.join(cfg.ai2thor.path_to_mask,housename+"_"+room_name[:-4]+".png")
                
                room_mask = render(
                    scene_black,
                    [floor_plan],
                    (1.0, 1.0, 1.0),
                    "flat",
                    outfile
                )[:, :, 0:1]

    path_to_center_info = cfg.ai2thor.path_to_center_info
    with open(path_to_center_info,"w") as f:
        json.dump(center_info,f,indent=4)
if __name__ == "__main__":
    main()