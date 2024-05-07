import os
import json
basepath = "/home/yandan/workspace/GAPartNet/dataset/render_tools/useful_rendered/rgb"
id_set = set()
for file in os.listdir(basepath):
    try:
        id = file.split("_")[1]
        id_set.add(id)
    except:
        continue

id_lst = {"id_lst":list(id_set)}
with open("/home/yandan/workspace/PhyScene/data/GPN_good_idx.json","w") as f:
    json.dump(id_lst,f, indent = 4)
