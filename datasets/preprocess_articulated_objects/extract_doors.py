
"""
copy useful obj files from 'textured_objs' to a new folder 'textured_objs_clean'
"""

import os
ID_PATH = '/home/yandan/workspace/GAPartNet/dataset/render_tools/meta/partnet_all_id_list.txt'

def matexport(mat,fout,hastexture):
    if hastexture:
        for line in mat:
            if line.startswith("illum"):
                fout.write("illum 2\n")
            elif line.startswith("Ns"):
                continue
                # fout.write("Ns 1\n")
            # elif line.startswith("Kd"):
            #     fout.write("Kd 0.5 0.5 0.5\n")
            elif line.startswith("Ks"):
                fout.write("Ks 0.0 0.0 0.0\n")
            else:
                fout.write(line)
    else:
        for line in mat:
            fout.write(line)
                

model_id_list = []
with open(ID_PATH, 'r') as fd:
    for line in fd:
        ls = line.strip().split(' ')
        if ls[0]=='Door':
            model_id_list.append(int(ls[1]))


path_to_models = "/home/yandan/dataset/partnet_mobility_part/"
door_path = "/home/yandan/dataset/GPN_Door"
objects = []
# model_id_list = ["47926"]
for model_id in model_id_list:
    in_path = os.path.join(path_to_models,str(model_id))
    out_path = os.path.join(door_path,str(model_id))
    
    os.system("cp -r "+in_path +" "+out_path )
        

        # os.system("cp "+fin +" "+fout )

    
