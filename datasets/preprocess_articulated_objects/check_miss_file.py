
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
        model_id_list.append(int(ls[1]))


path_to_models = "/home/yandan/dataset/partnet_mobility_part/"
objects = []
# get no cover list
no_cover_dir = "/home/yandan/workspace/GAPartNet/dataset/render_tools/useful_rendered/rgb/cover"
model_id_list = set()
for file in os.listdir(no_cover_dir):
    try:
        id = file.split("_")[1]
        model_id_list.add(id)
    except:
        continue
model_id_list = list(model_id_list)
model_id_list.sort()
for model_id in model_id_list:
    urdf_path = os.path.join(path_to_models,str(model_id),"mobility_annotation_gapartnet.urdf" )
    folderin = os.path.join(path_to_models,str(model_id),"textured_objs")
    # folderout = os.path.join(path_to_models,str(model_id),"textured_objs_clean")
    # if not os.path.exists(folderout):
    #     os.mkdir(folderout)
    #find useful obj files
    obj_lst = []
    f = open(urdf_path)
    for line in f.readlines():
        if "filename=" in line:
            obj_lst.append(line.split("filename=")[1].split("/")[1].split(".obj")[0])
    obj_lst = list(set(obj_lst))
    #copy useful objs
    for obj in obj_lst:
        fin = os.path.join(folderin,obj+".obj")
        if not os.path.exists(fin):
            a = 1
        # fout = os.path.join(folderout,obj+".obj")
        

    
