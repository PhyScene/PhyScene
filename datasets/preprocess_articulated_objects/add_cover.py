
import os
import json
from xml.etree.ElementTree import ElementTree, Element
import xml.dom.minidom
import trimesh
from PIL import Image
import numpy as np
from xml.dom.minidom import parse, parseString
def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def write_xml(tree,out_path):
    tree.write(out_path,xml_declaration=True)

def find_nodes(tree, path):
    return tree.findall(path)

def if_match(kv_map, key, value ):
    if value in kv_map.get(key):
        return True
    return False

def remove_node_by_keyvalue(nodelist,kv_map):
    result_nodes = []
    for node in nodelist:
        if not if_match(kv_map, "name", value):
            result_nodes.append(node)
    return result_nodes

def create_node(tag,property_map, content):
    element = Element(tag,property_map)
    element.text = content
    return element

def add_child_node(nodelist,element):
    for node in nodelist:
        node.append(element)

def load_mesh(objname):
    tr_mesh = trimesh.load(objname, force="mesh")  #47926
    return tr_mesh

def load_meshs(obj_lst):
    trimesh_meshes = [] 
    for objname in obj_lst:
        tr_mesh = load_mesh(objname)
        trimesh_meshes.append(tr_mesh)
    allmesh = trimesh.util.concatenate(trimesh_meshes)
    # allmesh.show()
    return allmesh

def get_cover(mesh):
    #bbox
    bounding_box = mesh.bounds
    min_coordinates = bounding_box[0]
    max_coordinates = bounding_box[1]
    #size
    dx = max_coordinates[0]-min_coordinates[0]
    thickness = 0.02
    dz = max_coordinates[2]-min_coordinates[2]
    
    box = trimesh.creation.box(extents=(dx, thickness, dz))
    # position
    cx = (max_coordinates[0]+min_coordinates[0])/2
    cy = max_coordinates[1]+thickness/2
    cz = (max_coordinates[2]+min_coordinates[2])/2
    center = [cx, cy, cz]
    box.apply_translation(center - box.centroid)
    # Print the resulting box center and extents
    # print("Box center:", box.centroid)
    # print("Box extents:", box.extents)
    return box

def add_material(tr_mesh,texture_path):

    
    vertices = tr_mesh.vertices
    faces = tr_mesh.faces
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm

    tr_mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture_path)
        )
    )

    return tr_mesh

def find_base_link_name(dom):
    for linknode in dom.getElementsByTagName('robot')[0].childNodes:
        if linknode.nodeName == "joint":
            baselinkname = ""
            for childnode in linknode.childNodes:
                if childnode.nodeName == "child":
                    baselinkname = childnode.getAttribute("link")

                if childnode.nodeName == "parent" and childnode.getAttribute("link") == "base":
                    return baselinkname
    return baselinkname
                    
def add_new_node(dom,baselinkname):
    emptynode2 = dom.createTextNode("\n\t\t")
    for linknode in dom.getElementsByTagName('robot')[0].childNodes:
        if linknode.nodeName == "link" and linknode.getAttribute("name") == baselinkname:
            # print(linknode.childNodes)
            new_visual_node = visual_node()
            new_collision_node = collision_node()
            linknode.childNodes = linknode.childNodes[:-1] + [emptynode2,new_visual_node,emptynode2,new_collision_node] + linknode.childNodes[-1:]
            

def visual_node():
    structured_string = "<visual name=\"cover_top\">\n" + \
		"\t\t\t<origin xyz=\"0 0 0\"/>\n" + \
		"\t\t\t<geometry>\n"  + \
		"\t\t\t\t<mesh filename=\"textured_objs/cover_top.obj\"/> \n" + \
		"\t\t\t</geometry>\n" + \
		"\t\t</visual>\n"
    doc = xml.dom.minidom.parseString(structured_string)
    return doc.childNodes[0]

def collision_node():
    structured_string = "<collision>\n" + \
		"\t\t\t<origin xyz=\"0 0 0\"/>\n" + \
		"\t\t\t<geometry>\n"  + \
		"\t\t\t\t<mesh filename=\"textured_objs/cover_top.obj\"/> \n" + \
		"\t\t\t</geometry>\n" + \
		"\t\t</collision>\n"
    doc = xml.dom.minidom.parseString(structured_string)
    return doc.childNodes[0]

visual_node()
texture_path = "texture_0.jpg"
# get no cover list
no_cover_dir = "/home/yandan/workspace/GAPartNet/dataset/render_tools/useful_rendered/rgb/nocover"
model_id_list = set()
for file in os.listdir(no_cover_dir):
    try:
        id = file.split("_")[1]
        model_id_list.add(id)
    except:
        continue
model_id_list = list(model_id_list)
model_id_list.sort()

path_to_models = "/home/yandan/dataset/partnet_mobility_part/"
objects = []
# model_id_list = ["45780"]
for model_id in model_id_list:
    print(model_id)
    folderout = os.path.join(path_to_models,str(model_id))
    urdf_path = os.path.join(folderout,"mobility_annotation_gapartnet.urdf" )

    obj_lst = []
    dom = xml.dom.minidom.parse(urdf_path)
    emptynode = dom.getElementsByTagName('robot')[0].childNodes[0]
    for linknode in dom.getElementsByTagName('robot')[0].childNodes:
        if linknode.nodeName == "link":
            for visualnode in linknode.childNodes:
                if visualnode.nodeName == "visual":
                    name = visualnode.getAttribute("name")
                    if "handle" in name or "drawer" in name or "door" in name:
                        continue
                    for geometrynode in visualnode.childNodes:
                        if geometrynode.nodeName == "geometry":
                            for meshnode in geometrynode.childNodes:
                                if meshnode.nodeName == "mesh":
                                    objname = meshnode.getAttribute("filename")
                                    objname = os.path.join(path_to_models,str(model_id),objname)
                                    obj_lst.append(objname)
    # #create cover mesh and export
    # mesh = load_meshs(obj_lst)
    # cover_mesh = get_cover(mesh)
    # cover_mesh = add_material(cover_mesh,texture_path)
    # # outpath = os.path.join(folderout,"textured_objs_clean","cover_top.obj")
    # outpath = os.path.join(folderout,"textured_objs","cover_top.obj")
    # cover_mesh.export(outpath)

    #update urdf
    baselinkname = find_base_link_name(dom)
    add_new_node(dom,baselinkname)
    outpath = os.path.join(folderout,"covered.urdf")
    with open(outpath,'w') as f:
        dom.writexml(f)

    os.system("mv "+ urdf_path+" "+ urdf_path+".nocover")
    os.system("mv "+ outpath+" "+urdf_path)
    
