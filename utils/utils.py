# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np
from PIL import Image
import trimesh
import math

from datasets.gapartnet_dataset import MapThreedfuture2gparnet

from simple_3dviz import Mesh
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.renderables.base import RenderableCollection

def get_textured_objects(bbox_params_t, objects_dataset, gapartnet_dataset, classes, cfg = None):
    # For each one of the boxes replace them with an object
    renderables = []
    renderables_remesh = []
    
    trimesh_meshes = []
    use_feature = cfg.task.dataset.use_feature
    class_num = cfg.task.dataset.class_num
    furniture_names = []
    matrial_save = None
    texture_color_lst = []
    scene_info = {"ThreedFront":dict(),"GPN":dict()}
    for data in ["3dfront","gpn"]:
        for j in range(bbox_params_t.shape[1]):

            query_label = classes[bbox_params_t[0, j, :class_num].argmax(-1)]  #TODO
            translation = bbox_params_t[0, j, class_num:class_num+3]
            query_size = bbox_params_t[0, j, class_num+3:class_num+6]
            
            theta = bbox_params_t[0, j, class_num+6]
            query_objfeat = bbox_params_t[0, j, class_num+7:]
            
            # MapThreedfuture2gparnet = dict()
            
            if query_label == 'empty':
                continue
            if data == "3dfront" and (gapartnet_dataset==None or query_label not in MapThreedfuture2gparnet):                    
                if use_feature:
                    furniture = objects_dataset.get_closest_furniture_to_objfeats_and_size(
                        query_label, query_objfeat, query_size
                    )
                else:
                    furniture = objects_dataset.get_closest_furniture_to_box(
                        query_label, query_size
                    )
                rot = (theta/math.pi*180) % 360
                print("3D FUTURE ",furniture.label,furniture.model_jid, rot)
                furniture_names.append(furniture.label)
                
                # Load the furniture and scale it as it is given in the dataset
                #raw mesh
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = raw_mesh.bbox
                centroid = (bbox[0] + bbox[1])/2

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables.append(raw_mesh)


                #load remesh model for collision rate
                raw_mesh = TexturedMesh.from_file(furniture.remesh_model_path)
                raw_mesh.scale(furniture.scale)

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables_remesh.append(raw_mesh)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
                tr_mesh.vertices *= furniture.scale
                tr_mesh.vertices -= centroid
                tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                trimesh_meshes.append(tr_mesh)
                
                if furniture.model_jid not in scene_info["ThreedFront"]:
                    scene_info["ThreedFront"][furniture.model_jid]  =[]
                scene_info["ThreedFront"][furniture.model_jid].append({"path":furniture.raw_model_path, \
                                                                "scale": furniture.scale, \
                                                                "size": furniture.size.tolist(), \
                                                                "label": furniture.label, \
                                                                "query_label": query_label, \
                                                                "query_size": [float(s/2) for s in query_size.tolist()],\
                                                                "theta": float(theta), \
                                                                "position": (translation-centroid).tolist(),\
                                                                "centroid": centroid.tolist()})

            elif data=="gpn" and not (gapartnet_dataset==None or query_label not in MapThreedfuture2gparnet):
               
                if use_feature:
                    furniture = gapartnet_dataset.get_closest_furniture_to_objfeats_and_size(
                        MapThreedfuture2gparnet[query_label], query_objfeat, query_size
                    )
                else:
                    furniture = gapartnet_dataset.get_closest_furniture_to_box_normsize(
                        MapThreedfuture2gparnet[query_label], query_size
                    )
                if furniture.model_jid == "22367":
                    a = 1
                scale = furniture.scale * query_size/furniture.size * 2
                print("GPN ",furniture.label,furniture.model_jid)
                furniture_names.append(furniture.label)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = furniture.bbox
                # centroid = (bbox[0] + bbox[1])/2
                centroid = (bbox[0] + bbox[1])/2*scale

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.

                ####raw model
                GPN_renderables = []
                for raw_model_path in furniture.raw_model_path:
                    # Load the furniture and scale it as it is given in the dataset
                    raw_mesh = TexturedMesh.from_file(raw_model_path)         
                    raw_mesh.scale(scale)
                
                    # Apply the transformations in order to correctly position the mesh
                    raw_mesh.affine_transform(t=-centroid)
                    raw_mesh.affine_transform(R=R, t=translation)

                    try:
                        GPN_renderables += raw_mesh.renderables
                    except:
                        GPN_renderables.append(raw_mesh)

                if len(GPN_renderables) == 1:
                    renderables.append(GPN_renderables[0])
                else:
                    renderables.append(RenderableCollection(GPN_renderables))

                ####remesh model
                #load remesh model for collision rate
                GPN_renderables = []
                for raw_model_path in furniture.sequence_model_path:
                    # Load the furniture and scale it as it is given in the dataset
                    raw_mesh = TexturedMesh.from_file(raw_model_path)                
                    raw_mesh.scale(scale)

                    # Apply the transformations in order to correctly position the mesh
                    raw_mesh.affine_transform(t=-centroid)
                    raw_mesh.affine_transform(R=R, t=translation)

                    try:
                        GPN_renderables += raw_mesh.renderables
                    except:
                        GPN_renderables.append(raw_mesh)


                if len(GPN_renderables) == 1:
                    renderables_remesh.append(GPN_renderables[0])
                else:
                    renderables_remesh.append(RenderableCollection(GPN_renderables))
                if furniture.model_jid not in scene_info["GPN"]:
                    scene_info["GPN"][furniture.model_jid]  =[]
                #final size = query size = scale*size
                scene_info["GPN"][furniture.model_jid].append({"path":furniture.raw_model_path, \
                                                        "scale": [float(s) for s in scale.tolist()], \
                                                        "size": [float(s/2) for s in furniture.size], \
                                                        "label": furniture.label, \
                                                        "query_label": query_label,\
                                                        "query_size": [float(s) for s in query_size.tolist()],\
                                                        "theta": float(theta), \
                                                        "position": (translation-centroid).tolist(), \
                                                        "centroid": centroid.tolist()})

    return renderables, trimesh_meshes, furniture_names, renderables_remesh, scene_info

def get_textured_objects_rescale(bbox_params_t, objects_dataset, gapartnet_dataset, classes, cfg = None):
    # For each one of the boxes replace them with an object
    renderables = []
    renderables_remesh = []
    
    trimesh_meshes = []
    use_feature = False #cfg.task.dataset.use_feature
    class_num = cfg.task.dataset.class_num
    furniture_names = []
    matrial_save = None
    texture_color_lst = []
    scene_info = {"ThreedFront":dict(),"GPN":dict()}
    for data in ["3dfront","gpn"]:
        for j in range(bbox_params_t.shape[1]):
            
            query_label = classes[bbox_params_t[0, j, :class_num].argmax(-1)]  #TODO
            translation = bbox_params_t[0, j, class_num:class_num+3]
            query_size = bbox_params_t[0, j, class_num+3:class_num+6]
            
            theta = bbox_params_t[0, j, class_num+6]
            query_objfeat = bbox_params_t[0, j, class_num+7:]
            
            # MapThreedfuture2gparnet = dict()
            
            if query_label == 'empty':
                continue
            if data == "3dfront" and (gapartnet_dataset==None or query_label not in MapThreedfuture2gparnet):                    
                if use_feature:
                    furniture = objects_dataset.get_closest_furniture_to_objfeats_and_size(
                        query_label, query_objfeat, query_size
                    )
                else:
                    furniture = objects_dataset.get_closest_furniture_to_box(
                        query_label, query_size
                    )
                # scale1 = furniture.scale 
                scale = query_size/furniture.size_recal() * furniture.scale
                rot = (theta/math.pi*180) % 360
                print("3D FUTURE ",furniture.label,furniture.model_jid, rot)
                furniture_names.append(furniture.label)
                
                # Load the furniture and scale it as it is given in the dataset
                #raw mesh
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(scale)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = raw_mesh.bbox
                # centroid = (bbox[0] + bbox[1])/2
                centroid = (bbox[0] + bbox[1])/2*scale
                centroid4TV = [centroid[0]*np.cos(theta)-centroid[2]*np.sin(theta),
                               centroid[1],
                               centroid[0]*np.sin(theta)+centroid[2]*np.cos(theta)]

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables.append(raw_mesh)


                #load remesh model for collision rate
                raw_mesh = TexturedMesh.from_file(furniture.remesh_model_path)
                raw_mesh.scale(scale)

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables_remesh.append(raw_mesh)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
                tr_mesh.vertices *= scale
                tr_mesh.vertices -= centroid
                tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                trimesh_meshes.append(tr_mesh)
                
                if furniture.model_jid not in scene_info["ThreedFront"]:
                    scene_info["ThreedFront"][furniture.model_jid]  =[]
                scene_info["ThreedFront"][furniture.model_jid].append({"path":furniture.raw_model_path, \
                                                                    #    "scale":scale,\
                                                                "scale": [float(s) for s in scale.tolist()], \
                                                                "size": furniture.size.tolist(), \
                                                                "label": furniture.label, \
                                                                "query_label": query_label, \
                                                                "query_size": [float(s/2) for s in query_size.tolist()],\
                                                                "theta": float(theta), \
                                                                "position": (translation-np.array(centroid4TV)).tolist(),\
                                                                "centroid": centroid4TV})

            elif data=="gpn" and not (gapartnet_dataset==None or query_label not in MapThreedfuture2gparnet):
               
                furniture = gapartnet_dataset.get_closest_furniture_to_box_normsize(
                        query_label, query_size
                    )
                if furniture.model_jid == "22367":
                    a = 1
                scale = furniture.scale * query_size/furniture.size * 2
                print("GPN ",furniture.label,furniture.model_jid)
                furniture_names.append(furniture.label)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = furniture.bbox
                # centroid = (bbox[0] + bbox[1])/2
                centroid = (bbox[0] + bbox[1])/2*scale
                centroid4TV = [centroid[0]*np.cos(theta)-centroid[2]*np.sin(theta),
                               centroid[1],
                               centroid[0]*np.sin(theta)+centroid[2]*np.cos(theta)]

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.

                ####raw model
                GPN_renderables = []
                for raw_model_path in furniture.raw_model_path:
                    # Load the furniture and scale it as it is given in the dataset
                    raw_mesh = TexturedMesh.from_file(raw_model_path)         
                    raw_mesh.scale(scale)
                
                    # Apply the transformations in order to correctly position the mesh
                    raw_mesh.affine_transform(t=-centroid)
                    # raw_mesh.affine_transform(R=R)
                    raw_mesh.affine_transform(R=R, t=translation)

                    try:
                        GPN_renderables += raw_mesh.renderables
                    except:
                        GPN_renderables.append(raw_mesh)

                if len(GPN_renderables) == 1:
                    renderables.append(GPN_renderables[0])
                else:
                    renderables.append(RenderableCollection(GPN_renderables))

                ####remesh model
                #load remesh model for collision rate
                GPN_renderables = []
                for raw_model_path in furniture.sequence_model_path:
                    # Load the furniture and scale it as it is given in the dataset
                    raw_mesh = TexturedMesh.from_file(raw_model_path)                
                    raw_mesh.scale(scale)

                    # Apply the transformations in order to correctly position the mesh
                    raw_mesh.affine_transform(t=-centroid)
                    raw_mesh.affine_transform(R=R, t=translation)

                    try:
                        GPN_renderables += raw_mesh.renderables
                    except:
                        GPN_renderables.append(raw_mesh)


                if len(GPN_renderables) == 1:
                    renderables_remesh.append(GPN_renderables[0])
                else:
                    renderables_remesh.append(RenderableCollection(GPN_renderables))
                if furniture.model_jid not in scene_info["GPN"]:
                    scene_info["GPN"][furniture.model_jid]  =[]
                #final size = query size = scale*size
                scene_info["GPN"][furniture.model_jid].append({"path":furniture.raw_model_path, \
                                                        "scale": [float(s) for s in scale.tolist()], \
                                                        "size": [float(s/2) for s in furniture.size], \
                                                        "label": furniture.label, \
                                                        "query_label": query_label,\
                                                        "query_size": [float(s) for s in query_size.tolist()],\
                                                        "theta": float(theta), \
                                                        "position": (translation-np.array(centroid4TV)).tolist(),\
                                                        "centroid": centroid4TV})

                # for raw_model_path in furniture.raw_model_path:
                #     # Load the furniture and scale it as it is given in the dataset
                #     # raw_mesh = TexturedMesh.from_file(raw_model_path)  
                    
                #     # Create a trimesh object for the same mesh in order to save
                #     # everything as a single scene
                #     tr_mesh = trimesh.load(raw_model_path, force="mesh")
                #     # tr_mesh.visual.material.image = Image.open(
                #     #     furniture.texture_image_path
                #     # )
                #     tr_mesh.vertices *= scale
                #     tr_mesh.vertices -= centroid
                #     tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                #     tr_mesh.show()
                #     trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes, furniture_names, renderables_remesh, scene_info

def get_bbox_objects(bbox_params_t, classes, cfg=None):
    # For each one of the boxes replace them with an object
    class_num = cfg.task.dataset.class_num
   
    scene_info = {"objects":[]}
    
    for j in range(bbox_params_t.shape[1]):
        query_label = classes[bbox_params_t[0, j, :class_num].argmax(-1)]  #TODO
        translation = bbox_params_t[0, j, class_num:class_num+3]
        query_size = bbox_params_t[0, j, class_num+3:class_num+6]
        theta = bbox_params_t[0, j, class_num+6]
        rot = (theta/math.pi*180) % 360

        if query_label == 'empty':
            continue
        
        scene_info["objects"].append({"label": query_label, \
                                        "position": (translation).tolist(), \
                                        "scale": query_size.tolist(), \
                                        "theta": float(theta),\
                                        "rot_degree": float(rot)})

    return scene_info


def get_textured_objects_from_bbox(bboxes, objects_dataset, gapartnet_dataset, classes, cfg = None):
    # For each one of the boxes replace them with an object
    renderables = []
    renderables_remesh = []
    
    trimesh_meshes = []
    use_feature = False
    furniture_names = []
    scene_info = {"ThreedFront":dict(),"GPN":dict()}
    for data in ["3dfront","gpn"]:
        for bbox in bboxes:
            query_label = bbox["label"]
            translation = np.array(bbox["position"])
            query_size = np.array(bbox["scale"])
            theta = bbox["theta"]
            # query_objfeat = bbox_params_t[0, j, class_num+7:]
            

            if data == "3dfront" and (gapartnet_dataset==None or query_label not in MapThreedfuture2gparnet):                    
                if use_feature:
                    furniture = objects_dataset.get_closest_furniture_to_objfeats_and_size(
                        query_label, query_objfeat, query_size
                    )
                else:
                    furniture = objects_dataset.get_closest_furniture_to_box(
                        query_label, query_size
                    )
                rot = (theta/math.pi*180) % 360
                print("3D FUTURE ",furniture.label,furniture.model_jid, rot)
                furniture_names.append(furniture.label)
                
                # Load the furniture and scale it as it is given in the dataset
                #raw mesh
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = raw_mesh.bbox
                centroid = (bbox[0] + bbox[1])/2

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables.append(raw_mesh)


                #load remesh model for collision rate
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables_remesh.append(raw_mesh)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
                tr_mesh.vertices *= furniture.scale
                tr_mesh.vertices -= centroid
                tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                trimesh_meshes.append(tr_mesh)
                
                if furniture.model_jid not in scene_info["ThreedFront"]:
                    scene_info["ThreedFront"][furniture.model_jid]  =[]
                scene_info["ThreedFront"][furniture.model_jid].append({"path":furniture.raw_model_path, \
                                                                "scale": furniture.scale, \
                                                                "label": furniture.label, \
                                                                "theta": float(theta), \
                                                                "position": (translation-centroid).tolist()})

            elif data=="gpn" and not (gapartnet_dataset==None or query_label not in MapThreedfuture2gparnet):
               
                if use_feature:
                    furniture = gapartnet_dataset.get_closest_furniture_to_objfeats_and_size(
                        MapThreedfuture2gparnet[query_label], query_objfeat, query_size
                    )
                else:
                    furniture = gapartnet_dataset.get_closest_furniture_to_box_normsize(
                        MapThreedfuture2gparnet[query_label], query_size
                    )
                scale = furniture.scale * query_size/furniture.size * 2
                print("GPN ",furniture.label,furniture.model_jid)
                furniture_names.append(furniture.label)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = furniture.bbox
                centroid = (bbox[0] + bbox[1])/2

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.

                ####raw model
                GPN_renderables = []
                for raw_model_path in furniture.raw_model_path:
                    # Load the furniture and scale it as it is given in the dataset
                    raw_mesh = TexturedMesh.from_file(raw_model_path)         
                    raw_mesh.scale(scale)
                
                    # Apply the transformations in order to correctly position the mesh
                    raw_mesh.affine_transform(t=-centroid)
                    raw_mesh.affine_transform(R=R, t=translation)

                    try:
                        GPN_renderables += raw_mesh.renderables
                    except:
                        GPN_renderables.append(raw_mesh)

                if len(GPN_renderables) == 1:
                    renderables.append(GPN_renderables[0])
                else:
                    renderables.append(RenderableCollection(GPN_renderables))

                ####remesh model
                #load remesh model for collision rate
                GPN_renderables = []
                for raw_model_path in furniture.sequence_model_path:
                    # Load the furniture and scale it as it is given in the dataset
                    raw_mesh = TexturedMesh.from_file(raw_model_path)                
                    raw_mesh.scale(scale)

                    # Apply the transformations in order to correctly position the mesh
                    raw_mesh.affine_transform(t=-centroid)
                    raw_mesh.affine_transform(R=R, t=translation)

                    try:
                        GPN_renderables += raw_mesh.renderables
                    except:
                        GPN_renderables.append(raw_mesh)


                if len(GPN_renderables) == 1:
                    renderables_remesh.append(GPN_renderables[0])
                else:
                    renderables_remesh.append(RenderableCollection(GPN_renderables))
                if furniture.model_jid not in scene_info["GPN"]:
                    scene_info["GPN"][furniture.model_jid]  =[]
                scene_info["GPN"][furniture.model_jid].append({"path":furniture.raw_model_path, \
                                                        "scale": scale.tolist(), \
                                                        "label": furniture.label, \
                                                        "theta": float(theta), \
                                                        "position": (translation-centroid).tolist()})

                # for raw_model_path in furniture.raw_model_path:
                #     # Load the furniture and scale it as it is given in the dataset
                #     # raw_mesh = TexturedMesh.from_file(raw_model_path)  
                    
                #     # Create a trimesh object for the same mesh in order to save
                #     # everything as a single scene
                #     tr_mesh = trimesh.load(raw_model_path, force="mesh")
                #     # tr_mesh.visual.material.image = Image.open(
                #     #     furniture.texture_image_path
                #     # )
                #     tr_mesh.vertices *= scale
                #     tr_mesh.vertices -= centroid
                #     tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                #     tr_mesh.show()
                #     trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes, furniture_names, renderables_remesh, scene_info


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )

    return floor, tr_floor
