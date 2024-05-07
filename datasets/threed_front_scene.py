
from collections import Counter
from dataclasses import dataclass
from functools import cached_property, reduce, lru_cache
import json
import os

import numpy as np
from PIL import Image

import trimesh

from .common import BaseScene

from simple_3dviz import Lines, Mesh, Spherecloud
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.misc import LightToCamera
import math
from utils.open3d_vis_utils import draw_box_label
from simple_3dviz.window import show


def rotation_matrix(axis, theta):
    """Axis-angle rotation matrix from 3D-Front-Toolbox."""
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


@dataclass
class Asset:
    """Contains the information for each 3D-FUTURE model."""
    super_category: str
    category: str
    style: str
    theme: str
    material: str

    @property
    def label(self):
        return self.category


class ModelInfo(object):
    """Contains all the information for all 3D-FUTURE models.

        Arguments
        ---------
        model_info_data: list of dictionaries containing the information
                         regarding the 3D-FUTURE models.
    """
    def __init__(self, model_info_data):
        self.model_info_data = model_info_data
        self._model_info = None
        # List to keep track of the different styles, themes
        self._styles = []
        self._themes = []
        self._categories = []
        self._super_categories = []
        self._materials = []

    @property
    def model_info(self):
        if self._model_info is None:
            self._model_info = {}
            # Create a dictionary of all models/assets in the dataset
            for m in self.model_info_data:
                # Keep track of the different styles
                if m["style"] not in self._styles and m["style"] is not None:
                    self._styles.append(m["style"])
                # Keep track of the different themes
                if m["theme"] not in self._themes and m["theme"] is not None:
                    self._themes.append(m["theme"])
                # Keep track of the different super-categories
                if m["super-category"] not in self._super_categories and m["super-category"] is not None:
                    self._super_categories.append(m["super-category"])
                # Keep track of the different categories
                if m["category"] not in self._categories and m["category"] is not None:
                    self._categories.append(m["category"])
                # Keep track of the different categories
                if m["material"] not in self._materials and m["material"] is not None:
                    self._materials.append(m["material"])

                super_cat = "unknown_super-category"
                cat = "unknown_category"

                if m["super-category"] is not None:
                    super_cat = m["super-category"].lower().replace(" / ", "/")

                if m["category"] is not None:
                    cat = m["category"].lower().replace(" / ", "/")

                self._model_info[m["model_id"]] = Asset(
                    super_cat,
                    cat, 
                    m["style"],
                    m["theme"],
                    m["material"]
                )

        return self._model_info

    @property
    def styles(self):
        return self._styles

    @property
    def themes(self):
        return self._themes

    @property
    def materials(self):
        return self._materials

    @property
    def categories(self):
        return set([s.lower().replace(" / ", "/") for s in self._categories])

    @property
    def super_categories(self):
        return set([
            s.lower().replace(" / ", "/")
            for s in self._super_categories
        ])

    @classmethod
    def from_file(cls, path_to_model_info):
        with open(path_to_model_info, "rb") as f:
            model_info = json.load(f)

        return cls(model_info)


class BaseThreedFutureModel(object):
    def __init__(self, model_uid, model_jid, position, rotation, scale):
        self.model_uid = model_uid
        self.model_jid = model_jid
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def _transform(self, vertices):
        # the following code is adapted and slightly simplified from the
        # 3D-Front toolbox (json2obj.py). It basically scales, rotates and
        # translates the model based on the model info.
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:]))*2
        vertices = vertices * self.scale
        if np.sum(axis) != 0 and not np.isnan(theta):
            R = rotation_matrix(axis, theta)
            vertices = vertices.dot(R.T)
        vertices += self.position

        return vertices

    def mesh_renderable(
        self,
        colors=(0.5, 0.5, 0.5, 1.0),
        offset=[[0, 0, 0]],
        with_texture=False
    ):
        if not with_texture:
            m = self.raw_model_transformed(offset)
            return Mesh.from_faces(m.vertices, m.faces, colors=colors)
        else:
            m = TexturedMesh.from_file(self.raw_model_path)
            m.scale(self.scale)
            # Extract the predicted affine transformation to position the
            # mesh
            theta = self.z_angle
            R = np.zeros((3, 3))
            R[0, 0] = np.cos(theta)
            R[0, 2] = -np.sin(theta)
            R[2, 0] = np.sin(theta)
            R[2, 2] = np.cos(theta)
            R[1, 1] = 1.

            # Apply the transformations in order to correctly position the mesh
            m.affine_transform(R=R, t=self.position)
            m.affine_transform(t=offset)
            return m
        
    def mesh_trimesh(
        self,
        colors=(0.5, 0.5, 0.5, 1.0),
        offset=[[0, 0, 0]],
        with_texture=False
    ):
        
        tr_mesh = trimesh.load(self.raw_model_path, force="mesh")
        tr_mesh.visual.material.image = Image.open(
            self.texture_image_path
        )
        tr_mesh.vertices *= self.scale
        theta = self.z_angle
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + self.position
        tr_mesh.vertices += offset

        return tr_mesh

        


class ThreedFutureModel(BaseThreedFutureModel):
    def __init__(
        self,
        model_uid,
        model_jid,
        model_info,
        position,
        rotation,
        scale,
        path_to_models
    ):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.model_info = model_info
        
        self._label = None
        self.expand_ratio = np.array([[-1,-1,-1,-1,-1,-1]])
        self.path_to_models = path_to_models

    @property
    def raw_model_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model.obj"
        )
    
    @property
    def remesh_model_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model_remesh.obj"
        )

    # add normalized point cloud of raw_model
    @property
    def raw_model_norm_pc_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model_norm_pc.npz"
        )

    @property
    def raw_model_norm_pc_lat_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model_norm_pc_lat64.npz" #yyd
        )

    @property
    def raw_model_norm_pc_lat32_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            # "raw_model_norm_pc_lat32_ratio.npz"
            "raw_model_norm_pc_lat32.npz"
        )
    
    @property
    def raw_model_norm_pc_lat_ulip_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "raw_model_norm_pc_lat_ulip.npz"
        )
    

    @property
    def texture_image_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "texture.png"
        )

    @property
    def path_to_bbox_vertices(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "bbox_vertices.npy"
        )

    # add normalized point cloud of raw_model
    def raw_model_norm_pc(self):
        points = np.load(self.raw_model_norm_pc_path)["points"].astype(np.float32)
        return points
    
    def raw_model_norm_pc_lat(self):
        latent = np.load(self.raw_model_norm_pc_lat_path)["latent"].astype(np.float32)
        return latent
    
    @cached_property
    def raw_model_norm_pc_lat32_cache(self):
        latent = np.load(self.raw_model_norm_pc_lat32_path)["latent"].astype(np.float32)
        return latent
    
    def raw_model_norm_pc_lat32(self):
        latent = self.raw_model_norm_pc_lat32_cache
        return latent
    
    @cached_property
    def raw_model_norm_pc_lat_ulip_cache(self):
        latent = np.load(self.raw_model_norm_pc_lat_ulip_path)["latent"].astype(np.float32)
        return latent
    
    def raw_model_norm_pc_lat_ulip(self):
        latent = self.raw_model_norm_pc_lat_ulip_cache
        return latent

    def raw_model(self):
        try:
            return trimesh.load(
                self.raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,
                skip_texture=True
            )
        except:
            print("Loading model failed", flush=True)
            print(self.raw_model_path, flush=True)
            raise

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        model = self.raw_model()
        faces = np.array(model.faces)
        vertices = self._transform(np.array(model.vertices)) + offset

        return trimesh.Trimesh(vertices, faces)

    def centroid(self, offset=[[0, 0, 0]]):
        return self.corners(offset).mean(axis=0)

    @cached_property
    def size(self):
        corners = self.corners()
        return np.array([
            np.sqrt(np.sum((corners[4]-corners[0])**2))/2,
            np.sqrt(np.sum((corners[2]-corners[0])**2))/2,
            np.sqrt(np.sum((corners[1]-corners[0])**2))/2
        ])

    def bottom_center(self, offset=[[0, 0, 0]]):
        centroid = self.centroid(offset)
        size = self.size
        return np.array([centroid[0], centroid[1]-size[1], centroid[2]])

    @cached_property
    def bottom_size(self):
        return self.size * [1, 2, 1]

    @cached_property
    def z_angle(self):
        # See BaseThreedFutureModel._transform for the origin of the following
        # code.
        ref = [0, 0, 1]
        axis = np.cross(ref, self.rotation[1:])
        theta = np.arccos(np.dot(ref, self.rotation[1:]))*2

        if np.sum(axis) == 0 or np.isnan(theta):
            return 0

        assert np.dot(axis, [1, 0, 1]) == 0
        assert 0 <= theta <= 2*np.pi

        if theta >= np.pi:
            theta = theta - 2*np.pi

        return np.sign(axis[1]) * theta

    @property
    def label(self):
        if self._label is None:
            self._label = self.model_info.label
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    @cached_property
    def corners_base(self):
        try:
            bbox_vertices = np.load(self.path_to_bbox_vertices, mmap_mode="r")
        except:
            bbox_vertices = np.array(self.raw_model().bounding_box.vertices)
            np.save(self.path_to_bbox_vertices, bbox_vertices)
        c = self._transform(bbox_vertices)
        return c

    def corners(self, offset=[[0, 0, 0]]):
        c = self.corners_base
        return c + offset

    def origin_renderable(self, offset=[[0, 0, 0]]):
        corners = self.corners(offset)
        return Lines(
            [
                corners[0], corners[4],
                corners[0], corners[2],
                corners[0], corners[1]
            ],
            colors=np.array([
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ]),
            width=0.02
        )

    def bbox_corners_renderable(
        self, sizes=0.1, colors=(1, 0, 0), offset=[[0, 0, 0]]
    ):
        return Spherecloud(self.corners(offset), sizes=sizes, colors=colors)

    def bbox_renderable(
        self, colors=(0.00392157, 0., 0.40392157, 1.), offset=[[0, 0, 0]]
    ):
        alpha = np.array(self.size)[None]
        epsilon = np.ones((1, 2)) * 0.1
        translation = np.array(self.centroid(offset))[None]
        R = np.zeros((1, 3, 3))
        theta = np.array(self.z_angle)
        R[:, 0, 0] = np.cos(theta)
        R[:, 0, 2] = -np.sin(theta)
        R[:, 2, 0] = np.sin(theta)
        R[:, 2, 2] = np.cos(theta)
        R[:, 1, 1] = 1.

        return Mesh.from_superquadrics(alpha, epsilon, translation, R, colors)

    def show(
        self,
        behaviours=[LightToCamera()],
        with_bbox_corners=False,
        offset=[[0, 0, 0]]
    ):
        renderables = self.mesh_renderable(offset=offset)
        if with_bbox_corners:
            renderables += [self.bbox_corners_renderable(offset=offset)]
        show(renderables, behaviours=behaviours)

    def one_hot_label(self, all_labels):
        return np.eye(len(all_labels))[self.int_label(all_labels)]

    def int_label(self, all_labels):
        return all_labels.index(self.label)

    def copy_from_other_model(self, other_model):
        model = ThreedFutureModel(
            model_uid=other_model.model_uid,
            model_jid=other_model.model_jid,
            model_info=other_model.model_info,
            position=self.position,
            rotation=self.rotation,
            scale=other_model.scale,
            path_to_models=self.path_to_models
        )
        model.label = self.label
        return model


class ThreedFutureExtra(BaseThreedFutureModel):
    def __init__(
        self,
        model_uid,
        model_jid,
        xyz,
        faces,
        model_type,
        position,
        rotation,
        scale
    ):
        super().__init__(model_uid, model_jid, position, rotation, scale)
        self.xyz = xyz
        self.faces = faces
        self.model_type = model_type

    def raw_model_transformed(self, offset=[[0, 0, 0]]):
        vertices = self._transform(np.array(self.xyz)) + offset
        faces = np.array(self.faces)
        return trimesh.Trimesh(vertices, faces)

    def show(
        self, behaviours=[LightToCamera(), SnapshotOnKey()], offset=[[0, 0, 0]]
    ):
        renderables = self.mesh_renderable(offset=offset)
        show(renderables, behaviours)


class Room(BaseScene):
    def __init__(
        self, scene_id, scene_type, bboxes, extras, json_path,
        path_to_room_masks_dir=None
    ):
        super().__init__(scene_id, scene_type, bboxes)
        self.json_path = json_path
        self.extras = extras

        self.uid = "_".join([self.json_path, scene_id])
        self.path_to_room_masks_dir = path_to_room_masks_dir
        if path_to_room_masks_dir is not None:
            self.path_to_room_mask = os.path.join(
                self.path_to_room_masks_dir, self.uid, "room_mask.png"
            )
        else:
            self.path_to_room_mask = None
        # a = self.floor_plan_polygon()

    @property
    def floor(self):
        return [ei for ei in self.extras if ei.model_type == "Floor"][0]

    @property
    @lru_cache(maxsize=512)
    def bbox(self):
        corners = np.empty((0, 3))
        for f in self.bboxes:
            corners = np.vstack([corners, f.corners()])
        return np.min(corners, axis=0), np.max(corners, axis=0)

    @cached_property
    def bboxes_centroid(self):
        a, b = self.bbox
        return (a+b)/2

    @property
    def furniture_in_room(self):
        return [f.label for f in self.bboxes]

    @property
    def floor_plan(self):
        def cat_mesh(m1, m2):
            v1, f1 = m1
            v2, f2 = m2
            v = np.vstack([v1, v2])
            f = np.vstack([f1, f2 + len(v1)])
            return v, f

        # Compute the full floor plan
        vertices, faces = reduce(
            cat_mesh,
            ((ei.xyz, ei.faces) for ei in self.extras if ei.model_type == "Floor")
        )
        return np.copy(vertices), np.copy(faces)

    @cached_property
    def floor_plan_bbox(self):
        vertices, faces = self.floor_plan
        return np.min(vertices, axis=0), np.max(vertices, axis=0)
    
    @staticmethod
    def calc_polygon_by_mesh(vertices, faces):
        if len(faces)>100: 
            return False, None
        #remove repeated vertex
        vertices_map = dict()
        for i in range(len(vertices)):
            vertices_map[i] = i
            for key in vertices_map.keys():
                if key==i:
                    break
                diff = np.absolute(vertices[vertices_map[key]]-vertices[i])
                if not (diff>0.0001).any():
                    vertices_map[i] = key
                    break
        for i in range(len(faces)):
            faces[i][0] = vertices_map[faces[i][0]]
            faces[i][1] = vertices_map[faces[i][1]]
            faces[i][2] = vertices_map[faces[i][2]]
        #reduce polygon
        faces_pool = faces
        polygon = list(faces_pool[0])
        faces_pool = np.delete(faces_pool,0,0)
        while(len(faces_pool)!=0):
            l1 = len(faces_pool)
            for idx in reversed(range(len(faces_pool))):
                flag = False  #mark if face is merged into polygon
                face = faces_pool[idx]
                l = len(polygon)
                for i in range(l):
                    for j in range(len(face)):
                        if polygon[i]==face[j]:
                            if polygon[(i+1)%l] == face[(j+1)%3]:
                                polygon.insert(i+1,face[(j+2)%3])
                                faces_pool = np.delete(faces_pool,idx,0)
                                flag = True
                                break
                            if polygon[(i+1)%l] == face[(j-1)%3]:
                                polygon.insert(i+1,face[(j+1)%3])
                                faces_pool = np.delete(faces_pool,idx,0)
                                flag = True
                                break
                    if flag:
                        break
            l2 = len(faces_pool)
            if l1==l2:
                #bug
                return False, None

        #delete invalid point
        for idx in reversed(range(len(polygon))):
            i = idx 
            j = idx-1
            k = idx-2
            if k<0:
                break
            pi = np.array([vertices[polygon[i]][0],vertices[polygon[i]][2]])
            pj = np.array([vertices[polygon[j]][0],vertices[polygon[j]][2]])
            pk = np.array([vertices[polygon[k]][0],vertices[polygon[k]][2]])
            vij = pj-pi
            vjk = pk-pj
            if np.cross(vij,vjk)==0: #chacheng,parallel = 0
                del polygon[j]

        i = 0
        while(i<len(polygon)):
            idx_left = polygon[i]
            j = len(polygon)-1
            while(j>i):
                if polygon[j] == idx_left:
                    polygon = polygon[:i+1]+polygon[j+1:]
                    break
                j -= 1
            i += 1

        points = [(vertices[p][0],vertices[p][2]) for p in polygon]


        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches

        # # Define the point sequence
        
        # fig, ax = plt.subplots()
        # p = patches.Polygon(points, closed=True)
        # # Add the polygon patch to the axis
        # ax.add_patch(p)
        # room_side=6.2

        # ax.set_xlim(-room_side, room_side)
        # ax.set_ylim(-room_side, room_side)

        # # Mask the area inside the polygon
        # ax.fill(*zip(*points), 'black')

        # # Show the plot
        # plt.show()
        return True, np.array(points)
    
    def floor_plan_polygon(self):
        vertices, faces = self.floor_plan
        flag, points = self.calc_polygon_by_mesh(vertices, faces)
        return flag, points
    
    @staticmethod
    def is_perpendicular(line1,line2):
        #k
        # slope1 = line1[1]/ line1[0]
        # slope2 = line2[1] / line2[0]

        if line1[1]*line2[1]==-line1[0]*line2[0]:
            return True
        else:
            return False
    
    @staticmethod
    def segments_intersect(A,B,C,D):
        #check bbox 
        if max(A[0],B[0])<min(C[0],D[0]) or max(C[0],D[0])<min(A[0],B[0]) or \
            max(A[1],B[1])<min(C[1],D[1]) or max(C[1],D[1])<min(A[1],B[1]):
            return False
        
        #calc cross product
        def cross_product(p1,p2,p3):
            out = (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0])
            return out
        
        #chech whether two segs are crossed
        if cross_product(A,B,C) * cross_product(A,B,D) <= 0 and \
            cross_product(C,D,A) * cross_product(C,D,B) <= 0:
            a = cross_product(C,D,A)
            b = cross_product(C,D,B)
            t = a/(a-b+0.0000001)
            x = A[0] + t*(B[0]-A[0])
            y = A[1] + t*(B[1]-A[1])
            # if (Room.dist(A,(x,y))<0.0001) or (Room.dist(B,(x,y))<0.0001):
            #     return False
            return (x,y)
        else:
            return False
    @staticmethod 
    def dist2(A,B):
        return (A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])
    
    @staticmethod
    def calc_box_from_polygon( points, S):
        l = len(points)
        boxes = {"translation":[],"size":[]}
        for i in range(l):
            # i = 6
            # if i==6:
            #     continue
            x = points[i]
            j = (i+1)%l
            k = (i+2)%l
            vij = np.array(points[j]-points[i])
            vjk = np.array(points[k]-points[j])
            # x->y, x->z
            vxz = np.array([-vij[1],vij[0]])
            # z = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S
            

            #####calc y
            C = x
            D = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S  #y
            line2 = D-C
            intersection_lst = []
            for t in range(1,l-1):
                A = points[(i+t)%l]
                B = points[(i+t+1)%l]
                line1 = B-A
                if Room.is_perpendicular(line1,line2) and np.cross(line2,line1)>0: #chuizhi & left 
                    intersection = Room.segments_intersect(A,B,C,D) #qiu xianduan jiaodian
                    if intersection:
                        intersection_lst.append(np.array([intersection[0],intersection[1]]))

            #find closest intersction
            y = D.copy()
            if len(intersection_lst)>0:
                min_dist = S*S
                for intersection in intersection_lst:
                    d = Room.dist2(intersection,x)
                    if d<min_dist:
                        y = intersection.copy()
                        min_dist = d
            # cross = np.cross(vij,vjk)
            # if cross > 0:  #-| limit
            #     y = points[j]
            # else: # -- or -|
            #     y = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S

            #####calc z
            C = x
            Dxz = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S
            Dyz = y + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S

            line2 = Dxz-C
            intersection_lst = []
            dmin2 = S*S
            for t in range(1,l-1):
                A = points[(i+t)%l]
                B = points[(i+t+1)%l]
                line1 = B-A
                if Room.is_perpendicular(line1,line2) and np.cross(line2,line1)>0: #chuizhi & left
                    #xz
                    intersection = Room.segments_intersect(A,B,x,Dxz) #qiu xianduan jiaodian
                    if intersection:
                        dmin2 = min(dmin2,Room.dist2(intersection,x))
                    #yz
                    intersection = Room.segments_intersect(A,B,y,Dyz) #qiu xianduan jiaodian
                    if intersection:
                        dmin2 = min(dmin2,Room.dist2(intersection,y))
                        

            #find closest intersction
            d = np.sqrt(dmin2)
            z = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*d
    
            # cross = np.cross(vij,vjk)
            # if cross > 0:  #-| limit
            #     y = points[j]
            # else: # -- or -|
            #     y = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S

            sizez = S
            meanz = 0
            if x[0]==y[0]:
                sizey = abs(x[1]-y[1])
                sizex = abs(x[0]-z[0])
                meany = (x[1]+y[1])/2
                meanx = (x[0]+z[0])/2
            else:
                sizex = abs(x[0]-y[0])
                sizey = abs(x[1]-z[1])
                meany = (x[1]+z[1])/2
                meanx = (x[0]+y[0])/2

            
            translation = np.array([meanx,meany,meanz])
            size = np.array([sizex,sizey,sizez])# half size
            boxes["translation"].append(translation)
            boxes["size"].append(size)

        boxes["translation"] = np.array(boxes["translation"])
        boxes["size"] = np.array(boxes["size"])
        # 
        gt_boxes = [[boxes["translation"][i][0],boxes["translation"][i][1],boxes["translation"][i][2],boxes["size"][i][0],boxes["size"][i][1],boxes["size"][i][2],0] for i in range(len(boxes["translation"]))]
        gt_boxes = np.array(gt_boxes)
        # print(gt_boxes)
        # vis = draw_box_label(gt_boxes, (0, 0, 1))

        return  boxes
        
        


    @cached_property
    def floor_plan_outer_bbox(self):
        S = 50
        flag, points = self.floor_plan_polygon()
        if flag:
            boxes = self.calc_box_from_polygon(points,S)
            return True, boxes
        else:
            return False, None


    @cached_property
    def floor_plan_centroid(self):
        a, b = self.floor_plan_bbox
        return (a+b)/2

    @cached_property
    def centroid(self):
        return self.floor_plan_centroid

    @property
    def count_furniture_in_room(self):
        return Counter(self.furniture_in_room)

    @property
    def room_mask(self):
        return self.room_mask_rotated(0)
    
    @cached_property
    def load_room_mask_cache(self):
        im = Image.open(self.path_to_room_mask).convert("RGB")
        return im

    def room_mask_rotated(self, angle=0):
        # The angle is in rad
        im = self.load_room_mask_cache
        # Downsample the room_mask image by applying bilinear interpolation
        im = im.rotate(angle * 180 / np.pi, resample=Image.BICUBIC)

        return np.asarray(im).astype(np.float32) / np.float32(255)

    def category_counts(self, class_labels):
        """List of category counts in the room
        """
        print(class_labels)
        if "empty" in class_labels:
            class_labels = class_labels[:-1]
        category_counts = [0]*len(class_labels)

        for di in self.furniture_in_room:
            category_counts[class_labels.index(di)] += 1
        return category_counts

    def ordered_bboxes_with_centroid(self):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        int_labels = np.array(
            [[f.int_label(all_labels)] for f in self.bboxes]
        )
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_frequencies(self, class_order):
        centroids = np.array([f.centroid(-self.centroid) for f in self.bboxes])
        label_order = np.array([
            [class_order[f.label]] for f in self.bboxes
        ])
        ordering = np.lexsort(np.hstack([centroids, label_order]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering[::-1]]

        return ordered_bboxes

    def furniture_renderables(
        self,
        colors=(0.5, 0.5, 0.5),
        with_bbox_corners=False,
        with_origin=False,
        with_bboxes=False,
        with_objects_offset=False,
        with_floor_plan_offset=False,
        with_floor_plan=False,
        with_texture=True
    ):
        if with_objects_offset:
            offset = -self.bboxes_centroid
        elif with_floor_plan_offset:
            offset = -self.floor_plan_centroid
        else:
            offset = [[0, 0, 0]]

        renderables = [
            f.mesh_renderable(
                colors=colors, offset=offset, with_texture=with_texture
            )
            for f in self.bboxes
        ]
        if with_origin:
            renderables += [f.origin_renderable(offset) for f in self.bboxes]
        if with_bbox_corners:
            for f in self.bboxes:
                renderables += [f.bbox_corners_renderable(offset=offset)]
        if with_bboxes:
            for f in self.bboxes:
                renderables += [f.bbox_renderable(offset=offset)]
        if with_floor_plan:
            vertices, faces = self.floor_plan
            vertices = vertices + offset
            renderables += [
                Mesh.from_faces(vertices, faces, colors=(0.8, 0.8, 0.8, 0.6))
            ]
        return renderables

    def show(
        self,
        behaviours=[LightToCamera(), SnapshotOnKey()],
        with_bbox_corners=False,
        with_bboxes=False,
        with_objects_offset=False,
        with_floor_plan_offset=False,
        with_floor_plan=False,
        background=(1.0, 1.0, 1.0, 1.0),
        camera_target=(0, 0, 0),
        camera_position=(-2, -2, -2),
        up_vector=(0, 0, 1),
        window_size=(512, 512)
    ):
        renderables = self.furniture_renderables(
            with_bbox_corners=with_bbox_corners,
            with_bboxes=with_bboxes,
            with_objects_offset=with_objects_offset,
            with_floor_plan_offset=with_floor_plan_offset,
            with_floor_plan=with_floor_plan
        )
        show(
            renderables, behaviours=behaviours,
            size=window_size, camera_position=camera_position,
            camera_target=camera_target, up_vector=up_vector,
            background=background
        )


    def furniture_trimesh_meshes(
        self,
        colors=(0.5, 0.5, 0.5),
        # with_bbox_corners=False,
        # with_origin=False,
        # with_bboxes=False,
        with_objects_offset=False,
        with_floor_plan_offset=False,
        with_floor_plan=False,
        with_texture=True
    ):
        if with_objects_offset:
            offset = -self.bboxes_centroid
        elif with_floor_plan_offset:
            offset = -self.floor_plan_centroid
        else:
            offset = [[0, 0, 0]]

        trimesh_meshes = [
            f.mesh_trimesh(
                colors=colors, offset=offset, with_texture=with_texture
            )
            for f in self.bboxes
        ]
        # if with_origin:
        #     renderables += [f.origin_renderable(offset) for f in self.bboxes]
        # if with_bbox_corners:
        #     for f in self.bboxes:
        #         renderables += [f.bbox_corners_renderable(offset=offset)]
        # if with_bboxes:
        #     for f in self.bboxes:
        #         renderables += [f.bbox_renderable(offset=offset)]
        if with_floor_plan:
            vertices, faces = self.floor_plan
            vertices = vertices + offset
            
            tr_floor = trimesh.Trimesh(
                np.copy(vertices), np.copy(faces), process=False
            )
            #gray color
            tr_floor.visual.vertex_colors = (np.ones((len(vertices), 3)) * 255/2).astype(np.uint8)
            trimesh_meshes += [tr_floor]
        return trimesh_meshes

    def save_mesh(self,path_to_scene="debug.obj"):
        trimesh_meshes = self.furniture_trimesh_meshes(with_floor_plan=True)
        if trimesh_meshes is not None:
            # whole_scene_mesh = merge_meshes( trimesh_meshes )
            whole_scene_mesh = trimesh.util.concatenate(trimesh_meshes)
            whole_scene_mesh.export(path_to_scene)
            # o3d.io.write_triangle_mesh(path_to_scene, whole_scene_mesh)


    def augment_room(self, objects_dataset):
        bboxes = self.bboxes
        # Randomly pick an asset to be augmented
        bi = np.random.choice(self.bboxes)
        query_label = bi.label
        query_size = bi.size + np.random.normal(0, 0.02)
        # Retrieve the new asset based on the size of the picked asset
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )
        bi_retrieved = bi.copy_from_other_model(furniture)

        new_bboxes = [
            box for box in bboxes if not box == bi
        ] + [bi_retrieved]

        return Room(
            scene_id=self.scene_id + "_augm",
            scene_type=self.scene_type,
            bboxes=new_bboxes,
            extras=self.extras,
            json_path=self.json_path,
            path_to_room_masks_dir=self.path_to_room_masks_dir
        )
