# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import numpy as np
import pickle

import numpy as np
from torch.utils.data import Dataset
import random
import os
import numpy as np
import json
from simple_3dviz.renderables.textured_mesh import TexturedMesh

from functools import cached_property
import json

OBJECT_CATEGORIES = [
    'Box', 'Camera', 'CoffeeMachine', 'Dishwasher', 'KitchenPot', 'Microwave', 'Oven', 'Phone', 'Refrigerator',
    'Remote', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Keyboard', 'Laptop', 'Door', 'Printer',
    'Suitcase', 'Bucket', 'Toilet'
]

ThreedFutureClasses = ['armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair', 'children_cabinet', 'coffee_table', \
                       'desk', 'double_bed', 'dressing_chair', 'dressing_table', 'kids_bed', 'nightstand', 'pendant_lamp', \
                       'shelf','single_bed','sofa','stool','table','tv_stand','wardrobe','start','end',\
                        'dining_table','corner_side_table','console_table','round_end_table',\
                        'wine_cabinet']

MapThreedfuture2gparnet = {'dressing_table':'Table', 
                                      'bookshelf':'StorageFurniture',
                                      'cabinet':'StorageFurniture',
                                      'children_cabinet':'StorageFurniture',
                                      'coffee_table':'Table',
                                      'desk':'Table',
                                      'shelf':'StorageFurniture',
                                      'table':'Table',
                                      'wardrobe':'StorageFurniture',
                                      'dining_table':'Table',
                                      'corner_side_table':'Table',
                                      'console_table':'Table', 
                                      'wine_cabinet':'StorageFurniture',
                                      'round_end_table':'Table', 
                                      'nightstand':'StorageFurniture',  #'StorageFurniture',
                                      'tv_stand': 'StorageFurniture',
                                      }

class GapartnetModel():
    def __init__(
        self,
        category,
        model_uid,
        model_jid,
        path_to_models,
        model_info=None,
        position=None,
        rotation=None,
        scale=1
        
    ):
        self.model_uid = model_uid
        self.model_jid = model_jid
        self.position = position
        self.rotation = rotation
        self.scale = scale

        self.model_info = model_info
        self.path_to_models = path_to_models
        self.label = category
        # self.raw_model_path = self.raw_model_path()
        self.compute_bbox()
        self.size = self.compute_size()
        self.pcd = None
        self.expand_ratio = np.array([0,0,0,0,0,0])
        self.compute_ratio()

    

    @property
    def raw_model_path(self):
        model_dir = os.path.join(
            self.path_to_models,
            self.model_jid,
            "textured_objs")
            # "textured_objs_clean")
        lst = os.listdir(model_dir)
        lst_obj = []
        for i in lst:
            if ".obj" in i and "stl" not in i and "modified" not in i and "mtl" not in i \
                and "remesh" not in i and "right_direction" not in i and "sequence" not in i \
                and "closed" not in i and "open" not in i:
                lst_obj.append(model_dir+"/"+i)
        return lst_obj

    @property
    def raw_model_path_close(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "textured_objs",
            "right_direction_close.obj")

    @property
    def raw_model_path_open(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "textured_objs",
            "right_direction_open.obj")
    
    @property
    def sequence_model_path(self):
        model_dir = os.path.join(
            self.path_to_models,
            self.model_jid,
            "textured_objs")
        lst_obj = [model_dir+"/right_direction_open.obj"]
        return lst_obj
    
    @property
    def remesh_model_path(self):
        return 
        # model_dir = os.path.join(
        #     self.path_to_models,
        #     self.model_jid,
        #     "textured_objs")
        # lst_obj = []
        # lst_obj = [model_dir+"/remesh_closed.obj"]
        # return lst_obj
    
    # add normalized point cloud of raw_model
    @property
    def raw_model_norm_pc_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "textured_objs",
            "raw_model_norm_pc.npz"
        )
    
    @property
    def raw_model_norm_pc_lat32_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "textured_objs",
            "raw_model_norm_pc_lat32.npz"
        )
    
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
    
    # add normalized point cloud of raw_model
    def raw_model_norm_pc(self):
        points = np.load(self.raw_model_norm_pc_path)["points"].astype(np.float32)
        return points
    
    # def raw_model_norm_pc_lat(self):
    #     latent = np.load(self.raw_model_norm_pc_lat_path)["latent"].astype(np.float32)
    #     return latent
    
    @property
    def texture_image_path(self):
        return os.path.join(
            self.path_to_models,
            self.model_jid,
            "texture.png"
        )
    
    # def compute_bbox(self):
        
    #     def expand_bbox(bbox):
    #         if self.bbox==None:
    #             self.bbox = [[],[]]
    #             self.bbox[0] = bbox[0]
    #             self.bbox[1] = bbox[1]
    #         else:
    #             if bbox[0][0]<self.bbox[0][0]:
    #                 self.bbox[0][0] = bbox[0][0]
    #             if bbox[0][1]<self.bbox[0][1]:
    #                 self.bbox[0][1] = bbox[0][1]
    #             if bbox[0][2]<self.bbox[0][2]:
    #                 self.bbox[0][2] = bbox[0][2]

    #             if bbox[1][0]>self.bbox[1][0]:
    #                 self.bbox[1][0] = bbox[1][0]
    #             if bbox[1][1]>self.bbox[1][1]:
    #                 self.bbox[1][1] = bbox[1][1]
    #             if bbox[1][2]>self.bbox[1][2]:
    #                 self.bbox[1][2] = bbox[1][2]
    #         return
        
    #     self.bbox = None
    #     for raw_model_path in self.raw_model_path:
    #         # Load the furniture and scale it as it is given in the dataset
    #         raw_mesh = TexturedMesh.from_file(raw_model_path)
    #         raw_mesh.scale(self.scale)
    #         bbox = raw_mesh.bbox
    #         expand_bbox(bbox)
    #     self.bbox = np.array(self.bbox)
        
    #     return self.bbox

    def compute_bbox(self):
        self.bbox = None
        # Load the furniture and scale it as it is given in the dataset
        raw_mesh = TexturedMesh.from_file(self.raw_model_path_close)
        raw_mesh.scale(self.scale)
        bbox = raw_mesh.bbox
        self.bbox = np.array(bbox)
        
        return self.bbox

    def compute_ratio(self):

        centroid = (self.bbox[0] + self.bbox[1])/2
        centroid = np.tile(centroid[None,:],[2,1])
        bbox_close = self.bbox - centroid

        raw_mesh = TexturedMesh.from_file(self.raw_model_path_open)
        raw_mesh.scale(self.scale)
        bbox_open = raw_mesh.bbox - centroid

        ratio = bbox_open/bbox_close
        # print(ratio)
        self.expand_ratio = np.log(ratio)  #[1,infinit) --> [0,infinite)
        self.expand_ratio = self.expand_ratio.reshape([1,6])
        # print(self.model_uid,self.model_jid,self.expand_ratio)
        return self.expand_ratio

    def compute_size(self):
        min = self.bbox[0]
        max = self.bbox[1]
        size = [max[i]-min[i] for i in range(3)]
        return size

class GAPartNetDataset(Dataset):
    def __init__(
        self,cfg,
        type=None,
        remove_largeopen=False
    ): 
        # split train & eval
        # self.model_id_list = []
        self.model_id_list_train = []
        self.model_id_list_val = []
        idx = 0
        with open(cfg.GAPartNet.ID_PATH, 'r') as fd:
            for line in fd:
                idx += 1
                ls = line.strip().split(' ')
                # self.model_id_list.append((ls[0], int(ls[1])))
                if idx%30!=0:
                    self.model_id_list_train.append((ls[0], int(ls[1])))
                else:
                    self.model_id_list_val.append((ls[0], int(ls[1])))
        path_to_models = cfg.GAPartNet.path_to_models
        self.objects = []
        if type=="train":
            self.model_id_list = self.model_id_list_train
        elif type == "test":
            self.model_id_list = self.model_id_list_val
        else:
            self.model_id_list = self.model_id_list_train+self.model_id_list_val

        # self.save_valid_expand_ratios(cfg)
        if remove_largeopen:

            #### #filter by open ratio
            # with open(cfg.GAPartNet.GPN_open_ratio) as f:
            #     import json
            #     j = json.load(f)
            #     valid_id = []
            #     for id in j.keys():
            #         valid_id.append(id)
            # self.objects = []

            #### #filter manually
            with open(cfg.GAPartNet.GPN_good_idx) as f:
                j = json.load(f)
                valid_id = j["id_lst"]
            self.objects = []
            for category, model_id in self.model_id_list:
                if str(model_id) not in valid_id:
                    continue
                object = GapartnetModel(category, str(model_id),str(model_id),path_to_models)
                self.objects.append(object)
        else:
            for category, model_id in self.model_id_list:
                object = GapartnetModel(category, str(model_id),str(model_id),path_to_models)
                self.objects.append(object)

        return

        
    # def save_valid_expand_ratios(self,cfg):
    #     GPN_open_ratio = dict()
    #     for obj in self.objects:
    #         # train_objects[obj.model_jid] = obj
    #         if obj.label not in ["StorageFurniture","Table"]:
    #             continue
    #         if obj.expand_ratio.max()>1.5:
    #             continue
    #         GPN_open_ratio[obj.model_jid] = dict()
    #         GPN_open_ratio[obj.model_jid]["class"] = obj.label
    #         GPN_open_ratio[obj.model_jid]["ratio"] = obj.expand_ratio.tolist()   

    #     with open(cfg.GAPartNet.GPN_open_ratio, "w") as myfile:
    #         json.dump(GPN_open_ratio,myfile,indent=4)
            
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        self.object[idx]
   
    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]
    
    def _filter_objects_by_id(self, id_lst):
        return [oi for oi in self.objects if oi.model_jid in id_lst]

    def _show_single_object(self,obj,floor_plan):
        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)

        # Extract the predicted affine transformation to position the
        # mesh

        
        renderables = []
        for raw_model_path in obj.raw_model_path:
            # Load the furniture and scale it as it is given in the dataset
            raw_mesh = TexturedMesh.from_file(raw_model_path)
            raw_mesh.scale(obj.scale)

            # Apply the transformations in order to correctly position the mesh
            try:
                renderables += raw_mesh.renderables
            except:
                renderables.append(raw_mesh)
        renderables += floor_plan
        return  renderables

        # show(
        #     renderables,
        #     behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
        #     size=(512,512),
        #     camera_position="-0.10923499,1.9325259,-7.19009",
        #     camera_target="0,0,0",
        #     up_vector="0,1,0",
        #     background="0.5,0.5,0.5,0.5",
        #     title="GPN MODEL"
        # )
        # return 
    

    def _show_objects_of_type(self,query_label,floor_plan):
        objects = self._filter_objects_by_label(query_label)
        for obj in objects:
            self._show_single_object(obj,floor_plan)

    def get_closest_furniture_to_box(self, query_label, query_size):
        # objects = self._filter_objects_by_label(query_label)
        objects = self._filter_objects_by_label("Table") + self._filter_objects_by_label("StorageFurniture")

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]
    
    def get_closest_furniture_to_box_normsize(self, query_label, query_size):
        # objects = self._filter_objects_by_label(query_label)
        # if query_label=='Table':
        #     a=1
        objects = self._filter_objects_by_label("Table") + self._filter_objects_by_label("StorageFurniture")
        # objects = self._filter_objects_by_id(["48036","47187","27619","45503"])

        mses = {}
        for i, oi in enumerate(objects):
            # mses[oi] = np.sum((oi.size/max(oi.size) - query_size/max(query_size))**2, axis=-1)
            mses[oi] = np.sum((oi.size/np.linalg.norm(oi.size) - query_size/np.linalg.norm(query_size))**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]
    
    def get_closest_furniture_to_objfeats_and_size(self, query_label, query_objfeat, query_size):
        objects = self._filter_objects_by_label(query_label)
        # objects = self.objects
        # objects = self._filter_objects_by_label("Table") + self._filter_objects_by_label("StorageFurniture")

        objs = []
        mses_feat = []
        mses_size = []
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                mses_feat.append( np.sum((oi.raw_model_norm_pc_lat32() - query_objfeat)**2, axis=-1) )
            elif query_objfeat.shape[0] == 512:
                mses_feat.append( np.sum((oi.raw_model_norm_pc_lat_ulip() - query_objfeat)**2, axis=-1) )
            else:
                mses_feat.append( np.sum((oi.raw_model_norm_pc_lat() - query_objfeat)**2, axis=-1) )
            mses_size.append( np.sum((oi.size - query_size)**2, axis=-1) )
            objs.append(oi)


        ind = np.lexsort( (mses_feat,mses_feat) )
        return objs[ ind[0] ]
    
    def random_choose_furniture(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)  
        return random.sample(objects,1)[0]

    # def get_closest_furniture_to_2dbox(self, query_label, query_size):
    #     objects = self._filter_objects_by_label(query_label)

    #     mses = {}
    #     for i, oi in enumerate(objects):
    #         mses[oi] = (
    #             (oi.size[0] - query_size[0])**2 +
    #             (oi.size[2] - query_size[1])**2
    #         )
    #     sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
    #     return sorted_mses[0]

    # @classmethod
    # def from_dataset_directory(
    #     cls, dataset_directory, path_to_model_info, path_to_models
    # ):
    #     objects = parse_threed_future_models(
    #         dataset_directory, path_to_models, path_to_model_info
    #     )
    #     return cls(objects)

    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
    
    