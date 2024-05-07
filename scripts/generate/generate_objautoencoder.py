"""Script used to test a Shape Autoencoder."""
import logging
import os
import sys
sys.path.insert(0,sys.path[0]+"/../../")
import numpy as np

import torch
from torch.utils.data import DataLoader
import pickle
from scripts.train.training_utils import id_generator, load_config
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets.base import filter_function
from models.networks import optimizer_factory, schedule_factory
# from scene_diffusion.stats_logger import StatsLogger, WandB
from scripts.train.training_utils import load_checkpoints
from models.networks.foldingnet_autoencoder import AutoEncoder, KLAutoEncoder, train_on_batch, validate_on_batch
from datasets.threed_front import ThreedFront
from datasets.threed_future_dataset import ThreedFutureNormPCDataset
from datasets.utils_io import export_pointcloud, load_pointcloud
from datasets.gapartnet_dataset import GAPartNetDataset

@hydra.main(version_base=None, config_path="../../configs", config_name="autoencoder")
def main(cfg: DictConfig):
    global config
    config = cfg
    os.environ["PATH_TO_SCENES"] = cfg.PATH_TO_SCENES
    os.environ["BASE_DIR"] = cfg.BASE_DIR
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(cfg.output_directory):
        os.makedirs(cfg.output_directory)

    # Create an experiment directory using the experiment_tag
    if cfg.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = cfg.experiment_tag

    experiment_directory = os.path.join(
        cfg.output_directory,
        experiment_tag
    )

    # Parse the config file
    config_VAE = cfg.obj_autoencoder

    ## add bed rooms
    config1 = {
        "filter_fn":                 "threed_front_bedroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config_VAE["dataset"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config_VAE["dataset"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "configs/data/bedroom_threed_front_splits.csv"
    }

    scenes_train_dataset = ThreedFront.from_dataset_directory(
        dataset_directory=config["dataset"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["dataset"]["path_to_model_info"],
        path_to_models=config["dataset"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config1, config_VAE["training"].get("splits", ["train", "val", "test"]), config_VAE["dataset"]["without_lamps"])
    )
    print("Loading train dataset with {} rooms".format(len(scenes_train_dataset)))


    # add dining rooms
    config2 = {
        "filter_fn":                 "threed_front_diningroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config_VAE["dataset"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config_VAE["dataset"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "configs/data/diningroom_threed_front_splits.csv"
    }
    scenes_train_dataset2 = ThreedFront.from_dataset_directory(
        dataset_directory=config["dataset"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["dataset"]["path_to_model_info"],
        path_to_models=config["dataset"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config2, config_VAE["training"].get("splits", ["train", "val", "test"]), config_VAE["dataset"]["without_lamps"])
    )
    print("Loading train dataset 2 with {} rooms".format(len(scenes_train_dataset2)))

    ## add living rooms
    config3 = {
        "filter_fn":                 "threed_front_livingroom",
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": config_VAE["dataset"]["path_to_invalid_scene_ids"],
        "path_to_invalid_bbox_jids": config_VAE["dataset"]["path_to_invalid_bbox_jids"],
        "annotation_file":           "configs/data/livingroom_threed_front_splits.csv"
    }
    scenes_train_dataset3 = ThreedFront.from_dataset_directory(
        dataset_directory=config["dataset"]["path_to_3d_front_dataset_directory"],
        path_to_model_info=config["dataset"]["path_to_model_info"],
        path_to_models=config["dataset"]["path_to_3d_future_dataset_directory"],
        filter_fn=filter_function(config3, config_VAE["training"].get("splits", ["train", "val", "test"]), config_VAE["dataset"]["without_lamps"])
    )
    print("Loading train dataset 3 with {} rooms".format(len(scenes_train_dataset3)))

    # scenes_validation_dataset = ThreedFront.from_dataset_directory(
    #     dataset_directory=config["data"]["path_to_3d_front_dataset_directory"],
    #     path_to_model_info=config["data"]["path_to_model_info"],
    #     path_to_models=config["data"]["path_to_3d_future_dataset_directory"],
    #     filter_fn=filter_function(config["data"], config["validation"].get("splits", ["test"]), config["data"]["without_lamps"])
    # )
    # print("Loading validation dataset with {} rooms".format(len(scenes_validation_dataset)))

    # Collect the set of objects in the scenes
    train_objects = {}
    for scene in scenes_train_dataset:
        for obj in scene.bboxes:
            train_objects[obj.model_jid] = obj
    # diningroom
    for scene in scenes_train_dataset2:
        for obj in scene.bboxes:
            train_objects[obj.model_jid] = obj
    # livingroom
    for scene in scenes_train_dataset3:
        for obj in scene.bboxes:
            train_objects[obj.model_jid] = obj


    # Build the dataset of Garpartnet
    pickled_GPN_dir = cfg.GAPartNet.pickled_GPN_dir
    pickled_GPN_path = "{}/gapartnet_model.pkl".format(pickled_GPN_dir)
    if os.path.exists(pickled_GPN_path):
        gapartnet_dataset = GAPartNetDataset.from_pickled_dataset(pickled_GPN_path)
    else:
        gapartnet_dataset = GAPartNetDataset(cfg)
        with open(pickled_GPN_path, "wb") as f:
            pickle.dump(gapartnet_dataset, f)

    for obj in gapartnet_dataset.objects:
        train_objects[obj.model_jid] = obj

    train_objects = [vi for vi in train_objects.values()]
    train_dataset = ThreedFutureNormPCDataset(train_objects)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_VAE["training"].get("batch_size", 128),
        num_workers=cfg.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} train objects".format(
        len(train_dataset))
    )

    # val_loader = DataLoader(
    #     validation_dataset,
    #     batch_size=config["validation"].get("batch_size", 1),
    #     num_workers=args.n_processes,
    #     collate_fn=validation_dataset.collate_fn,
    #     shuffle=False
    # )
    # print("Loaded {} validation objects".format(
    #     len(validation_dataset))
    # )

    # Build the network architecture to be used for training
    ### instead of using build_network, we directly build from config
    network = KLAutoEncoder(latent_dim=config_VAE["network"].get("objfeat_dim", 64),  kl_weight=config_VAE["network"].get("kl_weight", 0.001))
    if cfg.weight_file is not None:
        print("Loading weight file from {}".format(cfg.weight_file))
        network.load_state_dict(
            torch.load(cfg.weight_file, map_location=device)
        )
    network.to(device)
    ####
    n_all_params = int(sum([np.prod(p.size()) for p in network.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, network.parameters())]))
    print(f"Number of parameters in {network.__class__.__name__}:  {n_trainable_params} / {n_all_params}")

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config_VAE["training"], filter(lambda p: p.requires_grad, network.parameters()) ) 

    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, cfg, device)
    # Load the learning rate scheduler 
    lr_scheduler = schedule_factory(config_VAE["training"])

    generation_directory = os.path.join(
        cfg.output_directory,
        experiment_tag,
        "generation"
    )
    if not os.path.exists(generation_directory):
        os.makedirs(generation_directory)

    lat_list = []
    with torch.no_grad():
        print("====> Validation Epoch ====>")
        network.eval()
        for b, sample in enumerate(train_loader):
        # for b, sample in enumerate(val_loader):
            # Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list):
                    sample[k] = v.to(device)
            kl, lat, rec = network(sample["points"])
            idx = sample["idx"]
            lat_list.append(lat)

            for i in range(lat.shape[0]):
                lat_i = lat[i].cpu().numpy()
                pc_i = sample["points"][i].cpu().numpy()
                rec_i = rec[i].cpu().numpy()
                idx_i = idx[i].item()

                # save obj autoencoder results for vis check
                model_jid = train_dataset.get_model_jid(idx_i)["model_jid"]
                # model_jid = validation_dataset.get_model_jid(idx_i)["model_jid"]
                filename_input = "{}/{}.ply".format(generation_directory, model_jid)
                filename_rec  =  "{}/{}_rec.ply".format(generation_directory, model_jid)
                export_pointcloud(pc_i, filename_input)
                export_pointcloud(rec_i, filename_rec)


                latent_dim=config_VAE["network"].get("objfeat_dim", 64)
                #save objfeat i.e. latent
                obj = train_dataset.objects[idx_i]
                # obj = validation_dataset.objects[idx_i]
                assert model_jid == obj.model_jid
                raw_model_path = obj.raw_model_path
                if latent_dim == 32:
                    filename_lats = obj.raw_model_norm_pc_lat32_path
                else:
                    filename_lats = raw_model_path[:-4] + "_norm_pc_lat{:d}.npz".format(latent_dim)

                if os.path.exists(filename_lats):
                    continue

                np.savez(filename_lats, latent=lat_i)
                print(filename_lats)

            print('iter {}'.format(b), lat.shape, lat.min(), lat.max(), rec.shape)
            
        lat_all = torch.cat(lat_list, dim=0)
        print('before: std {}, min {}, max {}'.format(lat_all.flatten().std(), lat_all.min(), lat_all.max()) )
        scale_factor = 1.0 / lat_all.flatten().std()
        print('scale factor:', scale_factor)
        lat_scaled = lat_all * scale_factor
        print('after: std {}, min {}, max {}'.format(lat_scaled.flatten().std(), lat_scaled.min(), lat_scaled.max()) )
        print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main()

