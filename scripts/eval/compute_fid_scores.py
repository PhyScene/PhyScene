# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script for computing the FID/KID score between real and synthesized scenes.
"""
import argparse
import os
import sys
sys.path.insert(0,sys.path[0]+"/../../")

import torch

import numpy as np
from PIL import Image

from cleanfid import fid

import shutil
import random
from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder
from scene_synthesis.datasets.threed_front import CachedThreedFront


class ThreedFrontRenderDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx].image_path
        img = Image.open(image_path)
        return img


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "--path_to_real_renderings",
        default="../../data/preprocessed_data/BedRoom/",
        help="Path to the folder containing the real renderings"
    )
    parser.add_argument(
        "--path_to_synthesized_renderings",
        default="../../result/render/",
        help="Path to the folder containing the synthesized"
    )
    parser.add_argument(
        "--path_to_annotations",
        default="../../configs/data/bedroom_threed_front_splits.csv",
        help="Path to the folder containing the annotations"
    )

    args = parser.parse_args(argv)

    # Create Real datasets
    config = dict(
        train_stats="dataset_stats.txt",
        room_layout_size="256,256"
    )
    splits_builder = CSVSplitsBuilder(args.path_to_annotations)
    test_real = ThreedFrontRenderDataset(CachedThreedFront(
        args.path_to_real_renderings,
        config=config,
        scene_ids=splits_builder.get_splits(["train","val","test"])
        # scene_ids=splits_builder.get_splits(["train"])
    ))

    

    fid_scores = []
    kid_scores = []
    for _ in range(10):

        print("Generating temporary a folder with test_real images...")
        path_to_test_real = "/tmp/test_real/"
        if os.path.exists(path_to_test_real):
            os.system("rm -r "+path_to_test_real)
        os.makedirs(path_to_test_real)
        for i, di in enumerate(test_real):
            di.save("{}/{:05d}.png".format(path_to_test_real, i))
        # Number of images to be copied
        N = len(test_real)

        print("Generating temporary a folder with test_fake images...")
        path_to_test_fake = "/tmp/test_fake/"
        if os.path.exists(path_to_test_fake):
            os.system("rm -r "+path_to_test_fake)
        os.makedirs(path_to_test_fake)

        synthesized_images = [
            os.path.join(args.path_to_synthesized_renderings, oi)
            for oi in os.listdir(args.path_to_synthesized_renderings)
            if oi.endswith(".png")
        ]

        np.random.shuffle(synthesized_images)
        synthesized_images_subset = np.random.choice(synthesized_images, N)
        for i, fi in enumerate(synthesized_images_subset):
            shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

        # Compute the FID score
        fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
        kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
        fid_scores.append(fid_score)
        kid_scores.append(kid_score)
        print(fid_score,kid_score)
    print("fid: " ,sum(fid_scores) / len(fid_scores))
    print("kid: " ,sum(kid_scores) / len(kid_scores))
    # print(np.std(scores))


if __name__ == "__main__":
    main(None)
