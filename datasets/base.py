from typing import Dict
from torch.utils.data import Dataset

import numpy as np
import os
from .threed_front import ThreedFront, CachedThreedFront
from .threed_front_dataset import dataset_encoding_factory
from .common import BaseDataset
from .splits_builder import CSVSplitsBuilder


def get_raw_dataset(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    split=["train", "val"]
):
    dataset_type = config["dataset_type"]
    if "cached" in dataset_type:
        # Make the train/test/validation splits
        dataset_directory = config["dataset_directory"]
        annotation_file = config["annotation_file"]
        splits_builder = CSVSplitsBuilder(annotation_file)
        split_scene_ids = splits_builder.get_splits(split)

        dataset = CachedThreedFront(
            dataset_directory,
            config=config,
            scene_ids=split_scene_ids,
        )
    else:
        dataset_directory = config["dataset_directory"]
        path_to_model_info = config["path_to_model_info"]
        path_to_models = config["path_to_models"]
        path_to_room_masks_dir = config["path_to_room_masks_dir"]
        dataset = ThreedFront.from_dataset_directory(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir,
            path_to_bounds,
            filter_fn,
        )
    return dataset

def get_dataset_raw_and_encoded(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"],
):
    dataset = get_raw_dataset(config, filter_fn, path_to_bounds, split=split)
    encoding = dataset_encoding_factory(
        config.get("encoding_type"),
        dataset,
        augmentations,
        config.get("box_ordering", None)
    )

    return dataset, encoding

def get_encoded_dataset(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"]
):
    _, encoding = get_dataset_raw_and_encoded(
        config, filter_fn, path_to_bounds, augmentations, split
    )
    return encoding


def create_dataset(cfg: dict, phase: str, **kwargs: Dict) -> Dataset:
    """ Create a `torch.utils.data.Dataset` object from configuration.

    Args:
        cfg: configuration object, dataset configuration
        phase: phase string, can be 'train' and 'test'
    
    Return:
        A Dataset object that has loaded the designated dataset.
    """
    if cfg.name!="scene_synthesis":
        return DATASET.get(cfg.dataset.name)(cfg, phase, **kwargs)
    else:
        if phase=="train":
            dataset = get_encoded_dataset(
                cfg['dataset'],
                filter_function(
                    cfg['dataset'],
                    split=cfg["train"].get("splits", ["train", "val"])
                ),
                path_to_bounds=None,
                augmentations=cfg["dataset"].get("augmentations", None),
                split=cfg["train"].get("splits", ["train", "val"])
            )
            path_to_bounds = os.path.join(cfg.dataset.path_to_bounds)
            np.savez(
                path_to_bounds,
                sizes=dataset.bounds["sizes"],
                translations=dataset.bounds["translations"],
                angles=dataset.bounds["angles"],
                objfeats_32=dataset.bounds["objfeats_32"],
                # objfeats_pc_ulip=dataset.bounds["objfeats_pc_ulip"]
            )
            print("Saved the dataset bounds in {}".format(path_to_bounds))

        else:
            path_to_bounds = os.path.join(cfg.dataset.path_to_bounds)
            dataset = get_encoded_dataset(
                cfg["dataset"],
                filter_function(
                    cfg["dataset"],
                    split=cfg["test"].get("splits", ["test"])
                ),
                path_to_bounds=path_to_bounds,
                augmentations=None,
                split=cfg["test"].get("splits", ["test"])
            )
        return dataset



def filter_function(config, split=["train", "val"], without_lamps=False):
    print("Applying {} filtering".format(config["filter_fn"]))

    if config["filter_fn"] == "no_filtering":
        return lambda s: s

    # Parse the list of the invalid scene ids
    with open(config["path_to_invalid_scene_ids"], "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)

    # Parse the list of the invalid bounding boxes
    with open(config["path_to_invalid_bbox_jids"], "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    # Make the train/test/validation splits
    splits_builder = CSVSplitsBuilder(config["annotation_file"])
    split_scene_ids = splits_builder.get_splits(split)

    if "threed_front_bedroom" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("bed"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(13),
            BaseDataset.with_object_types(
                list(THREED_FRONT_BEDROOM_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(THREED_FRONT_BEDROOM_FURNITURE),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.contains_object_types(
                ["double_bed", "single_bed", "kids_bed"]
            ),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(6, 6, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "threed_front_livingroom" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("living"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(21),
            BaseDataset.with_object_types(
                list(THREED_FRONT_LIVINGROOM_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(
                THREED_FRONT_LIVINGROOM_FURNITURE
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "threed_front_diningroom" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("dining"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.at_most_boxes(21),
            BaseDataset.with_object_types(
                list(THREED_FRONT_LIVINGROOM_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(
                THREED_FRONT_LIVINGROOM_FURNITURE
            ),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(12, 12, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif "threed_front_library" in config["filter_fn"]:
        return BaseDataset.filter_compose(
            BaseDataset.with_room("library"),
            BaseDataset.at_least_boxes(3),
            BaseDataset.with_object_types(
                list(THREED_FRONT_LIBRARY_FURNITURE.keys())
            ),
            BaseDataset.with_generic_classes(THREED_FRONT_LIBRARY_FURNITURE),
            BaseDataset.with_valid_scene_ids(invalid_scene_ids),
            BaseDataset.with_valid_bbox_jids(invalid_bbox_jids),
            BaseDataset.room_smaller_than_along_axis(4.0, axis=1),
            BaseDataset.room_larger_than_along_axis(-0.005, axis=1),
            BaseDataset.floor_plan_with_limits(6, 6, axis=[0, 2]),
            BaseDataset.without_box_types(
                ["ceiling_lamp", "pendant_lamp"]
                if without_lamps else [""]
            ),
            BaseDataset.with_scene_ids(split_scene_ids)
        )
    elif config["filter_fn"] == "non_empty":
        return lambda s: s if len(s.bboxes) > 0 else False






THREED_FRONT_BEDROOM_FURNITURE = {
    "desk":                                    "desk",
    "nightstand":                              "nightstand",
    "king-size bed":                           "double_bed",
    "single bed":                              "single_bed",
    "kids bed":                                "kids_bed",
    "ceiling lamp":                            "ceiling_lamp",
    "pendant lamp":                            "pendant_lamp",
    "bookcase/jewelry armoire":                "bookshelf",
    "tv stand":                                "tv_stand",
    "wardrobe":                                "wardrobe",
    "lounge chair/cafe chair/office chair":    "chair",
    "dining chair":                            "chair",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
    "corner/side table":                       "table",
    "dining table":                            "table",
    "round end table":                         "table",
    "drawer chest/corner cabinet":             "cabinet",
    "sideboard/side cabinet/console table":    "cabinet",
    "children cabinet":                        "children_cabinet",
    "shelf":                                   "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "coffee table":                            "coffee_table",
    "loveseat sofa":                           "sofa",
    "three-seat/multi-seat sofa":              "sofa",
    "l-shaped sofa":                           "sofa",
    "lazy sofa":                               "sofa",
    "chaise longue sofa":                      "sofa",
}

THREED_FRONT_LIBRARY_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair/cafe chair/office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
}

THREED_FRONT_LIVINGROOM_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair/cafe chair/office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "tv stand":                                "tv_stand"
}