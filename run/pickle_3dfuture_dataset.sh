save_dir=$1   #data/pickled_data

python scripts/preprocess/pickle_threed_future_dataset.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            output_directory=${save_dir} \
            task=scene_bedroom

python scripts/preprocess/pickle_threed_future_dataset.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            output_directory=${save_dir} \
            task=scene_livingroom
                
python scripts/preprocess/pickle_threed_future_dataset.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            output_directory=${save_dir} \
            task=scene_diningroom