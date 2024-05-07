save_dir=$1  # autoencoder_output
experiment_tag=$2 # debug

python scripts/train/train_objautoencoder.py hydra/job_logging=none hydra/hydra_logging=none \
                output_directory=${save_dir} \
                continue_epoch=0 \
                experiment_tag=${experiment_tag} 
                
                            
# python scripts/generate/generate_objautoencoder.py hydra/job_logging=none hydra/hydra_logging=none \
#                 output_directory=${save_dir} \
#                 experiment_tag=${experiment_tag}  
#                 # weight_file=autoencoder_output/debug/model_02900 