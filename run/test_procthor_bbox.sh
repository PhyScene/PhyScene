save_dir=$1

# python scripts/generate/generate_bbox.py hydra/job_logging=none hydra/hydra_logging=none \
#                 exp_dir=${save_dir} \
#                 task=scene_livingroom \
#                 task.evaluator.weight_file=${save_dir}/model_134000  \
#                 evaluation.generate_result_json=true \
#                 evaluation.visual=true \
#                 task.test.batch_size=1 \
#                 task.dataset.path_to_bounds=demo/bounds.npz \
#                 ai2thor.path_to_result=/home/yandan/workspace/PhyScene/ai2thor/generate_bbox

#bedroom_new_1120
python scripts/generate/generate_bbox.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${save_dir} \
                task=scene_bedroom \
                task.evaluator.weight_file=${save_dir}/model_58000  \
                evaluation.generate_result_json=true \
                evaluation.visual=true \
                task.test.batch_size=1 \
                task.dataset.path_to_bounds=demo/bounds.npz \
                ai2thor.path_to_result=/home/yandan/workspace/PhyScene/ai2thor/generate_bbox

#diningroom_new_1120
# python scripts/generate/generate_bbox.py hydra/job_logging=none hydra/hydra_logging=none \
#                 exp_dir=${save_dir} \
#                 task=scene_diningroom \
#                 task.evaluator.weight_file=${save_dir}/model_98000  \
#                 evaluation.generate_result_json=true \
#                 evaluation.visual=true \
#                 task.test.batch_size=1 \
#                 task.dataset.path_to_bounds=demo/bounds.npz \
#                 ai2thor.path_to_result=/home/yandan/workspace/PhyScene/ai2thor/generate_bbox
