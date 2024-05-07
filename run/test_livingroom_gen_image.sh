save_dir=$1

python calc_ckl.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${save_dir} \
                task=scene_livingroom \
                task.network.room_mask_condition=true \
                task.evaluator.weight_file=${save_dir}/model_134000 \
                evaluation.generate_result_json=false \
                evaluation.jsonname="livingroom.json" \
                evaluation.overlap_type="rotated_bbox" \
                evaluation.visual=true \
                evaluation.render2img=true \
                evaluation.save_walkable_map=true \
                evaluation.without_floor=false \
                evaluation.gapartnet=true \
                evaluation.render_save_path="result_render/livingroom_wo_guide" \
                # task.evaluator.n_synthesized_scenes=100 
                # optimizer=collision 

