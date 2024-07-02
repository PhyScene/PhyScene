save_dir=$1

python scripts/eval/calc_ckl.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${save_dir} \
                task=scene_bedroom \
                task.network.room_mask_condition=true \
                continue_epoch=0 \
                task.evaluator.weight_file=${save_dir}/model_58000 \
                evaluation.generate_result_json=true \
                evaluation.jsonname="bedroom.json" \
                evaluation.overlap_type="mesh" \
                evaluation.visual=false \
                evaluation.render2img=false \
                evaluation.without_floor=false \
                evaluation.gapartnet=false \
                evaluation.render_save_path="result_render/livingroom_w_guide" \
                task.evaluator.n_synthesized_scenes=1000 

                

               
