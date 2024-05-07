save_dir=$1   #data/preprocessed_data

python scripts/preprocess/preprocess_data.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            output_directory=${save_dir}/BedRoom \
            task=scene_bedroom

python scripts/preprocess/preprocess_data.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            output_directory=${save_dir}/LivingRoom \
            task=scene_livingroom

python scripts/preprocess/preprocess_data.py \
            hydra/job_logging=none hydra/hydra_logging=none \
            output_directory=${save_dir}/DiningRoom \
            task=scene_diningroom

#make room_mask folder
basepath=$(pwd)
maskdir=data/room_mask
mkdir $maskdir
cd $maskdir
for roomname in `ls ${basepath}/${save_dir}/BedRoom`; do 
    ln -s ${basepath}/${save_dir}/BedRoom/${roomname} ${roomname}
done

for roomname in `ls ${basepath}/${save_dir}/LivingRoom`; do 
    ln -s ${basepath}/${save_dir}/LivingRoom/${roomname} ${roomname}
done

for roomname in `ls ${basepath}/${save_dir}/DiningRoom`; do 
    ln -s ${basepath}/${save_dir}/DiningRoom/${roomname} ${roomname}
done