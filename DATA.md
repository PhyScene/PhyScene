## Prepare Data                                                 
### Download Dataset
🏠 Scene: We train and evaluate the scene layout based on **3D FRONT** dataset. 

🪑 Rigid Object: We then utilize **3D FUTURE** as the rigid objects in the generated scene.

Download 3D FRONT and 3D FUTURE dataset with the instruction in [this website](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset).

💻 Articulated object: We utilize articulated objects in **GAPartNet** dataset for the generated scene.

Download GAPartNet dataset following the instructions in [this website](https://pku-epic.github.io/GAPartNet/).

### Preprocess data
#### Pickle scene & object dataset
```
sh run/pickle_3dfuture_dataset.sh save_dir 
# for example:
# sh run/pickle_3dfuture_dataset.sh data/pickled_data  
```
You will get (1) pickled scene dataset in ```PATH_TO_SCENES``` defined in the config file, and (2) pickled object dataset in ```save_dir/threed_future_model_roomtype.pkl```.

#### Pickle mesh info of each CAD model in both 3D FUTURE and GAPartNet
```
sh run/pickle_pcd.sh save_dir
# for example:
# sh run/pickle_pcd.sh data/pickled_data
```
You will get preprocessed pointcloud, saved as ```.npz``` and ```.ply```, for each object in ```3D FUTURE``` and ```GAPartNet```.

### Train Autoencoder, and generate latent feature
```
sh run/objautoencoder.sh save_dir experiment_tag 
# for example
# sh run/objautoencoder.sh autoencoder_output debug
```
You will get geometric latent feature, saved as ```raw_model_norm_pc_lat32.npz``` for each object.

#### Preprocess scene dataset of 3D FRONT 
```
sh run/preprocess_data.sh save_dir
# for example:
# sh run/preprocess_data.sh data/preprocessed_data
```
You will get room info and rendered image in ```save_dir```.
Here we set background color as gray for better visualization.
