
#### Create a Conda Environment
```
conda create --name physcene python=3.8.16
conda activate physcene
```
#### Install Python Packages
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### Compile Rotated_IoU Extension:   
```
# code is from https://github.com/lilanxiao/Rotated_IoU
cd models/loss/cuda_op
python setup.py install
```

#### Install ChamferDistancePytorch
```
cd ChamferDistancePytorch/chamfer3D
python setup.py install
```

#### Install Kaolin:
```
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html
```

#### Modify simple_3dviz
The assets have materials with informal uv map, which will cause error when loading mesh in simple_3dviz.
(1) You may need to modify a command line in  ```~/anaconda3/envs/physcene/lib/python3.8/site-packages/simple_3dviz/io/multi_mesh.py``` at ```line 153``` from:
```
except IndexError:
    face_uv = np.zeros((len(face_vertices), 2))
```
to
```
except:
    face_uv = np.zeros((len(face_vertices), 2))
```

(2) You may also need to modify a command line in  ```~/anaconda3/envs/physcene/lib/python3.8/site-packages/simple_3dviz/io/material.py``` at ```line 151``` from:
```
elif l.startswith("illum"):
    material["mode"] = {
        "0" : "constant",
        "1" : "diffuse",
        "2" : "specular"
    }[l.split()[1]]
```
to
```
elif l.startswith("illum"):
    modelst = {"0" : "constant",
                "1" : "diffuse",
                "2" : "specular"
                }
    if l.split()[1] in modelst:
        material["mode"] = modelst[l.split()[1]]
    else:
        material["mode"] = "specular"
```
