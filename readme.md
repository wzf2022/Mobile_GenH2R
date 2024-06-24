## installation
``` bash
conda create -n genh2r python=3.10
conda activate genh2r
# pip install gym
pip install pybullet
pip install numpy
# pip install yacs
pip install omegaconf
pip install scipy
pip install ipdb
pip install psutil
pip install lxml
pip install ray
pip install transforms3d
pip install easydict
pip install opencv-python
pip install imageio[ffmpeg]
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tqdm
pip install mayavi
pip install PyQt5
pip install open3d
pip install h5py
# pip install pymeshlab
# conda install libffi==3.3 # https://github.com/conda/conda/issues/12287#issuecomment-1491580210
pip install bezier
pip install wandb
```
### third_party
``` bash
git submodule init
git submodule update
```
#### PyKDL
``` bash
cd env/third_party/orocos_kinematics_dynamics
git submodule update --init
sudo apt-get update
sudo apt-get install libeigen3-dev libcppunit-dev
# sudo apt-get install doxygen graphviz
# sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
# C++ library compilation
cd orocos_kdl
mkdir build
cd build
cmake .. -DENABLE_TESTS:BOOL=ON
make
sudo make install
make check
# make docs
# python bindings compilation
cd ../../python_orocos_kdl
mkdir build
cd build
ROS_PYTHON_VERSION=3.10 cmake ..
make
sudo make install
# sudo ldconfig
cp PyKDL.so $CONDA_PREFIX/lib/python3.10/site-packages/
## test
python3 ../tests/PyKDLtest.py
## docs
# sudo apt-get install python3-sphinx
# sphinx-build ../doc docs
```
#### OMG Planner
``` bash
cd layers
python setup.py install
```
#### PointNet++
``` bash
cd third_party/Pointnet2_PyTorch
pip install pointnet2_ops_lib/.
# to test: import pointnet2_ops
```
#### SDFGen
``` bash
cd env/third_party/SDFGen
mkdir build
cd build
cmake ..
make
```
#### v-hacd
``` bash
cd env/third_party/v-hacd/app
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
#### CoACD
``` bash
cd env/third_party/CoACD
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make main -j
```
#### mano-pybullet
``` bash
cd env/third_party/mano_pybullet
pip install -e .
```

## repo structure
``` bash
tree -I "data|third_party|__pycache__|log"
```
```
.
├── config.py
├── env
│   ├── benchmark.py
│   ├── bodies_for_visualization.py
│   ├── body.py
│   ├── camera.py
│   ├── contact.py
│   ├── handover_env.py
│   ├── hand.py
│   ├── objects.py
│   ├── panda.py
│   ├── table.py
│   └── utils.py
├── evaluate.py
├── models
│   ├── encoders
│   │   └── pointnet2.py
│   ├── loss.py
│   ├── policy_network.py
│   └── utils.py
├── policies
│   ├── base_policy.py
│   ├── cartesian_planner.md
│   ├── cartesian_planner.py
│   ├── offline.py
│   ├── omg_planner.md
│   ├── omg_planner_original.py
│   └── pointnet2_policy.py
├── policy_runner.py
├── readme.md
├── scripts
│   └── gen_demo_for_different_seeds.sh
├── tools
│   └── process_dexycb.py
├── train
│   ├── data_provider.py
│   ├── train_imitation.md
│   └── train_imitation.py
└── utils
    ├── file_system.py
    ├── robot_kinematics.py
    ├── robotPose
    │   ├── _init_paths.py
    │   ├── __init__.py
    │   ├── kdl_parser.py
    │   ├── robot_pykdl.py
    │   └── urdf_parser_py
    │       ├── __init__.py
    │       ├── sdf.py
    │       ├── urdf.py
    │       └── xml_reflection
    │           ├── basics.py
    │           ├── core.py
    │           └── __init__.py
    └── transform.py
```
## data
### dexycb
Download `dex-ycb-cache-20220323.tar.gz` and `ycb_grasps.tar.gz` to `env/data/tmp`, then run
``` bash
cd env/data/tmp
tar -xvf dex-ycb-cache-20220323.tar.gz
mkdir ycb_grasps
tar -xvf ycb_grasps.tar.gz -C ycb_grasps
cd ../../..
python -m env.tools.process_dexycb
```
The processed 1000 scenes will be in `data/scene/00/00`, from `data/scene/00/00/00/00000000.npz` to `data/scene/00/00/09/00000999.npz`.
### acronym
#### objects
Download `models-OBJ.zip`, `models-textures.zip` from ShapeNet(TODO) and `acronym.tar.gz` from ACRONYM(TODO) to `env/data/tmp`, then run
``` bash
cd env/data/tmp
unzip models-OBJ.zip
unzip models-textures.zip
mv textures/* models/
tar -xvf acronym.tar.gz
cd ../../..
python -m env.tools.preprocess_acronym
python -m env.tools.process_acronym
```
#### hand dex grasps
create a separate conda environment
``` bash
conda create -n dexgraspnet python=3.9
conda activate dexgraspnet
pip install numpy==1.23.1
conda install pytorch pytorch=1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
# check torch.cuda.is_available()
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install trimesh
pip install plotly
pip install chumpy
pip install transforms3d
pip install ray
pip install opencv-python
pip install rtree
pip install regex
pip install open3d
pip install omegaconf
pip install networkx

cd env/third_party/DexGraspNet/third_party/manopth
pip install .
cd ../TorchSDF
bash install.sh

# pip install matplotlib
```

### assets
#### hand (in a separate environment with numpy=1.23.0)
Download `mano_v1_2` from [MANO website](http://mano.is.tue.mpg.de/) to `env/data/tmp`, then run
``` bash
cd env/data/tmp
unzip mano_v1_2.zip
cd ../../..
python -m env.tools.process_mano_hand
```

#### objects

