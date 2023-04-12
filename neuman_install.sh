pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install nvidia-pyindex
pip install nvidia-cublas
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html

# install igl via pip
cd support_libs
git clone https://github.com/libigl/libigl-python-bindings.git
cd libigl-python-bindings
pip install packaging
module load cmake/3.20.3
# pip install -e .
# might take more than 30 mins
sbatch -J build_igl -n 16 --mem-per-cpu=512MB -t 4:00:00 --wrap "pip install -e ."

cd ..
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d
# pip install -e .
