Dataset downloaded 10.5.2025 at 11:20 am
https://www.kaggle.com/datasets/moltean/fruits?resource=download-directory

Installing pytorch backend:
mkdir -p ~/opt
cd ~/opt
choose the correct version from https://pytorch.org/get-started/locally/
in my case 2.6.0, linux, libtorch, C++/Java, Cuda 12.6
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu121.zip
it can be downloaded with wget
unzip the file
and set the environment variables:

- export LIBTORCH=$HOME/opt/libtorch
- export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
- source ~/.bashrc
