1. [OpenCV for CPU guide](#opencv-for-cpu-guide)
2. [OpenCV for GPU guide](#opencv-for-gpu-guide)
3. [OpenVino for Raspberry guide](#openvino-for-raspberry-guide)

# OpenCV for CPU guide
simply:
```
pip install opencv-python
```

# OpenCV for GPU guide
```
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
```

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
	  -D PYTHON3_EXECUTABLE=$HOME/miniconda3/bin/python \
    -D OPENCV_EXTRA_MODULES_PATH=/$HOME/opencv_contrib/modules \
    -D CUDA_ARCH_BIN=6.1 \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON ..
```

```
make -j4
sudo make install
sudo ldconfig
```

# OpenVino for Raspberry guide
## TODO
