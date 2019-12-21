1. [OpenCV for CPU guide](#opencv-for-cpu-guide)
2. [OpenCV for GPU guide](#opencv-for-gpu-guide)
3. [OpenVino for Raspberry guide](#openvino-for-raspberry-guide)

# OpenCV for CPU guide
simply:
```
pip install opencv-python
```

# OpenCV for GPU guide
**You must have CUDA capability at least 5.3!**

Step 0: Install NVIDIA drivers, cuda-toolkit and NVCC
```
https://www.tensorflow.org/install/gpu
```
Step 1
```
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
```
Step 2
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
Step 3
```
make -j4
sudo make install
sudo ldconfig
```
Step 4
```
ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so ~/miniconda3/lib/python3.6/site-packages/cv2.so
```

# OpenVino for Raspberry guide
## TODO
