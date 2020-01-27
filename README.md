# Efficient Object Detection Algorithms Research
I would like to compare the models for real time object detection and their performance.  I want to (future: use cv2.dnn module and) est the following models:
* YOLOv3 [[paper](https://arxiv.org/pdf/1804.02767.pdf)][[code](https://pjreddie.com/darknet/yolo/)]
* EfficientDet [[paper](https://arxiv.org/pdf/1911.09070.pdf)][code]
* (future) VoVNet [[paper](https://arxiv.org/pdf/1904.09730v1.pdf)][code]

on selected devices:
* Notebook i5-8265U + 12GB RAM + NVIDIA MX230
* PC Core i5-8400 + 16GB RAM + NVIDIA GTX1060 6GB
* NVIDIA Jetson TX2
* (future) NVIDIA Xavier
* Raspberry Pi 4 B + Movidius NCS
* (future) Raspberry Pi 4 B + Movidius NCS2

# Installation guide
[here](https://github.com/bartoszptak/Efficient_Object_Detection_Algorithms_Research/blob/master/INSTALLATION_GUIDE.md)

# Results
All results will be posted [here](https://www.overleaf.com/read/xkmsnjnfxwrg).

# TODO
- [x] Create a script to check the FPS on the selected video
- [x] Create a script to calculate FLOPS
- [ ] Creation of a benchmark script

- [x] Starting YOLOv3 with cv2.dnn
- [ ] Starting EfficientDet with cv2.dnn
- [ ] Starting VoVNet with cv2.dnn
- [ ] EfficientDet training in different sizes

- [x] Adding support for the GPU (Nvidia CUDA GPUs)
- [ ] Adding support for the RAID (Raspberry & Movidious)
- [ ] Adding support for the GPU fp16 (Nvidia Jetson)
