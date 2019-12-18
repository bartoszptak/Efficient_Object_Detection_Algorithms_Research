# Efficient Object Detection Algorithms Research
I would like to compare the models for real time object detection and their performance. I want to use cv2.dnn module and test the following models:
* YOLOv3 [[paper](https://arxiv.org/pdf/1804.02767.pdf)][[code](https://pjreddie.com/darknet/yolo/)]
* EfficientDet [[paper](https://arxiv.org/pdf/1911.09070.pdf)][[code](https://github.com/xuannianz/EfficientDet)]
* VoVNet [[paper](https://arxiv.org/pdf/1904.09730v1.pdf)][code]

on selected devices:
* Notebook i5-5200U + 8GB RAM + NVIDIA GT940m
* Notebook i5-8265U + 12GB RAM + NVIDIA MX230
* PC Core i5-8400 + 16GB RAM + NVIDIA GTX1060 6GB
* Nvidia Jetson TX2
* Raspberry Pi 4 B + Movidius neural compute stick
* Raspberry Pi 4 B + Movidius MV224


# TODO
- [x] Create a script to check the FPS on the selected video
- [x] Create a script to calculate FLOPS
- [ ] Creation of a benchmark script

- [x] Starting YOLOv3 with cv2.dnn
- [ ] Starting EfficientDet with cv2.dnn
- [ ] EfficientDet training in different sizes

- [ ] Adding support for the GPU (Nvidia GPUs)
- [ ] Adding support for the RAID (Raspberry & Movidious)
- [ ] Adding support for the GPU fp16 (Nvidia Jetson)

# Actual project tree
```
.
├── bench_utils.py
├── LICENSE
├── main.py
├── models
│   └── YOLOv3
│       ├── YOLOv3.cfg
│       └── YOLOv3.weights
├── README.md
└── video
    └── mosque.mp4
```
