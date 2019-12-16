# Efficient Object Detection Research
I would like to compare the models for real time object detection and their performance. I want to use cv2.dnn module and test the following models:
* YOLOv3 [[paper](https://arxiv.org/pdf/1804.02767.pdf)][[code](https://pjreddie.com/darknet/yolo/)]
* EfficientDet [[paper](https://arxiv.org/pdf/1911.09070.pdf)][[code](https://github.com/xuannianz/EfficientDet)]

on selected devices:
* Notebook i5-5200U + 8GB RAM + NVIDIA GT940m
* Notebook i5-8265U + 12GB RAM + NVIDIA MX230
* PC Core i5-8400 + 16GB RAM + NVIDIA GTX1060 6GB
* Nvidia Jetson TX2
* Raspberry Pi 4 B + Movidius neural compute stick
* Raspberry Pi 4 B + Movidius MV224

| Model<br>name | Image<br>shape | COCO mAP<br>(from paper) | FPS<br>(device) |
|:--------------:|:--------------:|:------------------------:|:---------------:|
| YOLOv3-320 | 320x320 | 51.5 |  |
| YOLOv3-416 | 416x416 | 55.3 |  |
| YOLOv3-608 | 608x608 | 57.9 |  |
| EfficientDet-0 | 512x512 | 32.4 |  |
| EfficientDet-1 | 640x640 | 38.3 |  |
| EfficientDet-2 | 768x768 | 41.1 |  |
| EfficientDet-3 | 896x896 | 44.3 |  |

# TODO
- [x] Create a script to check the FPS on the selected video
- [ ] Create a script to calculate FLOPS
- [ ] Creation of a benchmark script

- [x] Starting YOLOv3 with cv2.dnn
- [ ] Starting EfficientDet with cv2.dnn
- [ ] EfficientDet training in different sizes

- [ ] Adding support for the GPU (Nvidia GPUs)
- [ ] Adding support for the RAID (Raspberry & Movidious)
- [ ] Adding support for the GPU fp16 (Nvidia Jetson)
