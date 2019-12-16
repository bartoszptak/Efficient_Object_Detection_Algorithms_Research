# Efficient Object Detection Research
I would like to compare the models for real time object detection and their performance. I want to use cv2.dnn module and test the following models:
* YOLOv3 [[paper](https://arxiv.org/pdf/1804.02767.pdf)][[code](https://pjreddie.com/darknet/yolo/)]
* EfficientDet [[paper](https://arxiv.org/pdf/1911.09070.pdf)][[code-tfkeras](https://github.com/xuannianz/EfficientDet)]

on selected devices:
* Notebook i5-5200U + 8GB RAM + NVIDIA GT940m
* Notebook i5-8265U + 12GB RAM + NVIDIA MX230
* PC Core i5-8400 + 16GB RAM + NVIDIA GTX1060 6GB
* Nvidia Jetson TX2
* Raspberry Pi 4 B + Movidius neural compute stick
* Raspberry Pi 4 B + Movidius MV224

|   Model name   | Image shape | COCO mAP (from paper) | FPS (device) |
|:--------------:|:-----------:|:---------------------:|:------------:|
| YOLOv3-320     |   320x320   |          51.5         |              |
| YOLOv3-416     |   416x416   |          55.3         |              |
| YOLOv3-608     |   608x608   |          57.9         |              |
| EfficientDet-0 |   512x512   |          32.4         |              |
| EfficientDet-1 |   640x640   |          38.3         |              |
| EfficientDet-2 |   768x768   |          41.1         |              |
| EfficientDet-3 |   896x896   |          44.3         |              |
