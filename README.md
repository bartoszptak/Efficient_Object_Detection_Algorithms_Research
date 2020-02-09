# Efficient Object Detection Algorithms Research
I would like to compare the models for real time object detection and their performance.
In the future I am going to modify the code so that all networks can use the cv2.dnn module.

# Models
* YOLOv3 [[paper](https://arxiv.org/pdf/1804.02767.pdf)][[code](https://pjreddie.com/darknet/yolo/)]
* EfficientDet [[paper](https://arxiv.org/pdf/1911.09070.pdf)][[code](https://github.com/xuannianz/EfficientDet)]
* (future) VoVNet [[paper](https://arxiv.org/pdf/1904.09730v1.pdf)][code]

# Devices
* Notebook i5-8265U + 12GB RAM + NVIDIA MX230
* PC Core i5-8400 + 16GB RAM + NVIDIA GTX1060 6GB
* NVIDIA Jetson TX2

# Installation guide
* [OpenCV for CPU guide](https://github.com/bartoszptak/Efficient_Object_Detection_Algorithms_Research/blob/master/INSTALLATION_GUIDE.md#opencv-for-cpu-guide)
* [OpenCV for GPU guide](https://github.com/bartoszptak/Efficient_Object_Detection_Algorithms_Research/blob/master/INSTALLATION_GUIDE.md#opencv-for-gpu-guide)

* download modified EfficientDet repos
```
git clone git@github.com:bartoszptak/EfficientDet.git
cd EfficientDet/
mv EfficientDet/* .
rm -r EfficientDet/ inference.py
```

* download models
```
python download_models.py
```

* (optionaly) download and prepare dataset for benchmark
```
mkdir data && cd data
wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
tar -vxf VOCtest_06-Nov-2007.tar
cd VOCdevkit/VOC2007/JPEGImages/
# make dataset smaller
rm 00{2..9}*.jpg

```

# Training results
| Model         | Size |    GFLOPS    | mAP | Train time |
|---------------|:----:|:------------:|:---:|:--------:|
| YOLOv3        |  512 |     99.42    |     | 1-05:32:00 |
| EffficientDet |  512 | 2.5<br>(from pdf) |  0.8870   |  23:12:18  |

# Benchmark results
### EffficientDet
| Device | Total FPS<br>batch=1 | Total FPS<br>batch=2 | Inference FPS<br>batch=1 | Inference FPS<br>batch=2 |
|----------------------|:---------------------:|:---------------------:|:-------------------------:|:-------------------------:|
| NVIDIA<br>MX230 | 5.67 | 6.49 | 12.63 | 14.13 |
| NVIDIA<br>GTX1060 | 15.07 | 17.77 | 34.43 | 42.05 |
| NVIDIA<br>Jetson TX2 |  |  |  |  |

### YOLOv3
| Device | Total FPS<br>batch=1 | Total FPS<br>batch=2 | Inference FPS<br>batch=1 | Inference FPS<br>batch=2 |
|----------------------|:---------------------:|:---------------------:|:-------------------------:|:-------------------------:|
| NVIDIA<br>MX230 | 5.22 | 5.30 | 7.40 | 7.81 |
| NVIDIA<br>GTX1060 | 11.36 | 11.43 | 27.99 | 29.88 |
| NVIDIA<br>Jetson TX2 |  |  |  |  |
