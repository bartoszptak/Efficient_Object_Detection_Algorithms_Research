#!/bin/bash

for engine in gpu cpu
do
    for batch in 2 1
    do
        echo yolo,416,$batch,$engine
        python main.py --mode bench --engine $engine --batch-size $batch --model yolo --size 416
        echo yolo,608,$batch,$engine
        python main.py --mode bench --engine $engine --batch-size $batch --model yolo --size 608

        echo eff,512,$batch,$engine
        python main.py --mode bench --engine $engine --batch-size $batch --model eff --size 512
        echo eff,640,$batch,$engine
        python main.py --mode bench --engine $engine --batch-size $batch --model eff --size 640
    done
done