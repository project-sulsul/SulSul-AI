## Building a Deep Learning Model to classify and recommend alcohols and its snacks  

### Overview

### Training and Test
- training
```
python3 train.py --data_path '/dataset/path/' --name 'exp' --model 'model_name' --pretrained --img_size 224 --lr 0.0005 --num_workers 8 --batch_size 4 --epochs 100 --optimizer 'adam' --lr_scheduling --check_point
```
- test
```
python3 evaluate.py --data_path 'the/directory/of/dataset' --model resnet18 --weight 'the/path/of/trained/weight/file' --img_size 224 --num_workers 8 --batch_size 4 --num_classes 18
```
- inference
```
python3 inference.py --src 'the/directory/of/image' --model_name resnet18 --weight 'the/path/of/trained/weight/file' --quantization --measure_latency
```

### Dataset