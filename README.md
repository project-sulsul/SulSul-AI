## Building a Deep Learning Model to classify alcohols and its snacks  

### Overview
**Features**
- [`Multi-Label Classification`](https://github.com/project-sulsul/SulSul-AI/blob/56bc18e75e52dcc455a434269ed7544db6551206/train.py#L142)

    <img src = "https://i.ytimg.com/vi/Epx2V3Kd3dE/maxresdefault.jpg" width=600>

- [`Image Padding`](https://github.com/project-sulsul/SulSul-AI/blob/56bc18e75e52dcc455a434269ed7544db6551206/utils/dataset.py#L46)
- [`Quantization Aware Training`](https://github.com/project-sulsul/SulSul-AI/blob/56bc18e75e52dcc455a434269ed7544db6551206/quantization/quantize.py#L18)

### Training and Test
- training
    ```
    python3 train.py --data_path 'your/dataset/directory' --name 'prj' --model 'resnet18' --pretrained --num_classes 39 --batch_size 32 --lr_scheduling --check_point
    ```

- test
    ```
    python3 evaluate.py --data_path 'your/dataset/directory' --quantization 'qat' --model_name 'resnet18' --weight 'your/best/weight/directory' --num_classes 39
    ```
- inference
    ```
    python3 inference.py --img_url 'image/url' --model_name 'resnet18' --quantization 'qat' --weight 'your/best/weight/directory' --num_classes 39 --threshold 0.5
    ```