import argparse
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.dataset import Padding
from quantization.quantize import ptq_serving, qat_serving


class_info = {
    # foods
    'beef': 0, 'chicken': 1, 'chicken_feet': 2, 'chicken_ribs': 3, 'dry_snacks': 4,
    'dubu_kimchi': 5, 'ecliptic': 6, 'egg_roll': 7, 'fish_cake_soup': 8,
    'french_fries': 9, 'gopchang': 10, 'hwachae': 11, 'jjambbong': 12,
    'jjapageti': 13, 'korean_ramen': 14, 'lamb_skewers': 15, 'nacho': 16,
    'nagasaki': 17, 'pizza': 18, 'pork_belly': 19, 'pork_feet': 20,
    'raw_meat': 21, 'salmon': 22, 'sashimi': 23, 'shrimp_tempura': 24,
    
    # drinks
    'beer': 25, 'cass': 26, 'chamisul_fresh': 27, 'chamisul_origin': 28,
    'chum_churum': 29, 'highball': 30, 'hite': 31, 'jinro': 32, 'kelly': 33, 
    'kloud': 34, 'ob': 35, 'saero': 36, 'soju': 37, 'tera': 38,
}

class_info_rev = {v: k for k, v in class_info.items()}


transformation = transforms.Compose([
    Padding(fill=(0, 0, 0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_image(img_url: str):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = transformation(img)
    img = img.unsqueeze(dim=0)
    return img, img_url


def inference(src: torch.Tensor, model: nn.Module, threshold: int=0.5):
    model.eval()
    result_list = []
    with torch.no_grad():
        outputs = model(src)
        result = outputs[0].detach().numpy()
        indices = np.where(result > threshold)[0]
        for idx in indices:
            result_list.append(class_info_rev[int(idx)])
    return result_list


def get_args_parser():
    parser = argparse.ArgumentParser(description='Inference', add_help=False)
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name')
    parser.add_argument('--img_url', type=str, required=True,
                        help='input image')
    parser.add_argument('--weight', type=str, required=True,
                        help='a path of trained weight file')
    parser.add_argument('--num_classes', type=int, default=25,
                        help='the number of classes')
    parser.add_argument('--quantization', type=str, default='none', choices=['none', 'qat', 'ptq'],
                        help='load quantized model or float32 model')
    parser.add_argument('--measure_latency', action='store_true',
                        help='print latency time')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='a threshold for filtering outputs')

    return parser


def main(args):
    q = True if args.quantization != 'none' else False

    # load model
    if args.model_name == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=args.num_classes, pre_trained=False, quantize=q)
        
    elif args.model_name == 'resnet18':
        from models.resnet import resnet18
        model = resnet18(num_classes=args.num_classes, pre_trained=False, quantize=q)
        
    elif args.model_name == 'resnet50':
        from models.resnet import resnet50
        model = resnet50(num_classes=args.num_classes, pre_trained=False, quantize=q)
        
    else:
        raise ValueError(f'model name {args.model_name} does not exists.')
    
    
    # quantization
    if args.quantization == 'ptq':
        model = ptq_serving(model=model, weight=args.weight)

    elif args.quantization == 'qat':
        model = qat_serving(model=model, weight=args.weight)

    else: # 'none'
        pass

    img, _ = load_image(args.img_url)
    result = inference(img, model, args.threshold)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)