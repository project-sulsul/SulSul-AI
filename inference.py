import argparse
from PIL import Image
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.dataset import Padding
from quantization.quantize import ptq_serving, qat_serving


classes = {
    0: '소고기', 1: '맥주', 2: '치킨', 3: '닭발', 4: '닭갈비',
    5: '마른안주', 6: '두부김치', 7: '황도', 8: '계란말이', 9: '어묵탕',
    10: '감자튀김', 11: '곱창', 12: '하이볼', 13: '화채', 14: '짬뽕',
    15: '짜파게티', 16: '라면', 17: '양꼬치', 18: '나초', 19: '나가사키 짬뽕',
    20: '피자', 21: '삼겹살', 22: '족발', 23: '육회', 24: '연어',
    25: '회', 26: '새우튀김', 27: '소주',
}


transformation = transforms.Compose([
    Padding(fill=(0, 0, 0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_image(src: str):    
    img = Image.open(src).convert('RGB')
    img = transformation(img)
    img = img.unsqueeze(dim=0)
    return img, src


def inference(src: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        outputs = model(src)
        # outputs = F.softmax(outputs)
        result = classes[torch.argmax(outputs, dim=1).item()]
    return result


def get_args_parser():
    parser = argparse.ArgumentParser(description='Inference', add_help=False)
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['shufflenet', 'resnet18', 'resnet50'],
                        help='model name')
    parser.add_argument('--src', type=str, required=True,
                        help='input image')
    parser.add_argument('--weight', type=str, required=True,
                        help='a path of trained weight file')
    parser.add_argument('--quantization', type=str, default='none', 
                        choices=['none', 'qat', 'ptq'],
                        help='load quantized model or float32 model')
    parser.add_argument('--num_classes', type=int, default=28,
                        help='the number of classes')
    return parser


def main(args):
    q = True if args.quantization is not 'none' else False

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

    img, _ = load_image(args.src)
    result = inference(img, model)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)