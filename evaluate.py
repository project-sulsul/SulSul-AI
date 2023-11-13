import os
import argparse
import time
from tqdm.auto import tqdm
from typing import *
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn

from utils.dataset import load_dataloader
from utils.plots import plot_results
from quantization.quantize import (
    converting_quantization, 
    ptq_serving, 
    qat_serving, 
    fuse_modules, 
    print_size_of_model,
)


classes = {
    0: '소고기', 1: '맥주', 2: '치킨', 3: '닭발', 4: '닭갈비',
    5: '마른안주', 6: '두부김치', 7: '황도', 8: '계란말이', 9: '어묵탕',
    10: '감자튀김', 11: '곱창', 12: '하이볼', 13: '화채', 14: '짬뽕',
    15: '짜파게티', 16: '라면', 17: '양꼬치', 18: '나초', 19: '나가사키 짬뽕',
    20: '피자', 21: '삼겹살', 22: '족발', 23: '육회', 24: '연어',
    25: '회', 26: '새우튀김', 27: '소주',
}

# score function
def score_fn(label, pred):
    # print confusion matrix
    print('### confusion matrix ###\n')
    print(confusion_matrix(label, pred))
    # print each score
    each_score = classification_report(
        label,
        pred,
        target_names=list(classes.values()),
    )
    print('\n\n### each score ###\n')
    print(each_score)


def test(
    test_loader,
    device,
    model: nn.Module,
    project_name: Optional[str]=None,
    plot_result: bool=False,
):
    if (project_name is None) and (not plot_result):
        raise ValueError('define project name')

    image_list, label_list, output_list = [], [], []
    
    start = time.time()

    model.eval()
    model = model.to(device)
    batch_acc = 0
    with torch.no_grad():
        for batch, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image_list.append(images)
            label_list.append(labels)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            output_index = torch.argmax(outputs, dim=1)

            output_list.append(output_index.cpu())

            acc = (output_index == labels).sum() / len(outputs)
            batch_acc += acc.item()
    
    print(f'{"="*20} Inference Time: {time.time()-start:.3f}s {"="*20}')    
    
    if project_name is not None and plot_result:
        plot_results(image_list, label_list, output_list, project_name)
    
    print(f'{"="*20} Test Average Accuracy {batch_acc/(batch+1)*100:.2f} {"="*20}')



def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Model', add_help=False)
    parser.add_argument('--data_path', type=str, required=True,
                        help='data directory for training')
    parser.add_argument('--subset', type=str, default='valid',
                        help='dataset subset')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name consisting of mobilenet, shufflenet, efficientnet, resnet18 and resnet50')
    parser.add_argument('--weight', type=str, required=True,
                        help='load trained model')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image resize size before applying cropping')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers in cpu')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch Size for training model')
    parser.add_argument('--num_classes', type=int, default=28,
                        help='class number of dataset')
    parser.add_argument('--project_name', type=str, default='prj',
                        help='create new folder named project name')
    parser.add_argument('--quantization', type=str, default='none', choices=['none', 'qat', 'ptq'],
                        help='evaluate the performance of quantized model or float32 model when setting none')
    parser.add_argument('--plot_result', action='store_true',
                        help='measure latency time')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                        help='set device for inference')
    return parser


def main(args):
    
    os.makedirs(f'./runs/test/{args.project_name}', exist_ok=True)
    
    test_loader = load_dataloader(
        path=args.data_path,
        img_size=args.img_size,
        subset=args.subset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    # setting device
    device = torch.device(args.device)

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
        model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    test(
        test_loader,
        device=device,
        model=model,
        project_name=args.project_name,
        plot_result=args.plot_result,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)