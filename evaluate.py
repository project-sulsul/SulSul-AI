import os
import argparse
import time
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
from typing import *

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


# score function
def score_fn(label, pred):
    # print confusion matrix
    print('### confusion matrix ###\n')
    print(confusion_matrix(label, pred))
    # print each score
    each_score = classification_report(
        label,
        pred,
        target_names=list(class_info.values()))
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
            label_list.append(labels.tolist())

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs > 0.5
            acc = (outputs == labels).float().mean()

            batch_acc += acc.item()
    
    print(f'{"="*20} Inference Time: {time.time()-start:.3f}s {"="*20}')    
    print(f'{"="*20} Test Average Accuracy {batch_acc/(batch+1)*100:.2f} {"="*20}')
    score_fn(sum(label_list, []), sum(output_list, []))


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluating Model', add_help=False)
    parser.add_argument('--data_path', type=str, required=True,
                        help='data directory for training')
    parser.add_argument('--subset', type=str, default='valid',
                        help='dataset subset')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name consisting of shufflenet, resnet18, resnet34 and resnet50')
    parser.add_argument('--weight', type=str, required=True,
                        help='load trained model')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image resize size before applying cropping')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers in cpu')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch Size for training model')
    parser.add_argument('--num_classes', type=int, default=39,
                        help='class number of dataset')
    parser.add_argument('--project_name', type=str, default='prj',
                        help='create new folder named project name')
    parser.add_argument('--quantization', type=str, default='none', choices=['none', 'qat', 'ptq'],
                        help='evaluate the performance of quantized model or float32 model when setting none')
    parser.add_argument('--plot_result', action='store_true',
                        help='save the plotting result')
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
        shuffle=True,
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

    elif args.model_name == 'resnet34':
        from models.resnet import resnet34
        model = resnet34(num_classes=args.num_classes, pre_trained=False, quantize=q)

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
        model.load_state_dict(torch.load(args.weight))
    
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