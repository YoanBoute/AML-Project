import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18, ASHResNet18, asm_hook_generator

from globals import CONFIG

@torch.no_grad()
def evaluate(model, data):
    model.eval()
    
    acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]
    for x, y in tqdm(data):
        with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)
    
    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f'Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


'''Generate a random activation map of given size, with as much ones (or positive elements) as indicated in the arg ratio_1 --> For experiment 2'''
def random_activation_map_generator(size : list[int], ratio_1 : float, binarized = True) :
    number_values = size[0] * size[1]
    # The total number of ones to put in the map is rounded to the upper int to make sure we never have a map full of 0, which would break the network
    number_ones = int(np.ceil(number_values * ratio_1))
    map = torch.zeros(size)

    if binarized :
        for _ in range(number_ones) :
            # We want to be sure to have the given ratio of 1, so we have to sample an element from the map until we find a 0 to update it
            while True :
                x = np.random.randint(0, size[0])
                y = np.random.randint(0, size[1])

                if map[x,y] == 0 :
                    map[x,y] = 1
                    break
    
    else :
        for _ in range(number_ones) :
            while True :
                x = np.random.randint(0, size[0])
                y = np.random.randint(0, size[1])

                if map[x,y] == 0 :
                    map[x,y] = random.random()
                    break
        
        # Every value that is not yet positive becomes a random negative number
        map[map == 0] = torch.tensor((- np.random.random(size=np.count_nonzero(map == 0))).tolist())

    return map


def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
            # Compute loss
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):

                if CONFIG.experiment in ['baseline', 'random']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                elif CONFIG.experiment in ['DA'] :
                    src_x, src_y, targ_x = batch 
                    src_x, src_y, targ_x = src_x.to(CONFIG.device), src_y.to(CONFIG.device), targ_x.to(CONFIG.device)

                    layers = CONFIG.experiment_args['layers_asm']
                    if type(layers) != list :
                        layers = [layers]
                    
                    # Create a feature extractor to get the output of the specified layers
                    feature_extractor = create_feature_extractor(model.resnet, return_nodes=layers)
                    # Pass the target sample through the feature extractor to get the activation maps
                    layer_outputs_target = feature_extractor(targ_x)
                    for layer in layers :
                        activation_map = layer_outputs_target[layer]
                        # Put the hook corresponding to each activation map in the model
                        model.put_asm_after_layer(layer, asm_hook_generator(activation_map))
                    
                    loss = F.cross_entropy(model(src_x), src_y)

                    # Remove the previous hooks to avoid overlapping hooks
                    for layer in layers :
                        model.remove_asm_after_layer(layer)


            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()

            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()
        
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))


def main():
    
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()

    ######################################################
    elif CONFIG.experiment in ['random', 'DA'] :
        model = ASHResNet18()
    ######################################################
    
    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
    else:
        evaluate(model, data['test'])
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)

    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, 'log.txt'), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
