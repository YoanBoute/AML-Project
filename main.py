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
from models.resnet import BaseResNet18, ASHResNet18, asm_hook_generator, asm_hook_generator_no_binarization, asm_hook_generator_top_k

from globals import CONFIG

from copy import deepcopy

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

    # Write results in csv file for later use
    results_file = os.path.join(CONFIG.save_dir, 'results.csv')
    if not os.path.exists(results_file) :
        with open(results_file, 'a') as res :
            res.write('Epoch, Ratio of 1s (if random), Location of ASM (if needed), K (if needed), Accuracy, Loss\n')
    ratio_1 = CONFIG.experiment_args['ratio_1'] if CONFIG.experiment_args.get('ratio_1') is not None else ''
    layer_ASM = CONFIG.experiment_args['layers_asm'] if CONFIG.experiment_args.get('layers_asm') is not None else ''
    K = CONFIG.experiment_args['K'] if CONFIG.experiment_args.get('K') is not None else ''
    # Estimation of the current epoch by counting the number of lines of the file
    with open(results_file, 'r') as readable_file :
        content = readable_file.read()
    current_epoch = content.count('\n')
    with open(results_file, 'a') as res :
        res.write(f'{current_epoch}, {ratio_1}, {layer_ASM}, {K}, {accuracy}, {loss} \n')


def random_activation_map_generator(size, ratio_1 : float, binarized = True) :
    """Generate a random activation map of given size, with as much ones (or positive elements) as indicated in the arg ratio_1 --> For experiment 2"""
    number_values = np.prod(list(size))
    # The total number of ones to put in the map is rounded to the upper int to make sure we never have a map full of 0, which would break the network
    number_ones = int(np.ceil(number_values * ratio_1))

    if binarized :
        # Generate a random binary map with the correct number of ones and the correct size
        permuted_indices = torch.randperm(number_values)[:number_ones]
        map = torch.zeros(number_values, dtype=torch.float32)
        map[permuted_indices] = 1
        map = map.reshape(size)

    else :
        # Generate a random map with only positive values
        map = torch.rand(size)
        deactivated_indices = torch.randperm(number_values)[:(number_values - number_ones)]
        map[deactivated_indices] = 0
        map = map.reshape(size)

    return map.to(CONFIG.device)


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

                if CONFIG.experiment in ['baseline', 'random', 'random_no_binarization', 'random_top_k']:
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)

                elif CONFIG.experiment in ['DA', 'DA_no_binarization', 'DA_top_k'] :
                    src_x, src_y, targ_x = batch
                    src_x, src_y, targ_x = src_x.to(CONFIG.device), src_y.to(CONFIG.device), targ_x.to(CONFIG.device)

                    if CONFIG.experiment_args.get('layers_asm') is None :
                        raise BaseException("Error : No layer was given to put a hook on")
                    layers = CONFIG.experiment_args['layers_asm']
                    if layers.startswith('[') :
                        layers = eval(layers)
                    else :
                        layers = [layers]

                    '''Specific case : If we indicate allConv in the layers, then an ASM hook has to be put after each convolution of the network'''
                    if layers[0] == 'allConv' :
                        layers.remove('allConv')
                        for name, mod in model.resnet.named_modules() :
                            if isinstance(mod, torch.nn.modules.conv.Conv2d) :
                                layers.append(name)

                    # Create a feature extractor to get the output of the specified layers
                    feature_extractor = create_feature_extractor(model.resnet, return_nodes=layers)
                    # Pass the target sample through the feature extractor to get the activation maps
                    layer_outputs_target = feature_extractor(targ_x)
                    new_model = deepcopy(model)

                    if CONFIG.experiment in ('DA_top_k'):
                        if CONFIG.experiment_args.get('K') is None:
                            raise BaseException("Error : K hyperparameter not set")
                        else:
                            K = CONFIG.experiment_args['K']
                    for layer in layers :
                        activation_map = layer_outputs_target[layer]
                        # Put the hook corresponding to each activation map in the model
                        if CONFIG.experiment in ('DA'):
                            new_model.put_asm_after_layer(layer, asm_hook_generator(activation_map))
                        elif CONFIG.experiment in ('DA_no_binarization'):
                            new_model.put_asm_after_layer(layer, asm_hook_generator_no_binarization(activation_map))
                        elif CONFIG.experiment in ('DA_top_k'):
                            new_model.put_asm_after_layer(layer, asm_hook_generator_top_k(activation_map, K))

                    loss = F.cross_entropy(model(src_x), src_y)

                    # Remove the previous hooks to avoid overlapping hooks
                    # for layer in layers :
                    #     model.remove_asm_after_layer(layer)

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

    elif CONFIG.experiment in ['random', 'random_no_binarization', 'random_top_k'] :
        model = ASHResNet18()

        if CONFIG.experiment_args.get('ratio_1') is None :
            raise BaseException("Error : The ratio of 1 in random activation maps has to be given")
        ratio_1 = CONFIG.experiment_args['ratio_1']

        if CONFIG.experiment in ('random_top_k'):
            if CONFIG.experiment_args.get('K') is None:
                raise BaseException("Error : K hyperparameter not set")
            else:
                K = CONFIG.experiment_args['K']

        # Create a feature extractor to get the output of all layers
        layers = []
        for layer_name, layer in model.resnet.named_modules() :
            if layer_name != '' :
                layers.append(layer_name)
        feature_extractor = create_feature_extractor(model.resnet, return_nodes=layers)
        # Pass a random sample through the feature extractor to get the activation maps sizes
        d = data['train'].dataset[0][0]
        layer_outputs = feature_extractor(torch.tensor([d.tolist()]))

        for layer in layers :
            if layer.endswith('.1') :
                layer = layer.split('.1')[0]
            activation_map_size = layer_outputs[layer].shape
            # Put the hook corresponding to each activation map in the model
            if CONFIG.experiment in ('random'):
                # Generate randomly the activation map
                M = random_activation_map_generator(activation_map_size, ratio_1, binarized=True)
                model.put_asm_after_layer(layer, asm_hook_generator(M))
            elif CONFIG.experiment in ('random_no_binarization'):
                M = random_activation_map_generator(activation_map_size, ratio_1, binarized=False)
                model.put_asm_after_layer(layer, asm_hook_generator_no_binarization(M))
            elif CONFIG.experiment in ('random_top_k'):
                M = random_activation_map_generator(activation_map_size, ratio_1, binarized=True)
                model.put_asm_after_layer(layer, asm_hook_generator_top_k(M, K))

    elif CONFIG.experiment in ['DA', 'DA_no_binarization', 'DA_top_k'] :
        model = ASHResNet18()

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
    torch.autograd.set_detect_anomaly(True)

    main()
