# Adapted from https://github.com/thuml/Autoformer/blob/main/utils/tools.py

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.config import Config

plt.switch_backend('agg')


def robust_loss(lb, ub, y):
    """
    Robust loss function
    args:
        lb: lower bound
        ub: upper bound
        y: target value
    return: robust loss
    """

    ub_diff = ub - y
    # if this is positive then needs to increase it
    # if it is negative then needs to reduce it
    lb_diff = y - lb
    # need to reduce this
    ub = ub * ub_diff.sign()
    # need to increase this
    lb = lb * lb_diff.sign()
    return (-lb.sum() + ub.sum())

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    
    def reset(self):
        if self.verbose:
            print('Resetting early stopping counter...')
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def seed(fix_seed=2021):
    if fix_seed:
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.cuda.manual_seed(fix_seed)

def adjust_learning_rate(optimizer, epoch, config: Config, verbose=True, scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if config.lradj == 'type1':
        lr_adjust = {epoch: config.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif config.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif config.lradj == 'constant':
        lr_adjust = {epoch: config.learning_rate}
    elif config.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}        
    elif config.lradj == '3':
        lr_adjust = {epoch: config.learning_rate if epoch < 10 else config.learning_rate*0.1}
    elif config.lradj == '4':
        lr_adjust = {epoch: config.learning_rate if epoch < 15 else config.learning_rate*0.1}
    elif config.lradj == '5':
        lr_adjust = {epoch: config.learning_rate if epoch < 25 else config.learning_rate*0.1}
    elif config.lradj == '6':
        lr_adjust = {epoch: config.learning_rate if epoch < 5 else config.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if verbose:
            print('Updating learning rate to {}'.format(lr))