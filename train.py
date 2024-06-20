import os
from utils.colprint import printg, printr, printy
from models.DLinear.Model import Model as DLinear
from models.Autoformer.Model import Model as Autoformer
from models.PatchTST.Model import Model as PatchTST
from models.SegRNN.Model import Model as SegRNN
import time
from utils.ModelWrapper import ModelWrapper
import torch
import argparse
from utils.generate_config import get_config_for

parser = argparse.ArgumentParser(description='Robust training')
parser.add_argument('--model', type=str, required=True, help='model name, options: [DLinear, SegRNN, PatchTST, Autoformer]')
parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--training_type', default='normal', type=str)
parser.add_argument('--version', default='', type=str)
args = parser.parse_args()

model_class = {'DLinear': DLinear, 'SegRNN': SegRNN, 'PatchTST': PatchTST, 'Autoformer': Autoformer}[args.model]

checkpoints_path = f'./checkpoints{args.version}/'
trained_models_path = f'./trained_models{args.version}/'

results_path = f'results{args.version}/results_{args.model}_{args.training_type}.pt'

data_paths = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv', 'electricity.csv', 'traffic.csv', 'weather.csv']
pred_lens = [96, 192, 336, 720]

n_iters = [20, 10, 5, 2, 1] # for occasional training
if os.path.exists(results_path):
    results = torch.load(results_path)
else:
    if args.training_type == 'occasional':
        results = torch.zeros(len(data_paths), len(pred_lens), len(n_iters), 3)
    else:
        results = torch.zeros(len(data_paths), len(pred_lens), 1, 3)

for a, data_path in enumerate(data_paths):
    for b, pred_len in enumerate(pred_lens):
        if args.training_type == 'occasional':
            for c, n_iter in enumerate(n_iters):
                config = get_config_for(data_path, pred_len, model=args.model, checkpoints_path=checkpoints_path, trained_models_path=trained_models_path, training_type=args.training_type, n_iter=n_iter)
                model_wrapper = ModelWrapper(model_class, config, fix_seed=False, verbose=args.verbose)
                print('Starting:', config.data_path, pred_len, args.training_type, n_iter)
                if results[a, b, c, -1] != 0:
                    printy(f'Skipping: {config.data_path} pred_len: {pred_len:03}, Training Type: {args.training_type}, n_iter: {n_iter}')
                    continue
                try:
                    start = time.time()
                    model_wrapper.train()
                    end = time.time()
                    metrics = model_wrapper.test()
                    results[a, b, c] = torch.tensor([metrics[0], metrics[1], end - start])
                    torch.save(results, results_path)
                    printg(f'Model: {model_wrapper}, Dataset: {config.data_path} pred_len: {pred_len:03}, Training Type: {args.training_type}, n_iter: {n_iter}, (mae, mse): {metrics[:2]}, Training Time: {end - start}',)
                except Exception as e:
                    printr(e)
                    printr(f'Error - Model: {model_wrapper}, Dataset: {config.data_path} pred_len: {pred_len:03}, Training Type: {args.training_type}, n_iter: {n_iter}')
        else:
            config = get_config_for(data_path, pred_len, model=args.model, checkpoints_path=checkpoints_path, trained_models_path=trained_models_path, training_type=args.training_type)
            model_wrapper = ModelWrapper(model_class, config, fix_seed=False, verbose=args.verbose)
            print('Starting:', config.data_path, pred_len, args.training_type)
            if results[a, b, 0, -1] != 0:
                printy(f'Skipping: {config.data_path} pred_len: {pred_len:03}, Training Type: {args.training_type}')
                continue
            try:
                start = time.time()
                model_wrapper.train()
                end = time.time()
                metrics = model_wrapper.test()
                results[a, b, 0] = torch.tensor([metrics[0], metrics[1], end - start])
                torch.save(results, results_path)
                printg(f'Model: {model_wrapper}, Dataset: {config.data_path} pred_len: {pred_len:03}, Training Type: {args.training_type}, (mae, mse): {metrics[:2]}, Training Time: {end - start}',)
            except Exception as e:
                printr(e)
                printr(f'Error - Model: {model_wrapper}, Dataset: {config.data_path} pred_len: {pred_len:03}, Training Type: {args.training_type}')            