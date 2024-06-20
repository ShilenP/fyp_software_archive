# Insired from https://github.com/zhouhaoyi/Informer2020/blob/main/exp/exp_informer.py.

import time
import torch
from utils.config import Config
from utils.data_factory import data_provider
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate, robust_loss
import os
from torch.optim import lr_scheduler 
from utils.metric import metric
from utils.tools import seed 

class ModelWrapper:
    """
    This class is a wrapper for the model. It is responsible for training, testing and saving the model.
    """
    def __init__(self, model_class, config: Config, fix_seed=None, verbose=True):
        """
        Initialize the ModelWrapper class.
        Args:
            model_class: type: The model class to be used.
            config: Config: The configuration object.
            fix_seed: int: The seed to be used for reproducibility.
            verbose: bool: Whether to print the logs or not.
        """
        self.model_class: type = model_class
        self.config: Config = config
        self.fix_seed: int | None = fix_seed
        self.verbose: bool = verbose
        self.model: nn.Module = model_class(config).float().to(config.device)
        self.trained_model_path: str = config.get_trained_model_path(self.model)
        self.checkpoint_path: str = config.get_checkpoint_path(self.model)

    def get_data(self, flag):
        """
        Get the data loader for the given flag.
        Args:
            flag: str: The flag for the data loader.
        Returns:
            Tuple: The dataset and data loader for the given flag.
        """
        return  data_provider(self.config, flag=flag, verbose=self.verbose)

    def is_trained_cached(self):
        """
        Check if the model is already trained and cached.
        Returns:
            bool: Whether the model is already trained and cached or not.
        """
        return os.path.exists(self.trained_model_path)

    def train(self):
        """
        Train the model.
        """
        if self.is_trained_cached():
            if self.verbose:
                print("Loading trained model from: ", self.trained_model_path)
            self.load_model(self.trained_model_path)
            return
        elif self.verbose:
            print("Training model from scratch")

        seed(self.fix_seed)
        _, train_loader = self.get_data('train')
        _, vali_loader = self.get_data('val')
        _, test_loader = self.get_data('test')

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        early_stopping = EarlyStopping(patience=self.config.patience, verbose=self.verbose)
        model_optim = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        criterion = nn.MSELoss()
        train_steps = len(train_loader)
        device = self.config.device

        if self.config.training_type == 'weighted':
            target_epsilon = self.config.target_epsilon
            epsilon = self.config.initial_epsilon
            epsilon_diff = self.config.epsilon_growth_factor * target_epsilon / self.config.train_epochs
            kappa = self.config.initial_kappa
            kappa_diff = self.config.kappa_growth_factor / self.config.train_epochs
            final_kappa = self.config.final_kappa
        elif self.config.training_type == 'occasional':
            epsilon = 0.0001

        scheduler = None
        if self.config.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                        steps_per_epoch = train_steps,
                                        pct_start = self.config.pct_start,
                                        epochs = self.config.train_epochs,
                                        max_lr = self.config.learning_rate)

        for epoch in range(self.config.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                outputs = self.predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs = outputs[:, -self.config.pred_len:, :]
                batch_y = batch_y[:, -self.config.pred_len:, :]

                if self.config.training_type == 'weighted':
                    if kappa != 1:
                        loss_fit = criterion(outputs, batch_y.to(device))
                        
                    if kappa:
                        lower_bound, upper_bound = (batch_x - epsilon), (batch_x + epsilon)
                        lb, ub = self.model.forward_bounds(lower_bound, upper_bound)
                        loss_robust = robust_loss(lb, ub, batch_y)
                        if kappa == 1:
                            loss = loss_robust
                        else:
                            loss = loss_fit * (1 - kappa) + loss_robust * kappa
                    else:
                        loss = loss_fit

                elif self.config.training_type == 'occasional' and ((i+1) % self.config.n_iter) == 0:
                    lower_bound, upper_bound = (batch_x - epsilon), (batch_x + epsilon)
                    lb, ub = self.model.forward_bounds(lower_bound, upper_bound)
                    loss = robust_loss(lb, ub, batch_y)
                else:
                    loss = criterion(outputs, batch_y)

                if self.verbose and (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, train_loss[-1]))
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
            if self.config.lradj == 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.config, self.verbose, scheduler)
                if scheduler:
                    scheduler.step()

            if self.verbose:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            if self.verbose:
                test_loss = self.vali(test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.checkpoint_path)

            if self.config.training_type == 'weighted':
                if early_stopping.counter > 1 and (target_epsilon > epsilon or final_kappa > kappa):
                    epsilon  = min(epsilon * 10, target_epsilon)
                    kappa = min(kappa + kappa_diff, final_kappa)
                    early_stopping.reset()

            if early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping")
                break

            if self.config.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.config, self.verbose, scheduler)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                
            if self.config.training_type == 'weighted':
                epsilon = min(epsilon + epsilon_diff, target_epsilon)
                kappa = min(kappa + kappa_diff, final_kappa)

        self.load_model(self.checkpoint_path)
        if not os.path.exists(self.trained_model_path):
            os.makedirs(self.trained_model_path)
        self.save_model(self.trained_model_path)

    def save_model(self, path):
        """
        Save the model to the given path.
        Args:
            path: str: The path to save the model.
        """
        torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))

    def load_model(self, path):
        """
        Load the model from the given path.
        Args:
            path: str: The path to load the model.
        """
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

    def predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Predict the output for the given input.
        Args:
            batch_x: torch.Tensor: The input tensor.
            batch_y: torch.Tensor: The target tensor.
            batch_x_mark: torch.Tensor: The input mask tensor.
            batch_y_mark: torch.Tensor: The target mask tensor.
        Returns:
            torch.Tensor: The predicted output tensor.
        """
        if 'former' not in str(self.model):
            return self.model(batch_x)
        else:
            batch_x_mark = batch_x_mark.float().to(self.config.device)
            batch_y_mark = batch_y_mark.float().to(self.config.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.config.device)
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

    def vali(self, data_loader, criterion):
        """
        Validate the model.
        Args:
            data_loader: DataLoader: The data loader for the validation data.
            criterion: The loss function.
        Returns:
            float: The validation loss.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.config.device)
                batch_y = batch_y.float()

                outputs = self.predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = outputs[:, -self.config.pred_len:, :]
                batch_y = batch_y[:, -self.config.pred_len:, :]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        return np.average(total_loss)
    
    def eval(self):
        self.model.eval()   

    def test(self):
        """
        Test the model.
        Returns:
            tuple: The metrics for the model.(mae, mse, rmse, mape, mspe, rse, corr)
        """
        self.model.eval()
        _, test_loader = self.get_data('test')
        preds = []
        trues = []
        inputx = []
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_loader:
                batch_x = batch_x.float().to(self.config.device)
                batch_y = batch_y.float()

                outputs = self.predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -self.config.pred_len:, :].cpu().numpy()
                inputs = batch_x.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(inputs)
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        return metric(preds, trues)
    
    def forward_bounds(self, lower_bound, upper_bound):
        try:
            return self.model.forward_bounds(lower_bound, upper_bound)
        except AttributeError:
            print('Model does not support forward bounds')
            return None, None
        
    def forward(self, x):
        return self.model.forward(x)

    def __str__(self) -> str:
        return str(self.model)