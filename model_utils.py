import os, tempfile
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
import numpy as np
import random
import pickle
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune import CLIReporter

from scipy.stats import kendalltau
from functools import partial
from pathlib import Path
from torch.optim import Adam
import torch
import torch.nn.functional as F
from math import sqrt


DEFAULT_CONFIG = {
    "hidden_channels": 200,
    "num_layers": 2,
    "num_timesteps": 2,
    "outputs": 1,
    "lr": 0.001,
    "batch_size": 8,
    "Scheduler": "ReduceLROnPlateau",
    "gamma": 0.9
}

class BaseModelTrainer:
    """
    Base class for training and evaluating models.

    Args:
        config (dict): Configuration parameters for the model training. Default is DEFAULT_CONFIG.
        sandbox (bool): Flag indicating whether to use the sandbox path for saving models and training data. Default is True.
        verbose (bool): Flag indicating whether to print training progress. Default is True.
        name (str): Name of the model. Default is 'default'.

    Attributes:
        model (AttentiveFP): The model to be trained.
        optimizer (torch.optim.Adam): The optimizer used for training the model.
        config (dict): Configuration parameters for the model training.
        path (str): Path for saving models and training data.
        verbose (bool): Flag indicating whether to print training progress.
        name (str): Name of the model.

    Methods:
        set_seed(seed=0): Set the random seed for reproducibility.
        save_model(): Save the trained model.
        load_model(): Load a saved model.
        get_predictions(test_loader): Get predictions from the trained model.
        train(loader, model, optimizer, device): Train the model.
        validate(loader, model, device): Validate the model.
        train_and_validate(train_loader, val_loader, num_epochs=50, save_models=False, es_patience=10, save_losses=True): Train and validate the model.
        train_val_DL_hp(config, data_train, data_val, max_num_epochs=50): Train and validate the model with hyperparameter tuning.
        tune_hyperparameters(hp_search_config, data_train, data_val, num_samples=200, max_num_epochs=50, gpus_per_trial=2, verbose=True): Tune hyperparameters for the model.
    """
    def __init__(self, config=DEFAULT_CONFIG, sandbox=True, verbose=True, name='default',seed=0,train_data = None,val_data=None,test_data=None,n_input_feautures=24):
        """
        Initializes an instance of the class.

        Args:
            config (dict): Configuration parameters for the model (default: DEFAULT_CONFIG).
            device (torch.device): The device to be used for computation (default: None).
            sandbox (bool): Flag indicating whether to use the sandbox path (default: True).
            verbose (bool): Flag indicating whether to print verbose output (default: True).
            name (str): Name of the instance (default: 'default').
        """
        try:
            assert type(config) == dict
        except AssertionError:
            raise ValueError('config is not a dictionary')

        try:
            assert all(key in config for key in DEFAULT_CONFIG.keys())
        except AssertionError:
            raise ValueError('config does not have all the necessary keys')

        if seed is not None:
            self.set_seed(seed)

        self.config = config
        self.HP_tuning_result = None

        if sandbox:
            self.path = 'sandbox/'
        else:
            self.path = ''
        self.verbose = verbose
        self.name = name  + '_seed_' + str(seed) if seed is not None else name

        os.makedirs(f'{self.path}models/', exist_ok=True)
        os.makedirs(f'{self.path}hp_tuning', exist_ok=True)
        os.makedirs(f'{self.path}training', exist_ok=True)
        os.makedirs(f'{self.path}testing', exist_ok=True)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.outputs = train_data[0].y.shape[-1]

        self.model = AttentiveFP(in_channels=n_input_feautures, hidden_channels=config["hidden_channels"], out_channels=self.outputs,
                    edge_dim=11, num_layers=config["num_layers"], num_timesteps=config["num_timesteps"],
                    dropout=0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=10**-5)

        self.n_input_feautures = n_input_feautures

    @staticmethod
    def set_seed(seed=0):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): The seed value to set. Defaults to 0.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)

    def set_model_params(self, config = None):
        """
        Set the model parameters.

        Args:
            model: The model to be trained.
            optimizer: The optimizer used to update the model's parameters.
            config: The configuration parameters for the model training.

        Returns:
            None
        """
        if config is None:
            config = self.config
        self.model = AttentiveFP(in_channels=self.n_input_feautures, hidden_channels=config["hidden_channels"], out_channels=self.outputs,
            edge_dim=11, num_layers=config["num_layers"], num_timesteps=config["num_timesteps"],
            dropout=0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=10**-5)
        self.config = config


    def save_model(self):
            """
            Saves the model's state dictionary to a file.

            The model's state dictionary is saved to a file with the name specified by `self.name` in the `models` directory
            located at `self.path`.

            Returns:
                None
            """
            torch.save(self.model.state_dict(), self.path + f'models/{self.name}.pt')

    def load_model(self):
            """
            Loads the model from the specified path.

            Returns:
                None
            """
            self.model.load_state_dict(torch.load(self.path + f'models/{self.name}.pt'))
    
    def train_and_validate(self, num_epochs=50, save_models=False, es_patience=10, save_losses=True,device=None):
        """
        Trains and validates the model for a specified number of epochs.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.
            num_epochs (int, optional): The number of epochs to train the model (default: 50).
            save_models (bool, optional): Whether to save the model after each improvement in validation loss (default: False).
            es_patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping (default: 10).
            save_losses (bool, optional): Whether to save the training and validation losses (default: True).

        Returns:
            tuple: A tuple containing the training losses and validation losses.

        """
        train_loader = DataLoader(self.train_data, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.config["batch_size"], shuffle=False)
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        min_val_los = 1000
        train_losses, val_losses = [], []

        if self.config["Scheduler"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(self.optimizer, patience=2, factor=self.config["gamma"])
        elif self.config["Scheduler"] == "ExponentialLR":
            scheduler = ExponentialLR(self.optimizer, gamma=self.config["gamma"])

        for epoch in range(num_epochs):
            train_loss = self.train(train_loader, self.model, self.optimizer, device)
            val_loss = self.validate(val_loader, self.model, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if save_losses:
                with open(f'{self.path}training/{self.name}_train_losses.pkl', 'wb') as f:
                    pickle.dump(train_losses, f)
                with open(f'{self.path}training/{self.name}_val_losses.pkl', 'wb') as f:
                    pickle.dump(val_losses, f)

            if self.config["Scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            elif self.config["Scheduler"] == "ExponentialLR":
                scheduler.step()
            if self.verbose:
                print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            if val_loss < min_val_los:
                min_val_los = val_loss
                counter = 0
                if save_models:
                    self.save_model()
            else:
                counter += 1
            if counter > es_patience:
                if self.verbose:
                    print('early stopping')
                break
        return train_losses, val_losses
        
        
    def tune_hyperparameters(self,config, num_samples=10, max_num_epochs=50, gpus_per_trial=0):
        """
        Tune hyperparameters for the model using Ray Tune with the ASHA scheduler.

        Args:
            data_train: Training dataset.
            data_val: Validation dataset.
            num_samples (int): Number of trials to run. Defaults to 10.
            max_num_epochs (int): Maximum number of epochs for training. Defaults to 50.
            gpus_per_trial (float): Number of GPUs to allocate per trial. Defaults to 0.

        """
        def train_step(data, model, optimizer, device,config):
            """
            Trains the model using the given data loader, optimizer, and device.

            Args:
                loader (DataLoader): The data loader for training data.
                model (torch.nn.Module): The model to be trained.
                optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
                device (torch.device): The device to be used for training.

            Returns:
                float: The root mean squared error (RMSE) of the model's performance.
            """
    
            model.train()
            model.to(device)
            total_loss = total_examples = 0
            data = ray.get(data)
            loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)

            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = F.mse_loss(out, data.y.view(-1, 1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs
                total_examples += data.num_graphs
            return np.sqrt(total_loss / total_examples)

        def validate(data, model, device,config):
            """
            Validates the model using the given data loader and device.

            Args:
                loader (DataLoader): The data loader for validation data.
                model (torch.nn.Module): The model to be validated.
                device (torch.device): The device to be used for validation.

            Returns:
                float: The root mean squared error (RMSE) of the model's performance.
            """
            model.eval()
            model.to(device)
            total_loss = total_examples = 0
            data = ray.get(data)
            loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    loss = F.mse_loss(out, data.y.view(-1, 1))
                    total_loss += loss.item() * data.num_graphs
                    total_examples += data.num_graphs
            return np.sqrt(total_loss / total_examples)

        def get_predictions(model,data,device,config):
                    """
                    Get predictions for the given test data.

                    Args:
                        test_loader (torch.utils.data.DataLoader): The data loader for the test data.

                    Returns:
                        tuple: A tuple containing two lists - predictions and ys.
                            - predictions (list): A list of predicted values.
                            - ys (list): A list of true values.
                    """
                    model.eval()
                    predictions = []
                    ys = []
                    model.to(device)
                    data = ray.get(data)
                    test_loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
                    with torch.no_grad():
                        for data in test_loader:
                            data = data.to(device)
                            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                            predictions.extend(out.detach().cpu().numpy())
                            ys.extend(data.y.detach().cpu().numpy())
                    return predictions, ys
        def train_model_with_ray(config,device=None,data_train=None, data_val=None,output=None,n_input_feautures=None):
            model = AttentiveFP(in_channels=n_input_feautures, hidden_channels=config["hidden_channels"], out_channels=output,
                    edge_dim=11, num_layers=config["num_layers"], num_timesteps=config["num_timesteps"],
                    dropout=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=10**-5)

            if train.get_checkpoint():
                with train.get_checkpoint().as_directory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                        data = pickle.load(fp)
                    start_epoch = data['epoch']
                    model.load_state_dict(data['net_state_dict'])
                    optimizer.load_state_dict(data['optimizer_state_dict'])
            else:
                start_epoch = 0
        

            for epoch in range(1000):
                train_loss = train_step(data_train, model, optimizer, device,config)
                val_loss = validate(data_val, model, device,config)
                preds, ys = get_predictions(model, data_val, device,config)
                tau, _ = kendalltau(preds, ys)
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                        pickle.dump({'epoch': epoch,
                                        'net_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict()}, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report({"kendall_tau": tau},
                    checkpoint=checkpoint)


        scheduler = ASHAScheduler(
            metric="kendall_tau",
            mode="max",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)


        if gpus_per_trial > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        data_train_ref = ray.put(self.train_data)  # large training dataset
        data_val_ref = ray.put(self.val_data)   

        result = tune.run(
            partial(train_model_with_ray, data_train=data_train_ref, data_val=data_val_ref,device=device,output=self.outputs,n_input_feautures=self.n_input_feautures),
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler)

        best_trial = result.get_best_trial("kendall_tau", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final kendall_tau: {}".format(best_trial.last_result["kendall_tau"]))
        self.HP_tuning_result = result
    
    def eval_best_HPs(self,data_val):
        
        best_trial = self.HP_tuning_result.get_best_trial("kendall_tau", "max", "last")
        best_trained_model = AttentiveFP(in_channels=self.n_input_feautures, hidden_channels=best_trial.config["hidden_channels"], out_channels=self.outputs,
                    edge_dim=11, num_layers=best_trial.config["num_layers"], num_timesteps=best_trial.config["num_timesteps"],
                    dropout=0.0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = Adam(best_trained_model.parameters(), lr=best_trial.config["lr"], weight_decay=10 ** -5)
        val_loader = DataLoader(data_val, batch_size=best_trial.config["batch_size"], shuffle=False)
        with self.HP_tuning_result.get_best_checkpoint(trial=best_trial, metric="kendall_tau", mode="max").as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                data = pickle.load(fp)
            start_epoch = data['epoch']
            best_trained_model.load_state_dict(data['net_state_dict'])
            optimizer.load_state_dict(data['optimizer_state_dict'])
        preds, ys = self.get_predictions(model = best_trained_model,test_loader = val_loader,device = device)
        tau, _ = kendalltau(preds, ys)
        print(f'Kendall Tau: {tau}')
        self.set_model_params(best_trial.config)

    def train(self, loader, model, optimizer, device):
        """
        Trains the model using the given data loader, optimizer, and device.

        Args:
            loader (DataLoader): The data loader for training data.
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model's parameters.
            device (torch.device): The device to be used for training.

        Returns:
            float: The root mean squared error (RMSE) of the model's performance.
        """
        model.train()
        model.to(device)
        total_loss = total_examples = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if self.outputs == 1:
                loss = F.mse_loss(out, data.y.view(-1, 1))
            else:
                loss = self.compute_weighted_loss(data.y, out, self.outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_examples += data.num_graphs
        return np.sqrt(total_loss / total_examples)
    

    def validate(self, loader, model, device):
        """
        Validates the model using the given data loader and device.

        Args:
            loader (DataLoader): The data loader for validation data.
            model (torch.nn.Module): The model to be validated.
            device (torch.device): The device to be used for validation.

        Returns:
            float: The root mean squared error (RMSE) of the model's performance.
        """
        model.eval()
        model.to(device)
        total_loss = total_examples = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                if self.outputs == 1:
                    loss = F.mse_loss(out, data.y.view(-1, 1))
                else:
                    loss = self.compute_weighted_loss(data.y, out, self.outputs)
                total_loss += loss.item() * data.num_graphs
                total_examples += data.num_graphs
        return np.sqrt(total_loss / total_examples)
        


class SingleTaskModelTrainer(BaseModelTrainer):
    """
    Trainer class for single-task models.

    Args:
        BaseModelTrainer (class): Base class for model trainers.

    Methods:
        train(loader, model, optimizer, device): Trains the model using the given data loader, optimizer, and device.
        validate(loader, model, device): Validates the model using the given data loader and device.

    Returns:
        float: The root mean squared error (RMSE) of the model's performance.
    """
    
    def get_predictions(self, model,test_loader,device):
            """
            Get predictions for the given test data.

            Args:
                test_loader (torch.utils.data.DataLoader): The data loader for the test data.

            Returns:
                tuple: A tuple containing two lists - predictions and ys.
                    - predictions (list): A list of predicted values.
                    - ys (list): A list of true values.
            """
            model.eval()
            predictions = []
            ys = []
            model.to(device)
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    predictions.extend(out.detach().cpu().numpy())
                    ys.extend(data.y.detach().cpu().numpy())
            return predictions, ys
    
class MultiTaskModelTrainer(BaseModelTrainer):
    """
    A class for training and validating a multi-task model.

    Args:
        BaseModelTrainer (class): The base class for model trainers.

    Methods:
        train(train_loader, epochs, outputs): Trains the model for the specified number of epochs.
        validate(val_loader, outputs): Validates the model using the validation data.
        compute_weighted_loss(targets, predictions, outputs): Computes the weighted loss for the model.
        get_predictions(test_loader): Gets the predictions of the model for the test data.
    """
    def compute_weighted_loss(self, targets, predictions, outputs):
        """
        Computes the weighted loss for the model.

        Args:
            targets (Tensor): The target values.
            predictions (Tensor): The predicted values.
            outputs (int): The number of outputs of the model.

        Returns:
            Tuple: A tuple containing the weighted loss and the number of labels.
        """
        weighted_loss = num_labels = 0
        for i in range(outputs):
            y_tmp = targets[:, i]
            out_tmp = predictions[:, i]
            present_label_indices = torch.nonzero(y_tmp != 0, as_tuple=False).view(-1)
            num_labels += len(present_label_indices)

            if len(present_label_indices) > 0:
                out_tmp_present = out_tmp[present_label_indices]
                y_tmp_present = y_tmp[present_label_indices]
                loss_tmp_present = F.mse_loss(out_tmp_present, y_tmp_present)
                weighted_loss += loss_tmp_present * len(present_label_indices)

        return weighted_loss / num_labels if num_labels > 0 else 0
    

    def get_predictions(self, model,test_loader,device):
            """
            Get predictions for the given test data.

            Args:
                test_loader (torch.utils.data.DataLoader): The data loader for the test data.

            Returns:
                tuple: A tuple containing two lists - predictions and ys.
                    - predictions (list): A list of predicted values.
                    - ys (list): A list of true values.
            """
            model.eval()
            preds, ys = [], []
            model.to(device)
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    for i in range(self.outputs):
                        y_tmp = data.y[:, i]
                        out_tmp = out[:, i]
                        present_label_indices = torch.nonzero(y_tmp != 0).view(-1)
                        if len(present_label_indices) > 0:
                            out_tmp_present = torch.index_select(out_tmp, 0, present_label_indices)
                            y_tmp_present = torch.index_select(y_tmp, 0, present_label_indices)
                            preds.extend(out_tmp_present.detach().cpu().numpy())
                            ys.extend(y_tmp_present.detach().cpu().numpy())
            return preds, ys
    
    def get_preds_per_task(self, model,test_loader,device):
        preds = [[] for i in range(self.outputs)]
        ys = [[] for i in range(self.outputs)]
        model.eval()
        preds, ys = [], []
        model.to(device)
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                for i in range(self.outputs):
                    y_tmp = data.y[:, i]
                    out_tmp = out[:, i]
                    present_label_indices = torch.nonzero(y_tmp != 0).view(-1)
                    if len(present_label_indices) > 0:
                        out_tmp_present = torch.index_select(out_tmp, 0, present_label_indices)
                        y_tmp_present = torch.index_select(y_tmp, 0, present_label_indices)
                        preds[i].extend(out_tmp_present.detach().cpu().numpy())
                        ys[i].extend(y_tmp_present.detach().cpu().numpy())
        return preds, ys

    
    def tune_hyperparameters(self,config, num_samples=10, max_num_epochs=50, gpus_per_trial=0, cpus_per_trial=1):
        """
        Tune hyperparameters for the model using Ray Tune with the ASHA scheduler.

        Args:
            data_train: Training dataset.
            data_val: Validation dataset.
            num_samples (int): Number of trials to run. Defaults to 10.
            max_num_epochs (int): Maximum number of epochs for training. Defaults to 50.
            gpus_per_trial (float): Number of GPUs to allocate per trial. Defaults to 0.

        """
        def compute_weighted_loss(targets, predictions, outputs):
            """
            Computes the weighted loss for the model.

            Args:
                targets (Tensor): The target values.
                predictions (Tensor): The predicted values.
                outputs (int): The number of outputs of the model.

            Returns:
                Tuple: A tuple containing the weighted loss and the number of labels.
            """
            weighted_loss = num_labels = 0
            for i in range(outputs):
                y_tmp = targets[:, i]
                out_tmp = predictions[:, i]
                present_label_indices = torch.nonzero(y_tmp != 0, as_tuple=False).view(-1)
                num_labels += len(present_label_indices)

                if len(present_label_indices) > 0:
                    out_tmp_present = out_tmp[present_label_indices]
                    y_tmp_present = y_tmp[present_label_indices]
                    loss_tmp_present = F.mse_loss(out_tmp_present, y_tmp_present)
                    weighted_loss += loss_tmp_present * len(present_label_indices)

            return weighted_loss / num_labels if num_labels > 0 else 0, num_labels
        def train_step(data, model, optimizer, device,config,outputs):
            """
            Trains the model for the specified number of epochs.

            Args:
                train_loader (DataLoader): The data loader for the training data.
                epochs (int): The number of epochs to train the model.
                outputs (int): The number of outputs of the model.
            """

            model.train()
            model.to(device)
            total_loss = total_examples = 0
            data = ray.get(data)
            loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)

            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                weighted_loss, num_labels = compute_weighted_loss(data.y, out, outputs)
                weighted_loss.backward()
                total_loss += float(weighted_loss) * data.num_graphs
                total_examples += data.num_graphs
                optimizer.step()
            return np.sqrt(total_loss / total_examples)


        def validate(data, model, device,config,outputs):
            """
            Validates the model using the validation data.

            Args:
                val_loader (DataLoader): The data loader for the validation data.
                outputs (int): The number of outputs of the model.
            """
            model.eval()
            model.to(device)
            total_loss = total_examples = 0
            data = ray.get(data)
            loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    weighted_loss, num_labels = compute_weighted_loss(data.y, out, outputs)
                    total_loss += float(weighted_loss) * data.num_graphs
                    total_examples += data.num_graphs

            return np.sqrt(total_loss / total_examples)

        
        def get_predictions(model,data,device,config,outputs):
            """
            Gets the predictions of the model for the test data.

            Args:
                test_loader (DataLoader): The data loader for the test data.

            Returns:
                Tuple: A tuple containing the predicted values and the target values.
            """
            model.eval()
            preds = []
            ys = []
            model.to(device)
            data = ray.get(data)
            test_loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    for i in range(outputs):
                        y_tmp = data.y[:, i]
                        out_tmp = out[:, i]
                        present_label_indices = torch.nonzero(y_tmp != 0).view(-1)
                        if len(present_label_indices) > 0:
                            out_tmp_present = torch.index_select(out_tmp, 0, present_label_indices)
                            y_tmp_present = torch.index_select(y_tmp, 0, present_label_indices)
                            preds.extend(out_tmp_present.detach().cpu().numpy())
                            ys.extend(y_tmp_present.detach().cpu().numpy())

            return preds, ys
                

        def train_model_with_ray(config,device=None,data_train=None, data_val=None,outputs=None,n_input_feautures=None):
            model = AttentiveFP(in_channels=n_input_feautures, hidden_channels=config["hidden_channels"], out_channels=outputs,
                    edge_dim=11, num_layers=config["num_layers"], num_timesteps=config["num_timesteps"],
                    dropout=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=10**-5)

            if train.get_checkpoint():
                with train.get_checkpoint().as_directory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                        data = pickle.load(fp)
                    start_epoch = data['epoch']
                    model.load_state_dict(data['net_state_dict'])
                    optimizer.load_state_dict(data['optimizer_state_dict'])
            else:
                start_epoch = 0
            

            for epoch in range(1000):
                train_loss = train_step(data_train, model, optimizer, device,config,outputs)
                val_loss = validate(data_val, model, device,config,outputs)
                preds, ys = get_predictions(model, data_val, device,config,outputs)
                tau, _ = kendalltau(preds, ys)
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                        pickle.dump({'epoch': epoch,
                                        'net_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict()}, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report({"kendall_tau": tau},
                    checkpoint=checkpoint)

            

        scheduler = ASHAScheduler(
            metric="kendall_tau",
            mode="max",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)


        if gpus_per_trial > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        data_train_ref = ray.put(self.train_data)
        data_val_ref = ray.put(self.val_data)

        result = tune.run(
            partial(train_model_with_ray, data_train=data_train_ref, data_val=data_val_ref,device=device,outputs=self.outputs,n_input_feautures=self.n_input_feautures),
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler)

        best_trial = result.get_best_trial("kendall_tau", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final kendall_tau: {}".format(best_trial.last_result["kendall_tau"]))
   
        self.HP_tuning_result = result
        
        with open(f'{self.path}hp_tuning/{self.name}_hp_tuning_result.pkl', 'wb') as f:
            pickle.dump(result, f)
    def eval_best_HPs(self,data_val):
            
            best_trial = self.HP_tuning_result.get_best_trial("kendall_tau", "max", "last")
            best_trained_model = AttentiveFP(in_channels=self.n_input_feautures, hidden_channels=best_trial.config["hidden_channels"], out_channels=self.outputs,
                        edge_dim=11, num_layers=best_trial.config["num_layers"], num_timesteps=best_trial.config["num_timesteps"],
                        dropout=0.0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            optimizer = Adam(best_trained_model.parameters(), lr=best_trial.config["lr"], weight_decay=10 ** -5)
            val_loader = DataLoader(data_val, batch_size=best_trial.config["batch_size"], shuffle=False)
            with self.HP_tuning_result.get_best_checkpoint(trial=best_trial, metric="kendall_tau", mode="max").as_directory() as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                    data = pickle.load(fp)
                start_epoch = data['epoch']
                best_trained_model.load_state_dict(data['net_state_dict'])
                optimizer.load_state_dict(data['optimizer_state_dict'])
            preds, ys = self.get_predictions(model = best_trained_model,test_loader = val_loader,device = device)
            tau, _ = kendalltau(preds, ys)
            print(f'Kendall Tau: {tau}')
            self.set_model_params(best_trial.config)

