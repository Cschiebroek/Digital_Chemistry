#basics
import os
from math import sqrt
import pandas
import numpy as np
import random
import pickle



#torch stuff
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.loader import DataLoader

import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set random seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

default_properties = ['LogVP', 'LogP', 'LogOH', 'LogBCF', 'LogHalfLife', 'BP', 'Clint', 'FU', 'LogHL', 'LogKmHL', 'LogKOA', 'LogKOC', 'MP', 'LogMolar']


def train_multi(train_loader, model, optimizer, device, outputs,props_to_train):
    idx_to_train = [default_properties.index(prop) for prop in props_to_train]
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        weighted_loss = num_labels = 0
        for i in range(outputs):
            if i not in idx_to_train:
                continue
            y_tmp = data.y[:, i]
            out_tmp = out[:, i]
            # Find indices where labels are available
            present_label_indices = torch.nonzero(y_tmp != -1).view(-1)
            num_labels += len(present_label_indices)

            if len(present_label_indices) > 0:
                # Extract only the available label indices
                out_tmp_present = torch.index_select(out_tmp, 0, present_label_indices)
                y_tmp_present = torch.index_select(y_tmp, 0, present_label_indices)

                # Calculate MSE loss only for available labels
                loss_tmp_present = F.mse_loss(out_tmp_present, y_tmp_present)
                if not torch.isnan(loss_tmp_present):
                    weighted_loss += loss_tmp_present * len(present_label_indices)

        weighted_loss = weighted_loss / num_labels
        weighted_loss.backward()

        total_loss += float(weighted_loss) * data.num_graphs
        total_examples += data.num_graphs

        # clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        del data

    return sqrt(total_loss / total_examples)


def validate_multi(val_loader, model, outputs,props_to_train):
    idx_to_train = [default_properties.index(prop) for prop in props_to_train]
    total_loss = total_examples = 0
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        weighted_loss = num_labels = 0
        for i in range(outputs):
            if i not in idx_to_train:
                continue

            y_tmp = data.y[:, i]
            out_tmp = out[:, i]
            # Find indices where labels are available
            present_label_indices = torch.nonzero(y_tmp != -1).view(-1)
            num_labels += len(present_label_indices)

            if len(present_label_indices) > 0:
                # Extract only the available label indices
                out_tmp_present = torch.index_select(out_tmp, 0, present_label_indices)
                y_tmp_present = torch.index_select(y_tmp, 0, present_label_indices)

                # Calculate MSE loss only for available labels
                loss_tmp_present = F.mse_loss(out_tmp_present, y_tmp_present)
                if not torch.isnan(loss_tmp_present):
                    weighted_loss += loss_tmp_present * len(present_label_indices)

        weighted_loss = weighted_loss / num_labels

        total_loss += float(weighted_loss) * data.num_graphs
        total_examples += data.num_graphs
        del data

    return sqrt(total_loss / total_examples)

def train_and_validate_multi(model, train_loader, val_loader, optimizer, num_epochs, outputs, verbose=True,props_to_train = default_properties,name='test'):
    train_losses = []
    val_losses = []
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.9,verbose=False)

    min_val_los = 10000000
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_multi(train_loader, model, optimizer, device, outputs,props_to_train)
        train_losses.append(train_loss)

        model.eval()
        val_loss = validate_multi(val_loader, model, outputs,props_to_train)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < min_val_los:
            min_val_los = val_loss
            counter = 0
            torch.save(model.state_dict(), f'{name}.pt')

        else:
            counter += 1
        if counter > 10:
            if verbose:
                print('early stopping')
            break
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


from scipy.stats import spearmanr,kendalltau

def get_preds_per_task(model,val_loader,outputs,props_to_predict):
    idx_to_pred = [default_properties.index(prop) for prop in props_to_predict]
    model.to(device)
    model.eval()
    preds = tuple([[] for i in range(len(props_to_predict))])
    ys = tuple([[] for i in range(len(props_to_predict))])
    counter = 0
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        for i in range(outputs):
            if i not in idx_to_pred:
                continue

            y_tmp = data.y[:, i]
            out_tmp = out[:, i]
            # Find indices where labels are available
            present_label_indices = torch.nonzero(y_tmp != -1).view(-1)

            counter += len(present_label_indices)

            # Create arrays with -1 values for indices not in present_label_indices
            preds_tmp = torch.full_like(y_tmp, -1)
            ys_tmp = torch.full_like(y_tmp, -1)

            if len(present_label_indices) > 0:
                # Replace -1 values with predictions and true values where available
                preds_tmp[present_label_indices] = out_tmp[present_label_indices]
                ys_tmp[present_label_indices] = y_tmp[present_label_indices]
            tupp_idx = props_to_predict.index(default_properties[i])
            preds[tupp_idx].extend(preds_tmp.detach().cpu().numpy().tolist())
            ys[tupp_idx].extend(ys_tmp.detach().cpu().numpy().tolist())

    print(counter)
    return preds, ys

def vals_to_df(vals,props_to_predict,scaled=True):
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    df = pd.DataFrame(vals).T
    df.columns = props_to_predict
    if scaled:
        df = df.replace(-1, float('nan'))
        for prop in default_properties:
            if prop not in df.columns:
                df[prop] = float('nan')
        df = df[default_properties]
        df[default_properties] = scaler.inverse_transform(df[default_properties])
        #drop properties with only nans
        df = df.dropna(axis=1, how='all')
    return df
