from pred_elem_seq import (
    Building,
    write_dic_to_file,
    read_dic_from_file,
    clear_sim_folders,
    get_n_model_params,
)
from pandas import DataFrame
from dataclasses import dataclass, field
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pathlib import Path
import pickle
import time
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from itertools import repeat
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor as Pool
import pyarrow.parquet as pq
import copy
import math
from collections import defaultdict
import random
from joblib import Parallel, delayed
from torch_lr_finder import LRFinder
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler
from torch import autocast
from torch.utils.data import IterableDataset, get_worker_info

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AnnConfig:
    # contains all neural network configuration values
    building: Building
    nn_type: str
    pretrained: bool = False
    loss_func: str = "mse"
    activation: str = "relu"
    solver: str = "adamW"
    div_f_lr: int = 10
    fin_div_f_lr: int = 40
    pct_start: float = 0.3
    max_lr: float = 4e-4
    opt_betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-2
    batch_size: int = 1000
    epochs: int = 6
    n_lhs: int = 53**2
    n_lhs_uni: int = 0
    n_smp_uni: int = 0
    train_perc: float = 0.7
    seq_len: int = 8
    pred_len: int = 1
    n_lin_blocks: int = 0
    n_lin_layers: int = 0
    n_nodes: int = 0
    n_conv_blocks: int = 0
    n_conv_layers: int = 0
    n_channels: int = 0
    multidx_names: list = field(init=False)
    check_train_loss: int = field(init=False)
    kernels: list = field(init=False)
    avg_pool_krnl: int = field(init=False)
    conv_dim0: int = field(init=False)
    conv_dim1: int = field(init=False)
    conv_flat: int = field(init=False)
    lin_dim: int = field(init=False)
    out_dim0: int = field(init=False)
    obj_names: list = field(init=False)
    out_dim1: int = field(init=False)

    def __post_init__(self):
        # check if ann configuration attributes are correctly defined
        if self.n_lhs == 0:
            if self.n_lhs_uni == 0 and self.n_smp_uni == 0:
                raise ValueError("Need to specify n_lhs or n_lhs_uni and n_smp_uni")
            elif self.n_lhs_uni == 0:
                raise ValueError(
                    "If n_smp_uni is specified, n_lhs_uni must be specified as well"
                )
            elif self.n_smp_uni == 0:
                raise ValueError(
                    "If n_lhs_uni is specified, n_smp_uni must be specified as well"
                )
            else:
                self.multidx_names = ["start_pt", "uni_param", "time", "uni_sim"]
        else:
            if self.n_lhs_uni != 0:
                raise ValueError(
                    "If n_lhs is specified, n_lhs_uni cannot be specified as well"
                )
            elif self.n_smp_uni != 0:
                raise ValueError(
                    "If n_lhs is specified, n_smp_uni cannot be specified as well"
                )
            else:
                self.multidx_names = ["time", "lhs_sim"]

        # set all other ann configuration attributes
        self.check_train_loss = self.batch_size * round(
            1e6 / (self.batch_size * self.pred_len)
        )

        # if predicting time series
        if self.pred_len > 1:
            self.kernels = list(range(10, 10 - self.n_conv_layers, -1))
        else:
            # the default kernel sizes are the sequence length descending by 1 for each convolutional layer
            self.kernels = list(
                range(self.seq_len, self.seq_len - self.n_conv_layers, -1)
            )

        if self.nn_type == "conv":
            self.conv_dim0 = (
                self.building.n_vars
                + self.building.n_unknown_params
            )
            self.conv_dim1 = self.seq_len
            self.conv_flat = 0
            self.lin_dim = 0
            self.avg_pool_krnl = 0

        elif self.nn_type == "conv_lin":
            self.conv_dim0 = (
                self.building.n_vars
                + self.building.n_unknown_params
            )
            self.conv_dim1 = self.seq_len
            # if predicting time series
            if self.pred_len > 1:
                self.avg_pool_krnl = 100
                self.conv_flat = (
                    self.conv_dim1 // self.avg_pool_krnl
                ) * self.n_channels
            else:
                self.avg_pool_krnl = 0
                self.conv_flat = self.conv_dim1 * self.n_channels
            self.lin_dim = self.conv_flat

        else:
            raise NotImplementedError(
                "nn_type must be 'conv' or 'conv_lin'"
            )

        self.out_dim0 = self.building.n_objs
        self.obj_names = self.building.obj_names
        self.out_dim1 = self.pred_len


class ConvNet(nn.Module):
    def __init__(self, ann):
        super().__init__()
        seq = nn.ModuleDict()
        # give sequence dictionary attribute to CNN PyTorch model
        self.seq = seq
        # set AnnConfig object as attribute
        self.cfg = ann.cfg
        # for each convolutional block
        for b in range(ann.cfg.n_conv_blocks):
            # if it's the first convolutional block
            if b == 0:
                bias = False
                n_chnls_in = ann.cfg.conv_dim0
                n_chnls_out = ann.cfg.n_channels
            # if it's the last convolutional block
            elif b == ann.cfg.n_conv_blocks - 1:
                bias = True
                n_chnls_in = ann.cfg.n_channels
                n_chnls_out = ann.cfg.n_channels
            else:
                bias = False
                n_chnls_in = ann.cfg.n_channels
                n_chnls_out = ann.cfg.n_channels

            # add shortcut
            # no bias because subsequent batch normalization layer makes it redundant
            seq[f"conv_blk{b}"] = nn.Conv1d(
                in_channels=n_chnls_in,
                out_channels=n_chnls_out,
                kernel_size=1,
                padding="same",
                bias=bias,
            )
            # add a batch normalization layer for every shortcut but the last one
            if b != ann.cfg.n_conv_blocks - 1:
                seq[f"bn_conv_blk{b}"] = nn.BatchNorm1d(n_chnls_out)

            # if using parametric relu as the activation function
            if ann.cfg.activation == "prelu":
                seq[f"act_conv_blk{b}"] = nn.PReLU(n_chnls_out)
            elif ann.cfg.activation == "relu":
                seq[f"act_conv_blk{b}"] = nn.ReLU()
            else:
                raise ValueError("Activation function must be 'prelu' or 'relu'")

            # for each convolutional layer
            for c in range(ann.cfg.n_conv_layers):
                if b == 0 and c == 0:
                    bias = False
                    n_chnls_in = ann.cfg.conv_dim0
                    n_chnls_out = ann.cfg.n_channels
                elif b == ann.cfg.n_conv_blocks - 1 and c == ann.cfg.n_conv_layers - 1:
                    # include bias because there is no subsequent batch normalization layer
                    bias = True
                    n_chnls_in = ann.cfg.n_channels
                    n_chnls_out = ann.cfg.n_channels
                else:
                    bias = False
                    n_chnls_in = ann.cfg.n_channels
                    n_chnls_out = ann.cfg.n_channels

                # add convolutional 1D layer
                # no bias because subsequent batch normalization layer makes it redundant
                seq[f"conv_blk{b}_lyr{c}"] = nn.Conv1d(
                    in_channels=n_chnls_in,
                    out_channels=n_chnls_out,
                    kernel_size=ann.cfg.kernels[c],
                    padding="same",
                    bias=bias,
                )
                # add a batch normalization for all convolutional layers but the last one
                if b != ann.cfg.n_conv_blocks - 1 or c != ann.cfg.n_conv_layers - 1:
                    seq[f"bn_conv_blk{b}_lyr{c}"] = nn.BatchNorm1d(n_chnls_out)

                # if using parametric ReLu as the activation function
                if ann.cfg.activation == "prelu":
                    seq[f"act_conv_blk{b}_lyr{c}"] = nn.PReLU(n_chnls_out)
                # if using standard ReLu as the activation function
                elif ann.cfg.activation == "relu":
                    seq[f"act_conv_blk{b}_lyr{c}"] = nn.ReLU()
                else:
                    raise ValueError("Activation function must be 'prelu' or 'relu'")

        # add final linear output layer
        seq[f"out"] = nn.Conv1d(
            in_channels=ann.cfg.n_channels,
            out_channels=ann.cfg.out_dim0,
            kernel_size=1,
            padding="same",
            bias=True,
        )

    def forward(self, x_conv):
        # for each convolutional block
        for b in range(self.cfg.n_conv_blocks):
            # convolve shortcut
            x_conv_sc = self.seq[f"conv_blk{b}"](x_conv)
            if b != self.cfg.n_conv_blocks - 1:
                # batch normalize
                x_conv_sc = self.seq[f"bn_conv_blk{b}"](x_conv_sc)

            # for each convolutional layer in the block
            for c in range(self.cfg.n_conv_layers):
                # convolve
                x_conv = self.seq[f"conv_blk{b}_lyr{c}"](x_conv)
                if b != self.cfg.n_conv_blocks - 1 or c != self.cfg.n_conv_layers - 1:
                    # batch normalize
                    x_conv = self.seq[f"bn_conv_blk{b}_lyr{c}"](x_conv)
                # activate
                x_conv = self.seq[f"act_conv_blk{b}_lyr{c}"](x_conv)

            # add shortcut and main conv layers
            x_conv = x_conv_sc + x_conv
            x_conv = self.seq[f"act_conv_blk{b}"](x_conv)

        output = self.seq[f"out"](x_conv)
        return output


class ConvLinNet(nn.Module):
    def __init__(self, ann):
        super().__init__()
        seq = nn.ModuleDict()
        # give sequence dictionary attribute to CNN PyTorch model
        self.seq = seq
        # set AnnConfig object as attribute
        self.cfg = ann.cfg

        # for each convolutional block
        for b in range(ann.cfg.n_conv_blocks):
            # if it's the first convolutional block
            if b == 0:
                n_chnls_in = ann.cfg.conv_dim0
                n_chnls_out = ann.cfg.n_channels
            else:

                n_chnls_in = ann.cfg.n_channels
                n_chnls_out = ann.cfg.n_channels

            # add shortcut
            # no bias because subsequent batch normalization layer makes it redundant
            seq[f"conv_blk{b}"] = nn.Conv1d(
                in_channels=n_chnls_in,
                out_channels=n_chnls_out,
                kernel_size=1,
                padding="same",
                bias=False,
            )
            seq[f"bn_conv_blk{b}"] = nn.BatchNorm1d(n_chnls_out)

            # if using parametric relu as the activation function
            if ann.cfg.activation == "prelu":
                seq[f"act_conv_blk{b}"] = nn.PReLU(n_chnls_out)
            elif ann.cfg.activation == "relu":
                seq[f"act_conv_blk{b}"] = nn.ReLU()
            else:
                raise ValueError("Activation function must be 'prelu' or 'relu'")

            # for each convolutional layer
            for c in range(ann.cfg.n_conv_layers):
                if b == 0 and c == 0:
                    n_chnls_in = ann.cfg.conv_dim0
                    n_chnls_out = ann.cfg.n_channels
                else:
                    n_chnls_in = ann.cfg.n_channels
                    n_chnls_out = ann.cfg.n_channels

                # add convolutional 1D layer
                # no bias because subsequent batch normalization layer makes it redundant
                seq[f"conv_blk{b}_lyr{c}"] = nn.Conv1d(
                    in_channels=n_chnls_in,
                    out_channels=n_chnls_out,
                    kernel_size=ann.cfg.kernels[c],
                    padding="same",
                    bias=False,
                )
                seq[f"bn_conv_blk{b}_lyr{c}"] = nn.BatchNorm1d(n_chnls_out)

                # if using parametric ReLu as the activation function
                if ann.cfg.activation == "prelu":
                    seq[f"act_conv_blk{b}_lyr{c}"] = nn.PReLU(n_chnls_out)
                # if using standard ReLu as the activation function
                elif ann.cfg.activation == "relu":
                    seq[f"act_conv_blk{b}_lyr{c}"] = nn.ReLU()
                else:
                    raise ValueError("Activation function must be 'prelu' or 'relu'")

        for b in range(ann.cfg.n_lin_blocks):
            # if it's the first convolutional block
            if b == 0:
                # no bias because subsequent batch normalization layer makes it redundant
                bias = False
                n_nodes_in = ann.cfg.lin_dim
                n_nodes_out = ann.cfg.n_nodes
            # if it's the last convolutional block
            elif b == ann.cfg.n_lin_blocks - 1:
                # include bias because there is no subsequent batch normalization layer
                bias = True
                n_nodes_in = ann.cfg.n_nodes
                n_nodes_out = ann.cfg.n_nodes
            # if it's any other block
            else:
                # no bias because subsequent batch normalization layer makes it redundant
                bias = False
                n_nodes_in = ann.cfg.n_nodes
                n_nodes_out = ann.cfg.n_nodes

            # add shortcut
            seq[f"lin_blk{b}"] = nn.Linear(
                in_features=n_nodes_in,
                out_features=n_nodes_out,
                bias=bias,
            )

            # batch normalization is not used right before the output layer
            if b != ann.cfg.n_lin_blocks - 1:
                seq[f"bn_lin_blk{b}"] = nn.BatchNorm1d(n_nodes_out)

            # if using parametric relu as the activation function
            if ann.cfg.activation == "prelu":
                seq[f"act_lin_blk{b}"] = nn.PReLU(n_nodes_out)
            elif ann.cfg.activation == "relu":
                seq[f"act_lin_blk{b}"] = nn.ReLU()
            else:
                raise ValueError("Activation function must be 'prelu' or 'relu'")

            for l in range(ann.cfg.n_lin_layers):
                if b == 0 and l == 0:
                    bias = False
                    n_nodes_in = ann.cfg.lin_dim
                    n_nodes_out = ann.cfg.n_nodes
                elif b == ann.cfg.n_lin_blocks - 1 and l == ann.cfg.n_lin_layers - 1:
                    # include bias because there is no subsequent batch normalization layer
                    bias = True
                    n_nodes_in = ann.cfg.n_nodes
                    n_nodes_out = ann.cfg.n_nodes
                else:
                    bias = False
                    n_nodes_in = ann.cfg.n_nodes
                    n_nodes_out = ann.cfg.n_nodes

                # add fully connected layer
                seq[f"lin_blk{b}_lyr{l}"] = nn.Linear(
                    in_features=n_nodes_in,
                    out_features=n_nodes_out,
                    bias=bias,
                )

                # batch normalization is not used right before the output layer
                if b != ann.cfg.n_lin_blocks - 1 or l != ann.cfg.n_lin_layers - 1:
                    seq[f"bn_lin_blk{b}_lyr{l}"] = nn.BatchNorm1d(n_nodes_out)

                # if using parametric ReLu as the activation function
                if ann.cfg.activation == "prelu":
                    seq[f"act_lin_blk{b}_lyr{l}"] = nn.PReLU(n_nodes_out)
                # if using standard ReLu as the activation function
                elif ann.cfg.activation == "relu":
                    seq[f"act_lin_blk{b}_lyr{l}"] = nn.ReLU()
                else:
                    raise ValueError("Activation function must be 'prelu' or 'relu'")

        # add final linear output layer
        seq[f"out"] = nn.Linear(
            in_features=ann.cfg.n_nodes,
            out_features=ann.cfg.out_dim0,
            bias=True,
        )

    def forward(self, x):
        # for each convolutional block
        for b in range(self.cfg.n_conv_blocks):
            # convolve shortcut
            x_sc = self.seq[f"conv_blk{b}"](x)
            # batch normalize
            x_sc = self.seq[f"bn_conv_blk{b}"](x_sc)

            # for each convolutional layer in the block
            for c in range(self.cfg.n_conv_layers):
                # convolve
                x = self.seq[f"conv_blk{b}_lyr{c}"](x)
                # batch normalize
                x = self.seq[f"bn_conv_blk{b}_lyr{c}"](x)
                # activate
                x = self.seq[f"act_conv_blk{b}_lyr{c}"](x)

            # add shortcut and main conv layers
            x = x_sc + x
            x = self.seq[f"act_conv_blk{b}"](x)

        # flatten output of last convolutional layer
        x_lin = torch.flatten(x, start_dim=1)
        # for each fully connected block
        for b in range(self.cfg.n_lin_blocks):
            # shortcut
            x_lin_sc = self.seq[f"lin_blk{b}"](x_lin)
            if b != self.cfg.n_lin_blocks - 1:
                x_lin_sc = self.seq[f"bn_lin_blk{b}"](x_lin_sc)

            # for each fully connected layer in the block
            for l in range(self.cfg.n_lin_layers):
                x_lin = self.seq[f"lin_blk{b}_lyr{l}"](x_lin)
                # batch normalization is not used right before the output layer
                if b != self.cfg.n_lin_blocks - 1 or l != self.cfg.n_lin_layers - 1:
                    x_lin = self.seq[f"bn_lin_blk{b}_lyr{l}"](x_lin)
                x_lin = self.seq[f"act_lin_blk{b}_lyr{l}"](x_lin)

            # add shortcut and main conv layers
            x_lin = x_lin_sc + x_lin
            x_lin = self.seq[f"act_lin_blk{b}"](x_lin)

        # final output layer
        output = self.seq[f"out"](x_lin)

        return output


class NormMSE(nn.Module):
    def __init__(self):
        super(NormMSE, self).__init__()

    def forward(self, outputs, labels):
        # add tiny value to labels to prevent division by zero
        labels = labels + 1e-10
        return torch.mean(((outputs - labels) / labels) ** 2)


class ANN(object):
    def __init__(
        self,
        ann_config,
    ):
        self.cfg = ann_config

    def find_lr(self, ann_datasets):
        start_t = time.time()

        if self.cfg.nn_type == "conv":
            ann_model = ConvNet(self).to(device)
        elif "conv_lin" in self.cfg.nn_type:
            ann_model = ConvLinNet(self).to(device)
        else:
            raise ValueError(
                "nn_type must be 'lin', 'conv', 'conv_lin', or 'conv_lin_inj'"
            )

        optimizer = configure_optimizers(
            ann_model, 1e-5, self.cfg.weight_decay, self.cfg.opt_betas
        )

        # make DataLoaders based on train_set and val_set TensorDatasets.
        # We don't shuffle the data because each batch needs to be resized and/or split into tensors to generate the
        # convolutional time sequence and/or process univariate functions
        train_loader = DataLoader(
            ann_datasets.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        test_loader = DataLoader(
            ann_datasets.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        criterion = nn.MSELoss()
        lr_finder = LRFinder(ann_model, optimizer, criterion, device="cuda")
        # lr_finder.range_test(train_loader, end_lr=1e-3, num_iter=30)
        lr_finder.range_test(
            train_loader, test_loader, end_lr=1e-3, num_iter=25  # , step_mode="linear"
        )
        fig, ax = plt.subplots()
        lr_finder.plot(skip_start=0, skip_end=0, ax=ax, suggest_lr=False)
        # get datafiles directory in current working directory
        datafiles_dir = Path("pred_elem_seq/datafiles")

        fig.savefig(
            datafiles_dir
            / f"figures/learn rate finder/learn_rate_range_finder - {self.cfg.nn_type}.svg"
        )

        elapsed_t = {"learn rate finder time (minutes)": (time.time() - start_t) / 60}
        write_dic_to_file(
            elapsed_t,
            datafiles_dir
            / f"figures/learn rate finder/learn_rate_range_finder - {self.cfg.nn_type}.log",
        )

    def train_test(self, ann_datasets, trial_name="0", pretrn_model=None):
        # if training with a single input tensor (i.e., variables and parameters together)
        def _train():
            # Start training
            for epoch in range(epochs):
                # if using an IterableDataset instead of a TensorDataset
                if iterable_dataset:
                    train_set.set_epoch(epoch)
                # st_time = time.time()
                train_loss = torch.tensor(0.0, device=device)
                for i, (inputs, labels) in enumerate(train_loader):
                    # prevents overflow of steps, which causes an error when using OneCycleLR with
                    # ShardedIterableDataset
                    if i == steps_per_epoch:
                        break
                    # send batches to GPU if not already loaded onto GPU
                    if seq_len > 0:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    # zero the parameter gradients (the parameter gradients are set to None by default)
                    optimizer.zero_grad()
                    # run forward pass with autocasting
                    with autocast(device_type="cuda", dtype=torch.float16):
                        # forward
                        outputs = ann_model(inputs)
                        # calculate loss based on batch
                        loss = criterion(outputs, labels)
                    # scales loss.  Calls backward() on scaled loss to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                    grad_scaler.scale(loss).backward()
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    grad_scaler.step(optimizer)
                    # updates the scaler for next iteration.
                    grad_scaler.update()
                    # step through the learning rate scheduler
                    scheduler.step()
                    # add training loss to running sum
                    train_loss += loss.detach()
                    # prof.step()

                    # when the number of trained rows reaches check_train_loss, print the training loss to the console
                    if (
                        round(((epoch - 1) * steps_per_epoch + (i + 1)) * batch_size)
                        % check_train_loss
                        == 0
                    ):
                        avg_train_loss = train_loss * batch_size / check_train_loss

                        print(
                            f"{epoch:02} - {(i + 1) / steps_per_epoch * 100:05.2f}%\t- train loss: "
                            f"{avg_train_loss:.5}\t- learn rate: {scheduler.get_lr()[0]:.3}"
                        )
                        # print(f"time [s]: {time.time() - st_time}")
                        # st_time = time.time()
                        train_loss = torch.tensor(0.0, device=device)

        # if testing with a single input tensor (i.e., variables and parameters together)
        def _test():
            # instantiate test loss at zero
            test_loss = torch.tensor(0.0, device=device)

            # set ann model's batch normalization layers to evaluation mode
            ann_model.eval()

            for k, (inputs, labels) in enumerate(test_loader):
                # send batches to GPU, if not already loaded
                if seq_len > 0:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                # forward
                with torch.no_grad():
                    outputs = ann_model(inputs)
                # calculate mean squared error
                test_loss += criterion(outputs, labels).detach()

            # record loss from test dataset to grid search dataframe
            finl_lss = test_loss.item() / steps_per_epoch
            return finl_lss

        run_st_time = time.time()

        # copy over ANN attributes to make code cleaner
        building = self.cfg.building
        nn_type = self.cfg.nn_type
        loss_func = self.cfg.loss_func
        pretrained = self.cfg.pretrained
        solver = self.cfg.solver
        div_f_lr = self.cfg.div_f_lr
        fin_div_f_lr = self.cfg.fin_div_f_lr
        pct_start = self.cfg.pct_start
        max_lr = self.cfg.max_lr
        opt_betas = self.cfg.opt_betas
        weight_decay = self.cfg.weight_decay
        batch_size = self.cfg.batch_size
        check_train_loss = self.cfg.check_train_loss
        n_lhs = self.cfg.n_lhs
        seq_len = self.cfg.seq_len
        pred_len = self.cfg.pred_len
        n_nodes = self.cfg.n_nodes
        n_lin_layers = self.cfg.n_lin_layers
        n_lin_blocks = self.cfg.n_lin_blocks
        n_channels = self.cfg.n_channels
        n_conv_layers = self.cfg.n_conv_layers
        n_conv_blocks = self.cfg.n_conv_blocks
        kernels = self.cfg.kernels
        epochs = self.cfg.epochs

        # copy over ann_datasets attributes to make code cleaner
        train_set = ann_datasets.train_set
        test_set = ann_datasets.test_set

        if nn_type == "conv":
            # instantiate a convolutional neural network object, where the variables and parameters are all
            # convolved (no linear fully connected layers)
            ann_model = ConvNet(self).to(device)
        elif "conv_lin" in nn_type:
            # instantiate a convolutional neural network object with linear layers, where the parameters are either
            # part of the convolved inputs or injected just before the linear layers
            ann_model = ConvLinNet(self).to(device)
        else:
            raise ValueError(
                "nn_type must be 'lin', 'conv', 'conv_lin', or 'conv_lin_inj'"
            )

        # use the AdamW solver for now
        if solver == "adamW":
            init_lr = max_lr / div_f_lr
            optimizer = configure_optimizers(
                ann_model, init_lr, weight_decay, opt_betas
            )
        else:
            # todo: add other optimizers
            raise NameError("must add other optimizers")

        # if tensors are already loaded onto GPU, don't pin_memory
        if seq_len == 0:
            pin_memory = False
        # otherwise, pin dense CPU tensors to memory
        else:
            pin_memory = True

        if isinstance(train_set, ShardedIterableDataset):
            iterable_dataset = True
            persistent_workers = True
        else:
            iterable_dataset = False
            persistent_workers = False

        # make DataLoaders based on train_set TensorDataset.
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=os.cpu_count(),
            persistent_workers=persistent_workers,
            drop_last=True,
        )

        if isinstance(train_set, ShardedIterableDataset):
            steps_per_epoch = math.ceil(train_set.n_samples / batch_size)
        else:
            steps_per_epoch = len(train_loader)

        # use one cycle learn rate scheduler based on Leslie Smith's super-convergence article
        scheduler = OneCycleLR(
            optimizer,
            max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=pct_start,
            div_factor=div_f_lr,
            final_div_factor=fin_div_f_lr,
        )

        if loss_func == "mse":
            criterion = nn.MSELoss()
        elif loss_func == "norm_mse":
            criterion = NormMSE()
        else:
            raise NameError("must add other loss functions")

        print(
            f"#####\ntrial: {trial_name}\nepochs: {epochs}\nmax learn rate: {max_lr}\ndiv factor: {div_f_lr}\n"
            f"final div factor: {fin_div_f_lr}\nweight decay: {weight_decay}\n"
            f"batch size: {batch_size}\nlin layers: {n_nodes} X {n_lin_layers} X {n_lin_blocks}\n"
            f"conv layers: {n_channels} X {n_conv_layers} X {n_conv_blocks}\n"
            f"kernels: {kernels} X {n_conv_blocks}\nseq len: {seq_len}\npred len: {pred_len}\n#####"
        )

        # create a GradScaler once at the beginning of training.
        grad_scaler = GradScaler("cuda")

        # train ANN model
        _train()

        # write ann_model to file
        ann_dir_path = Path("pred_elem_seq/datafiles/ann")
        pretrained_path = "pre_"
        ext_path = ".sav"

        if nn_type == "conv":
            ann_type_path = (
                f"{nn_type} - {building.name} - seq_{seq_len} - pred_{pred_len} - n_lhs_{n_lhs} - dec_{weight_decay} - "
                f"conv_{n_channels}X{n_conv_layers}X{n_conv_blocks} - krnl_{kernels} - trial_{trial_name}"
            )
        elif "conv_lin" in nn_type:
            ann_type_path = (
                f"{nn_type} - {building.name} - seq_{seq_len} - pred_{pred_len} - n_lhs_{n_lhs} - dec_{weight_decay} - "
                f"lin_{n_nodes}X{n_lin_layers}X{n_lin_blocks} - conv_{n_channels}X{n_conv_layers}X{n_conv_blocks} - "
                f"krnl_{kernels} - trial_{trial_name}"
            )
        else:
            raise ValueError(
                "nn_type must be 'lin', 'conv', 'conv_lin', or 'conv_lin_inj'"
            )

        # save neural network model as binary .sav file
        ann_model_path = ann_dir_path / (ann_type_path + ext_path)

        # dump PyTorch model to pickle file
        with open(ann_model_path, "wb") as fw:
            pickle.dump(ann_model, fw)

        # if a testing dataset is provided, test the trained ANN
        if isinstance(test_set, TensorDataset) or isinstance(
            test_set, ShardedIterableDataset
        ):
            print(
                f"#####\nSTART TESTING FOR:\ntrial: {trial_name}\nepochs: {epochs}\nmax learn rate: {max_lr}\n"
                f"div factor: {div_f_lr}\nfinal div factor: {fin_div_f_lr}\nweight decay: {weight_decay}\n"
                f"batch size: {batch_size}\nlin layers: {n_nodes} X {n_lin_layers} X {n_lin_blocks}\n"
                f"conv layers: {n_channels} X {n_conv_layers} X {n_conv_blocks}\n"
                f"kernels: {kernels} X {n_conv_blocks}\nseq len: {seq_len}\npred len: {pred_len}\n#####"
            )

            if isinstance(test_set, ShardedIterableDataset):
                iterable_dataset = True
                persistent_workers = True

            else:
                iterable_dataset = False
                persistent_workers = False
            # make DataLoader based on test_set TensorDataset.
            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=os.cpu_count(),
                persistent_workers=persistent_workers,
            )

            if isinstance(test_set, ShardedIterableDataset):
                steps_per_epoch = math.ceil(test_set.n_samples / batch_size)
            else:
                steps_per_epoch = len(test_loader)

            # get final loss
            final_loss = _test()
            print(f"Final loss -\t\t{final_loss:.3f}")

        else:
            final_loss = None

        # record training + validation_1 + validation_2 time to grid search dataframe
        run_time = (time.time() - run_st_time) / 3600
        print(f"run time: {run_time}")

        res_path = "res_"
        ext_path = ".json"

        # save neural network model as binary .sav file
        if pretrained:
            res_model_path = ann_dir_path / (
                res_path + pretrained_path + ann_type_path + ext_path
            )
        else:
            res_model_path = ann_dir_path / (res_path + ann_type_path + ext_path)

        # create surrogate model training results dictionary
        dict_res = {
            "final_loss": final_loss,
            "run_time": run_time,
        }
        # include all attributes (except for building object) of the AnnConfig object in the results dictionary
        ann_cfg_attr = copy.deepcopy(vars(self.cfg))
        del ann_cfg_attr["building"]
        dict_res.update(ann_cfg_attr)
        # write results dictionary to JSON file
        write_dic_to_file(dict_res, res_model_path)

        return {"run_time": run_time, "loss": final_loss, "model": ann_model}


class PartialStandardScaler(StandardScaler):
    def transform(self, X, columns=None):
        if columns is None:
            return super().transform(X)
        else:
            if isinstance(X, pd.DataFrame):
                return (X.loc[:, columns] - self.mean_[columns]) / self.scale_[columns]

            elif isinstance(X, np.ndarray):
                return (X[:, columns] - self.mean_[columns]) / self.scale_[columns]
            else:
                raise ValueError("X must be a pandas.DataFrame or numpy.ndarray")

    def inverse_transform(self, X, columns=None):
        if columns is None:
            return super().inverse_transform(X)
        if X.ndim == 1:
            return self.mean_[columns] + X[columns] * self.scale_[columns]
        elif X.ndim == 2:
            return self.mean_[columns] + X[:, columns] * self.scale_[columns]
        else:
            raise ValueError("X must be a 1D or 2D numpy.ndarray")


class AnnDatasets(object):
    def __init__(self, ann_config, dataset_obj_names, sim_precision="float32"):

        self.cfg = ann_config
        self.dataset_obj_names = dataset_obj_names
        self.n_dataset_objs = len(dataset_obj_names)
        self.sim_precision = sim_precision
        # if the design of experiment is a univariate-focused one
        if self.cfg.n_lhs_uni != 0:
            # keep univariate batches together when shuffling and partitioning datasets
            self.shuffle_group = self.cfg.n_smp_uni * self.cfg.seq_len
        else:
            # keep rolled inputs together when shuffling and partitioning datasets
            self.shuffle_group = self.cfg.seq_len
        self.train_set = None
        self.test_set = None
        self.cross_val_set = None
        self.train_multidx = None
        self.test_multidx = None
        self.cross_val_multidx = None

    @staticmethod
    def get_lhs_factors(param_dim, smp_dim):
        # check if the square root of n_lhs is a prime number. If so, run an orthogonal-array-based LHS design
        try:
            sampler = LatinHypercube(d=param_dim, strength=2)
            factors = sampler.random(smp_dim)
        # otherwise, run a simple random LHS design
        except ValueError:
            sampler = LatinHypercube(d=param_dim, strength=1)
            factors = sampler.random(smp_dim)

        return factors
    
    @staticmethod
    def get_doe_chunks(res_ch, idx_ch):
        doe_ch = res_ch.loc[idx_ch]
        return doe_ch

    def orth_lhs(
        self,
        weather_list=None,
        samp_per_weath=None,
    ):
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building

        # if only the unknown parameters are included when generating the LHS factors (single weather)
        if weather_list is None:
            fac_params = building.unknown_params
        # if the unknown building parameters and different simulation weather are included
        else:
            fac_params = {
                "weather_idx": {"lower_bound": 0, "upper_bound": len(weather_list) - 1}
            } | building.unknown_params

        # get factors of orthogonal-array-based LHS design based
        factors = self.get_lhs_factors(len(fac_params), self.cfg.n_lhs)
        # get lower and upper bounds of features in LHS design
        low_bnd = np.array([param["lower_bound"] for param in fac_params.values()])
        up_bnd = np.array([param["upper_bound"] for param in fac_params.values()])

        # if only the unknown parameters are part of the LHS design (single weather)
        if weather_list is None:
            # the DoE data only has the unknown parameter values
            doe_params_data = factors * (up_bnd - low_bnd) + low_bnd
            # get the DoE parameter columns names
            doe_params_cols = building.unknown_param_names
            # create iterables for additional argument in parallel building.evaluate call
            weather_sim_list = [None for _ in range(doe_params_data.shape[0])]
        # if the unknown building parameters and different simulation weather are included
        else:
            # the sampled unknown parameter values are continuous
            doe_params_data = factors[:, 1:] * (up_bnd[1:] - low_bnd[1:]) + low_bnd[1:]
            # if running samp_per_weath building simulation for each weather file in weather_list
            if isinstance(samp_per_weath, int):
                assert self.cfg.n_lhs == samp_per_weath * len(weather_list), (
                    "if running `samp_per_weath` building simulations for each weather file "
                    "in weather_list, ann_config.n_lhs must be equal to `samp_per_weath` * len(weather_list)"
                )
                weather_indcs = np.arange(len(weather_list)).repeat(samp_per_weath)
            # otherwise if randomly sampling from weather_list
            else:
                # get the integer indices of the buildings/geojsons in the city_hub_list by rounding the scaled factors to
                # the nearest integer
                weather_indcs = np.round(
                    factors[:, 0] * (up_bnd[0] - low_bnd[0]) + low_bnd[0]
                ).astype(int)
            # get the weather file paths as strings from the weather_list
            weather_sim_list = [weather_list[idx] for idx in weather_indcs]
            # get the DoE parameter columns names
            doe_params_cols = building.unknown_param_names

        # precision of doe parameter values must be float64 because float32 is not json serializable (parameter values
        # are serialized during simulation). However, first downsample to float32, then upsample so that the ANN learns
        # on the correct input values
        doe_params = pd.DataFrame(
            data=doe_params_data,
            columns=doe_params_cols,
            dtype="float32",
        ).astype("float64")

        # clear simulation folders
        clear_sim_folders()

        # run IDF file with doe parameter values
        pool = Pool()
        # all simulated energy results are saved to a list
        sim_energies_list = list(
            pool.map(
                building.evaluate,
                doe_params.values,
                weather_sim_list,
            )
        )
        pool.shutdown()

        # concatenate simulation results into one dataframe
        sim_energies = pd.DataFrame(
            data=pd.concat(sim_energies_list, ignore_index=True),
            columns=self.dataset_obj_names,
            dtype=self.sim_precision,
        )
        # cast float64 doe parameter values to float32 to save disk space
        for col in doe_params.columns:
            doe_params.loc[:, col] = doe_params.loc[:, col].astype("float32")

        # insert weather paths in design of experiment parameter values
        if weather_list:
            doe_params.insert(
                0, "weather_path", pd.Series(weather_sim_list, dtype="string")
            )

        return sim_energies, doe_params

    def uni_lhs(self):
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building

        # get factors of simple random LHS design based
        factors = self.get_lhs_factors(building.n_unknown_params, self.cfg.n_lhs_uni)

        # get starting points dataframe with orthogonal-array-based LHS design
        low_bnd = np.array(
            [param["lower_bound"] for param in building.unknown_params.values()]
        )
        up_bnd = np.array(
            [param["upper_bound"] for param in building.unknown_params.values()]
        )
        # precision of doe parameter values must be float64 because float32 is not json serializable (parameter values
        # are serialized during simulation)
        starting_pts = pd.DataFrame(
            data=factors * (up_bnd - low_bnd) + low_bnd,
            columns=building.unknown_param_names,
            dtype="float64",
        )
        # instantiate dataframe that will hold all DoE parameter values
        doe_params = pd.DataFrame(
            index=range(
                self.cfg.n_lhs_uni * building.n_unknown_params * self.cfg.n_smp_uni
            ),
            columns=building.unknown_param_names,
            dtype="float64",
        )

        # for each starting point
        for s_p in range(self.cfg.n_lhs_uni):
            # for each unknown building parameter, generate univariate sample
            for p, (par_name, param) in enumerate(building.unknown_params.items()):
                # create dataframe with building parameter values in starting point repeated n_smp_uni times
                inputs = starting_pts.iloc[[s_p] * self.cfg.n_smp_uni]
                if self.cfg.n_smp_uni > 2:
                    # change the univariate building parameter values as a linear sweep between its lower and upper bound
                    inputs[par_name] = np.linspace(
                        param["lower_bound"],
                        param["upper_bound"],
                        self.cfg.n_smp_uni,
                    )
                elif self.cfg.n_smp_uni == 2:
                    # sample two random points between the lower and upper bounds of the parameter
                    inputs[par_name] = (
                        np.sort(np.random.rand(self.cfg.n_smp_uni))
                        * (param["upper_bound"] - param["lower_bound"])
                        + param["lower_bound"]
                    )
                else:
                    raise ValueError("n_smp_uni must be equal to or greater than 2")

                # populate doe_params
                doe_params.iloc[
                    (s_p * building.n_unknown_params + p)
                    * self.cfg.n_smp_uni : (s_p * building.n_unknown_params + p + 1)
                    * self.cfg.n_smp_uni
                ] = inputs

        # clear simulation folders
        clear_sim_folders()

        # run IDF file with doe parameter values
        pool = Pool()
        # all simulated energy results are saved to a list
        sim_energies_lst = list(pool.map(building.evaluate, doe_params.values))
        pool.shutdown()
        # concatenate simulation results into one dataframe
        sim_energies = pd.DataFrame(
            data=pd.concat(sim_energies_lst, ignore_index=True),
            dtype=self.sim_precision,
        )
        # cast float64 doe parameter values to float32 to save disk space
        for col in doe_params.columns:
            doe_params[col] = doe_params[col].astype("float32")

        return sim_energies, doe_params
    
    def get_sim_time_idx_rolls(self):
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building

        # append the first index `seq_len - 1` times to the beginning of building.sim_time_idx
        sim_time_idx_rep = pd.Index(
            [building.sim_time_idx[0] for _ in range(self.cfg.seq_len - 1)]
        ).append(building.sim_time_idx)
        # slide a window of seq_len hours across sim_time_idx_rep and flatten it
        sim_time_idx_rolls = pd.DatetimeIndex(
            sliding_window_view(sim_time_idx_rep, self.cfg.seq_len).flatten()
        )
        return sim_time_idx_rolls

    def get_sim_time_idx_rolls_uni(self):
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building

        # append the first index `seq_len - 1` times to the beginning of building.sim_time_idx
        sim_time_idx_rep = pd.Index(
            [building.sim_time_idx[0] for _ in range(self.cfg.seq_len - 1)]
        ).append(building.sim_time_idx)
        # slide a window of seq_len hours across sim_time_idx_rep and flatten it
        sim_time_idx_rolls = sliding_window_view(
            sim_time_idx_rep, self.cfg.seq_len
        ).flatten()
        # repeat seq_len chunks of the reshaped rolled array n_smp_uni times
        sim_idx_rolls_3d = np.repeat(
            sim_time_idx_rolls.reshape((-1, self.cfg.seq_len))[:, :, None],
            self.cfg.n_smp_uni,
            axis=2,
        )
        # transpose 3D repeated array so that when it is flattened, the index is in the correct order
        sim_time_idx_rolls_uni = pd.DatetimeIndex(
            sim_idx_rolls_3d.transpose((0, 2, 1)).flatten()
        )
        return sim_time_idx_rolls_uni

    def get_time_ser_doe(self, sim_energies, doe_params):
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building

        # repeat design of experiment parameters by number of indices in building.sim_time_idx
        doe_params_rep = doe_params.iloc[
            np.arange(self.cfg.n_lhs).repeat(len(building.sim_time_idx))
        ].reset_index(drop=True)
        # tile building variables data based on number of samples in DoE
        varbls = building.var_data.iloc[
            np.tile(np.arange(len(building.sim_time_idx)), self.cfg.n_lhs)
        ].reset_index(drop=True)
        # concatenate objectives and parameters together into DoEdataframe
        doe = pd.concat([sim_energies, varbls, doe_params_rep], axis=1)
        # get MultiIndex of design of experiment
        multi_idx = pd.MultiIndex.from_arrays(
            [
                pd.DatetimeIndex(np.tile(building.sim_time_idx, self.cfg.n_lhs)),
                np.repeat(np.arange(self.cfg.n_lhs), len(building.sim_time_idx)),
            ],
            names=self.cfg.multidx_names,
        )
        doe.index = multi_idx
        # reset index of doe so that we have record of the MultiIndex
        doe = doe.reset_index(names=self.cfg.multidx_names)
        return doe

    def get_tensor_datasets(
        self,
        sim_energies,
        doe_params,
        cross_val_indcs,
        shuffle_doe=True,
        scaler=None,
        scaler_path=None,
        load_to_gpu=False,
    ):
        start_t = time.time()
        # if the datasets are for individual hour predictions
        if self.cfg.pred_len == 1:
            # get DoE from outputs (sim_energies) and inputs (doe_params) by rolling inputs
            if self.cfg.n_lhs_uni != 0:
                doe = self.get_indvl_hour_uni_doe(sim_energies, doe_params)
            else:
                doe = self.get_indvl_hour_doe(sim_energies, doe_params, False)
        # if datasets are for time series prediction (i.e., the hourly consumption over the year)
        else:
            doe = self.get_time_ser_doe(sim_energies, doe_params)

        print(
            f"done getting full design of experiment in {(time.time() - start_t)/60:.1f} mins"
        )

        if shuffle_doe:
            # shuffle groups of batch_size rows in the DoE dataframe. During training, each batch will be resized
            # into a 3D Tensor, with the time sequence as the 2nd dimension (for convolutional NN),
            # or into a 2D Tensor, with the variables rolled column-wise (for linear NN).
            rng = np.random.default_rng(0)
            all_i_arr = np.arange(doe.shape[0])
            rng.shuffle(all_i_arr.reshape(-1, self.shuffle_group))
            # shuffle doe dataframe
            doe = doe.iloc[all_i_arr]
            # del all_i_arr
            print(
                f"done shuffling design of experiment in {(time.time() - start_t)/60:.1f} minutes"
            )

        # if there is a training data (in addition to testing data), and we are not doing a k-folds cross validation
        if self.cfg.train_perc != 0 and cross_val_indcs is None:
            # get training indices
            train_n = self.shuffle_group * round(
                self.cfg.train_perc * len(doe) / self.shuffle_group
            )
            train_idx = range(0, train_n)
            # get testing indices
            test_idx = range(train_n, len(doe))
        # if we are doing k-folds cross validation
        elif isinstance(cross_val_indcs, dict):
            train_idx = cross_val_indcs["train_idx"]
            test_idx = cross_val_indcs["test_idx"]
        # if there is only testing data
        else:
            train_idx = None
            test_idx = range(len(doe))

        # number of multidx columns
        n_idx_cols = len(self.cfg.multidx_names)

        # if there is training data
        if isinstance(train_idx, range):
            # fit StandardScaler to training data that removes the mean and scales to unit variance
            # PartialStandardScaler is a custom class that can transform a subset of features
            scaler = PartialStandardScaler()
            scaler.fit(
                doe.iloc[
                    train_idx,
                    n_idx_cols + self.n_dataset_objs :,
                ]
            )
            # write scaler to file
            with open(scaler_path, "wb") as fw:
                pickle.dump(scaler, fw)
        elif scaler is None:
            raise ValueError(
                "scaler must be provided if there is no training dataset (i.e., train_perc = 0)"
            )

        xy_tensors = self.get_xy_tensors(doe, scaler, train_idx, test_idx)
        # get indices of prediction objectives (i.e., "heat", "cool", etc.)
        obj_indcs = [
            self.dataset_obj_names.index(obj) for obj in self.cfg.building.obj_names
        ]
        # index into the labels based on the prediction objectives
        if self.cfg.train_perc != 0:
            xy_tensors["y_train"] = xy_tensors["y_train"][:, obj_indcs]
            print("number of y columns = ", xy_tensors["y_train"].shape[1])
        elif self.cfg.train_perc != 1:
            xy_tensors["y_test"] = xy_tensors["y_test"][:, obj_indcs]
            print("number of y columns = ", xy_tensors["y_test"].shape[1])

        # if there is a training dataset
        if self.cfg.train_perc != 0:
            # assign training dataset to AnnDatasets attribute
            self.train_set = TensorDataset(
                xy_tensors["x1_train"], xy_tensors["y_train"]
            )
            # add training multiindex
            # the row indices skip by seq_len, starting at seq_len - 1
            self.train_multidx = doe.iloc[
                train_idx[self.cfg.seq_len - 1 :: self.cfg.seq_len],
                :n_idx_cols,
            ]
        # if there is a testing dataset
        if self.cfg.train_perc != 1:
            # assign testing dataset to AnnDatasets attribute
            self.test_set = TensorDataset(
                xy_tensors["x1_test"], xy_tensors["y_test"]
            )
            # add testing multiindex
            # the row indices skip by seq_len, starting at seq_len - 1
            self.test_multidx = doe.iloc[
                test_idx[self.cfg.seq_len - 1 :: self.cfg.seq_len], :n_idx_cols
            ]

        # if the dataset is small enough to load directly to GPU
        if load_to_gpu:
            if self.cfg.train_perc != 0:
                self.train_set = TensorDataset(
                    self.train_set.tensors[0].to(device),
                    self.train_set.tensors[1].to(device),
                )
            self.test_set = TensorDataset(
                self.test_set.tensors[0].to(device),
                self.test_set.tensors[1].to(device),
            )

    def get_xy_tensors(
        self,
        doe: DataFrame,
        scaler: PartialStandardScaler,
        train_idx: range | None,
        test_idx: range,
    ) -> dict:

        # get building object from ann_config for easy access
        building = self.cfg.building
        # get total number of building parameters
        n_all_params = building.n_unknown_params
        # number of multidx columns
        n_idx_cols = len(self.cfg.multidx_names)

        # if the neural network type is convolutional and the parameters are part of the starting inputs of the NN
        # x1 are reshaped variables and parameters (n_samples X [n_vars + n_params] X seq_len)
        if self.cfg.nn_type in ("conv_lin", "conv"):
            # if there is a training dataset
            if self.cfg.train_perc != 0:
                x1_train = torch.tensor(
                    scaler.transform(
                        doe.iloc[train_idx, n_idx_cols + self.n_dataset_objs :]
                    )
                    .reshape(
                        (
                            -1,
                            self.cfg.seq_len,
                            building.n_vars + n_all_params,
                        ),
                    )
                    .transpose((0, 2, 1)),
                    dtype=torch.float,
                )
            else:
                x1_train = None

            # if there is a testing dataset
            if self.cfg.train_perc != 1:
                x1_test = torch.tensor(
                    scaler.transform(
                        doe.iloc[test_idx, n_idx_cols + self.n_dataset_objs :]
                    )
                    .reshape(
                        (
                            -1,
                            self.cfg.seq_len,
                            building.n_vars + n_all_params,
                        ),
                    )
                    .transpose((0, 2, 1)),
                    dtype=torch.float,
                )
            else:
                x1_test = None

            x2_train = None
            x2_test = None
        else:
            raise ValueError(
                "nn_type must be 'conv' or 'conv_lin'"
            )

        # if predicting individual hours
        # labels are the last sim energy rows in the sequences (i.e., current hour) (n_samples x n_objs)
        if self.cfg.pred_len == 1:
            # if there is a training dataset
            if self.cfg.train_perc != 0:
                y_train = torch.tensor(
                    # the row indices skip by seq_len, starting at seq_len - 1
                    doe.iloc[
                        train_idx[self.cfg.seq_len - 1 :: self.cfg.seq_len],
                        n_idx_cols : n_idx_cols + self.n_dataset_objs,
                    ].values,
                    dtype=torch.float,
                )
            else:
                y_train = None

            # if there is a testing dataset
            if self.cfg.train_perc != 1:
                y_test = torch.tensor(
                    # the row indices skip by seq_len, starting at seq_len - 1
                    doe.iloc[
                        test_idx[self.cfg.seq_len - 1 :: self.cfg.seq_len],
                        n_idx_cols : n_idx_cols + self.n_dataset_objs,
                    ].values,
                    dtype=torch.float,
                )
            else:
                y_test = None
        # if predicting time series energy
        # labels are the reshaped sim energy (n_samples x n_objs x seq_len)
        else:
            # if there is a training dataset
            if self.cfg.train_perc != 0:
                y_train = torch.tensor(
                    doe.iloc[train_idx, n_idx_cols : n_idx_cols + self.n_dataset_objs]
                    .values.reshape(
                        (-1, self.cfg.seq_len, building.n_objs),
                    )
                    .transpose((0, 2, 1)),
                    dtype=torch.float,
                )
            else:
                y_train = None

            # if there is a testing dataset
            if self.cfg.train_perc != 1:
                y_test = torch.tensor(
                    doe.iloc[test_idx, n_idx_cols : n_idx_cols + self.n_dataset_objs]
                    .values.reshape(
                        (-1, self.cfg.seq_len, building.n_objs),
                    )
                    .transpose((0, 2, 1)),
                    dtype=torch.float,
                )
            else:
                y_test = None

        # return features and labels
        # there are two feature tensors (x1: vars and x2: params) if the neural network type is convolutional and the
        # parameters are injected into the NN after the convolutional layers
        return {
            "x1_train": x1_train,
            "x1_test": x1_test,
            "x2_train": x2_train,
            "x2_test": x2_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_indvl_hour_doe(self, sim_energies, doe_params, parll_indexing=False):
        # get number of EnerrgyPlus simulations
        n_lhs = doe_params.shape[0]
        # get start and end simulation numbers/indices
        lhs_start = doe_params.index[0]
        lhs_end = doe_params.index[-1] + 1
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building
        # get datetimeindex with rolled indices based on sequence length
        sim_time_idx_rolls = self.get_sim_time_idx_rolls()

        # instantiate design of experiment MultiIndex based on time and number of LHS simulation (n_lhs)
        doe_index = pd.MultiIndex.from_arrays(
            [
                pd.DatetimeIndex(np.tile(sim_time_idx_rolls, n_lhs)),
                np.arange(lhs_start, lhs_end).repeat(len(sim_time_idx_rolls)),
            ],
            names=self.cfg.multidx_names,
        )
        # get objectives, variables, and building parameters from LHS DoE
        objs = sim_energies.reset_index(drop=True)
        # if different weather is run during the simulations
        if "weather_path" in doe_params.columns:
            varbls = pd.concat(
                [
                    pd.read_csv(
                        Path(
                            f"pred_elem_seq/datafiles/inputs/variables/weather_invariant/{building.name}_{Path(weather_path).stem}.csv"
                        ),
                        index_col=0,
                    )
                    for weather_path in doe_params.loc[:, "weather_path"]
                ],
                axis=0,
            ).reset_index(drop=True)
        else:
            varbls = building.var_data.iloc[
                np.tile(np.arange(len(building.sim_time_idx)), n_lhs)
            ].reset_index(drop=True)
        # if different weather is run during the simulations
        if "weather_path" in doe_params.columns:
            params = doe_params.iloc[
                np.arange(n_lhs).repeat(len(building.sim_time_idx)), 1:
            ].reset_index(drop=True)
        else:
            params = doe_params.iloc[
                np.arange(n_lhs).repeat(len(building.sim_time_idx))
            ].reset_index(drop=True)
        # concatenate objectives, variables, and parameters together into results dataframe
        res = pd.concat([objs, varbls, params], axis=1)
        # set Multi Index for results data based on time and lhs indices
        res.index = pd.MultiIndex.from_arrays(
            [
                pd.DatetimeIndex(np.tile(building.sim_time_idx, n_lhs)),
                np.arange(lhs_start, lhs_end).repeat(len(building.sim_time_idx)),
            ]
        )

        # parallelize the indexing
        if parll_indexing:
            # split results dataframe and DoE index into chunks to save memory and process in parallel
            res_chunks = np.split(res, n_lhs)
            doe_idx_chunks = np.split(doe_index, n_lhs)
            # delete intermediary variables to save memory
            # del res
            pool = Pool()
            # index into results dataframe with sliced index, in order to get right number and order
            # of rows. Assign reordered results dataframe to doe dataframe
            doe_chunks = pool.map(get_doe_chunks, res_chunks, doe_idx_chunks)
            pool.shutdown()
            doe = pd.concat(list(doe_chunks))
        # index into res dataframe on one cpu
        else:
            doe = res.loc[doe_index]

        # reset index of doe so that we have record of the MultiIndex (shuffling a MultiIndex automatically sorts
        # the MultiIndex, which is not what we want. We want to maintain groups of self.cfg.seq_len rows)
        doe = doe.reset_index(names=self.cfg.multidx_names)

        return doe

    def get_indvl_hour_uni_doe(self, sim_energies, doe_params):
        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building
        # get number of building parameters
        n_all_params = building.n_unknown_params
        # get rolled datetimeindex indices for univariate doe
        sim_time_idx_rolls_uni = self.get_sim_time_idx_rolls_uni()
        # get overall DoE index with kept hours (use parallel processing to go faster)
        doe_idx = pd.MultiIndex.from_arrays(
            [
                np.arange(self.cfg.n_lhs_uni).repeat(
                    len(sim_time_idx_rolls_uni) * n_all_params
                ),
                np.tile(
                    np.arange(n_all_params).repeat(len(sim_time_idx_rolls_uni)),
                    self.cfg.n_lhs_uni,
                ),
                pd.DatetimeIndex(
                    np.tile(sim_time_idx_rolls_uni, self.cfg.n_lhs_uni * n_all_params)
                ),
                np.tile(
                    np.arange(self.cfg.n_smp_uni).repeat(self.cfg.seq_len),
                    len(building.sim_time_idx) * self.cfg.n_lhs_uni * n_all_params,
                ),
            ],
            names=self.cfg.multidx_names,
        )
        # get objectives, variables, and building parameters from univariate LHS DoE
        objs = sim_energies
        # get weather, schedule, and time-based variables as dataframe
        vars_df = building.var_data
        varbls = vars_df.iloc[
            np.tile(
                np.arange(len(building.sim_time_idx)),
                self.cfg.n_lhs_uni * n_all_params * self.cfg.n_smp_uni,
            )
        ].reset_index(drop=True)
        params = doe_params.iloc[
            np.arange(self.cfg.n_lhs_uni * n_all_params * self.cfg.n_smp_uni).repeat(
                len(building.sim_time_idx)
            )
        ].reset_index(drop=True)
        # concatenate objectives, variables, and parameters together into results dataframe
        res = pd.concat([objs, varbls, params], axis=1)
        # set MultiIndex for results dataframe based on time, param, lhs, and smp_uni indices
        res.index = pd.MultiIndex.from_arrays(
            [
                np.arange(self.cfg.n_lhs_uni).repeat(
                    len(building.sim_time_idx) * n_all_params * self.cfg.n_smp_uni
                ),
                np.tile(
                    np.arange(n_all_params).repeat(
                        len(building.sim_time_idx) * self.cfg.n_smp_uni
                    ),
                    self.cfg.n_lhs_uni,
                ),
                pd.DatetimeIndex(
                    np.tile(
                        building.sim_time_idx,
                        self.cfg.n_lhs_uni * n_all_params * self.cfg.n_smp_uni,
                    )
                ),
                np.tile(
                    np.arange(self.cfg.n_smp_uni).repeat(len(building.sim_time_idx)),
                    self.cfg.n_lhs_uni * n_all_params,
                ),
            ]
        )
        # split all hours results dataframe and DoE index into chunks to save memory and process in parallel
        res_chunks = np.split(res, self.cfg.n_lhs_uni)
        doe_idx_chunks = np.split(doe_idx, self.cfg.n_lhs_uni)
        # delete intermediary variables to save memory
        # del res
        # del doe_idx
        pool = Pool()
        # index into results dataframe with sliced index, in order to get right number and order
        # of rows. Assign reordered results dataframe to doe dataframe
        doe_chunks = pool.map(get_doe_chunks, res_chunks, doe_idx_chunks)
        pool.shutdown()
        doe = pd.concat(list(doe_chunks))
        # reset index of doe so that we have record of the MultiIndex (shuffling a MultiIndex automatically sorts
        # the MultiIndex, which is not what we want. We want to maintain groups of n_smp_uni * seq_len rows)
        doe = doe.reset_index()

        return doe

    def get_shard_datasets(
        self,
        sim_energies,
        doe_params,
        shuffle_doe=True,
        scaler=None,
        scaler_path=None,
        save_shards=False,
        shard_size=500,
    ):
        # if the shards are to be saved to disk
        if save_shards:
            if self.cfg.train_perc != 0 and scaler is None:
                #  instantiate StandardScaler for training data that removes the mean and scales to unit variance
                # PartialStandardScaler is a custom class that can transform a subset of features
                scaler = PartialStandardScaler()
            elif self.cfg.train_perc != 0 and scaler is not None:
                raise ValueError(
                    "scaler should not be provided if there is a training datasets (i.e., train_perc > 0)"
                )
            elif self.cfg.train_perc == 0 and scaler is None:
                raise ValueError(
                    "scaler must be provided if there is only a testing dataset (i.e., train_perc = 0)"
                )

            # if the datasets are for individual hour predictions
            if self.cfg.pred_len == 1:
                # get DoE from outputs (sim_energies) and inputs (doe_params) by rolling inputs
                if self.cfg.n_lhs_uni != 0:
                    # todo: implement this univariate method
                    self.get_indvl_hour_uni_shards(sim_energies, doe_params)
                else:
                    self.get_indvl_hour_shards(
                        sim_energies, doe_params, scaler, shuffle_doe, shard_size
                    )
            # if datasets are for time series prediction (i.e., the hourly consumption over the year)
            else:
                # todo: implement this time series method
                self.get_time_ser_shards(sim_energies, doe_params)

            if self.cfg.train_perc != 0:
                # write scaler to file
                with open(scaler_path, "wb") as fw:
                    pickle.dump(scaler, fw)

        if self.cfg.train_perc != 0:
            # get training and testing ShardedIterableDatasets
            # get training shard paths
            shard_dir_train = Path(
                f"pred_elem_seq/datafiles/xy/shards/train/n_lhs_{self.cfg.n_lhs}"
            )
            input_paths_train, label_paths_train = self.get_shard_path_lists(
                shard_dir_train
            )
            # get training ShardedIterableDataset
            self.train_set = ShardedIterableDataset(
                input_paths_train, label_paths_train
            )

        if self.cfg.train_perc != 1:
            # get testing shard paths
            shard_dir_test = Path(
                f"pred_elem_seq/datafiles/xy/shards/test/n_lhs_{self.cfg.n_lhs}"
            )
            input_paths_test, label_paths_test = self.get_shard_path_lists(
                shard_dir_test
            )
            # get testing ShardedIterableDataset
            self.test_set = ShardedIterableDataset(input_paths_test, label_paths_test)

    def get_indvl_hour_shards(
        self, sim_energies, doe_params, scaler, shuffle_doe=True, shard_size=500
    ):

        start_t = time.time()
        # number of multidx columns
        n_idx_cols = len(self.cfg.multidx_names)
        # get design of experiments as shards and scale partially
        n = doe_params.shape[0]
        # instantiate list to hold DoE shards, train indices, and test indices
        doe_list = []
        train_idx_list = []
        test_idx_list = []
        for shard_i, index in enumerate(range(0, n, shard_size)):
            # partition design of experiment based on index andshard_size
            doe_params_part, sim_energies_part = self.partition_xy(
                doe_params, sim_energies, index, shard_size, n
            )
            # get design of experiment shard based on rolled time index
            doe_shrd = self.get_indvl_hour_doe(
                sim_energies_part, doe_params_part, parll_indexing=False
            )
            if shuffle_doe:
                # shuffle doe shard
                rng = np.random.default_rng(0)
                all_i_arr = np.arange(doe_shrd.shape[0])
                rng.shuffle(all_i_arr.reshape(-1, self.shuffle_group))
                doe_shrd = doe_shrd.iloc[all_i_arr]
            # append doe training shard to list
            doe_list.append(doe_shrd)
            # if there is only training data
            if self.cfg.train_perc == 1:
                train_idx = range(doe_shrd.shape[0])
                test_idx = None
            # if there is only testing data
            elif self.cfg.train_perc == 0:
                train_idx = None
                test_idx = range(doe_shrd.shape[0])
            # if there is training adn testing data
            else:
                # get training indices
                train_n = self.shuffle_group * round(
                    self.cfg.train_perc * doe_shrd.shape[0] / self.shuffle_group
                )
                train_idx = range(0, train_n)
                # get testing indices
                test_idx = range(train_n, doe_shrd.shape[0])

            # add training and testing indices to list (for segmenting DoE into training and testing tensors later)
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)

            # if there is training data
            if isinstance(train_idx, range):
                # fit scaler to training data
                scaler.partial_fit(
                    doe_shrd.iloc[train_idx, n_idx_cols + self.n_dataset_objs :]
                )
            print(f"done shard {shard_i + 1} out of {math.ceil(n / shard_size)}")

        print(f"done getting doe shards in {(time.time() - start_t)/60:.1f} mins")
        start_t = time.time()

        # create shard save directories if not already existent
        if self.cfg.train_perc == 1:
            # get directory to save training shard tensors
            train_shard_dir = self.get_train_shard_dir()
            test_shard_dir = None
        elif self.cfg.train_perc == 0:
            # get directory to save training shard tensors
            test_shard_dir = self.get_test_shard_dir()
            train_shard_dir = None
        else:
            # get directory to save training shard tensors
            train_shard_dir = self.get_train_shard_dir()
            # get directory to save training shard tensors
            test_shard_dir = self.get_test_shard_dir()

        # if there are too many shards to fit in memory
        if len(doe_list) > 20:
            # reshape and save shards in serial
            for shard_i, (doe, train_idx, test_idx) in enumerate(
                zip(doe_list, train_idx_list, test_idx_list)
            ):
                self.reshape_save_shard(
                    doe,
                    scaler,
                    train_idx,
                    test_idx,
                    train_shard_dir,
                    test_shard_dir,
                    shard_i,
                )
        # otherwise, reshape and save shards in parallel
        else:
            # todo: remove threading once joblib is updated
            Parallel(n_jobs=8, backend="threading")(
                delayed(self.reshape_save_shard)(
                    doe,
                    scaler,
                    train_idx,
                    test_idx,
                    train_shard_dir,
                    test_shard_dir,
                    shard_i,
                )
                for shard_i, (doe, train_idx, test_idx) in enumerate(
                    zip(doe_list, train_idx_list, test_idx_list)
                )
            )
        print(f"done saving shards in {(time.time() - start_t)/60:.1f} mins")

    def get_test_shard_dir(self) -> Path:
        test_shard_dir = Path(
            f"pred_elem_seq/datafiles/xy/shards/test/n_lhs_{self.cfg.n_lhs}"
        )
        # make directory if nonexistent
        if not os.path.isdir(test_shard_dir):
            test_shard_dir.mkdir()
        return test_shard_dir

    def get_train_shard_dir(self) -> Path:
        train_shard_dir = Path(
            f"pred_elem_seq/datafiles/xy/shards/train/n_lhs_{self.cfg.n_lhs}"
        )
        # make directory if nonexistent
        if not os.path.isdir(train_shard_dir):
            train_shard_dir.mkdir()
        return train_shard_dir

    def get_shard_path_lists(self, shard_dir):
        building = self.cfg.building
        # get middle part of x shard file name based on building and dataset attributes
        shard_x_str = (
            f"{building.name} - n_lhs_{self.cfg.n_lhs} - par_{building.n_unknown_params} - "
            f"time_{building.sim_time_freq}"
        )
        input_paths = sorted(
            [
                pth
                for pth in shard_dir.iterdir()
                if pth.stem.startswith("x") and shard_x_str in pth.stem
            ]
        )

        # get middle part of y shard file name based on building and dataset attributes
        shard_y_str = (
            f"{building.name} - n_lhs_{self.cfg.n_lhs} - par_{building.n_unknown_params} - "
            f"time_{building.sim_time_freq} - {self.sim_precision}"
        )
        # if there is only one objective (i.e., "heat" or "cool")
        if len(building.obj_names) == 1:
            # get starting string for labels shard
            shard_y_start = f"y_{building.obj_names[0]} -"
        # otherwise, if there are multiple objectives
        else:
            # get starting string for labels shard
            shard_y_start = f"y -"
        label_paths = sorted(
            [
                pth
                for pth in shard_dir.iterdir()
                if pth.stem.startswith(shard_y_start) and shard_y_str in pth.stem
            ]
        )

        return input_paths, label_paths

    def reshape_save_shard(
        self, doe, scaler, train_idx, test_idx, train_shard_dir, test_shard_dir, shard_i
    ):
        xy_tensors = self.get_xy_tensors(doe, scaler, train_idx, test_idx)

        if self.cfg.train_perc != 0:
            # todo: include logic for x2_train tensor
            self.save_shard(
                xy_tensors["x1_train"], xy_tensors["y_train"], shard_i, train_shard_dir
            )

        if self.cfg.train_perc != 1:
            # todo: include logic for x2_test tensor
            self.save_shard(
                xy_tensors["x1_test"], xy_tensors["y_test"], shard_i, test_shard_dir
            )

    def save_shard(self, x, y, shard_i, shard_dir):
        building = self.cfg.building

        # outputting individual energy objectives
        for i, energy in enumerate(self.dataset_obj_names):
            # get specific energy tensor
            y_indvl = y[:, [i]]
            # save indvl y tensor shard
            torch.save(
                y_indvl,
                shard_dir / f"y_{energy} - shard_{shard_i:02d} - {building.name} - "
                f"n_lhs_{self.cfg.n_lhs} - par_{building.n_unknown_params} - "
                f"time_{building.sim_time_freq} - "
                f"{self.sim_precision} - n_{x.shape[0]}.pt",
            )
        # save y tensor shard
        torch.save(
            y,
            shard_dir / f"y - shard_{shard_i:02d} - {building.name} - "
            f"n_lhs_{self.cfg.n_lhs} - par_{building.n_unknown_params} - "
            f"time_{building.sim_time_freq} - "
            f"{self.sim_precision} - n_{x.shape[0]}.pt",
        )
        # save x shard
        torch.save(
            x,
            shard_dir / f"x - shard_{shard_i:02d} - {building.name} - "
            f"n_lhs_{self.cfg.n_lhs} - par_{building.n_unknown_params} - "
            f"time_{building.sim_time_freq} - n_{x.shape[0]}.pt",
        )

    def partition_xy(self, doe_params, sim_energies, index, shard_size, n):
        # necessary because last loop is possibly incomplete
        partial_size = min(shard_size, n - index)
        # get size of sim_time_idx
        len_sim_time_idx = len(self.cfg.building.sim_time_idx)
        # get doe_params, y_heating, and y_cooling shard
        doe_params_shrd = doe_params.iloc[index : index + partial_size]
        sim_energies_shrd = sim_energies.iloc[
            index * len_sim_time_idx : (index + partial_size) * len_sim_time_idx
        ]
        return doe_params_shrd, sim_energies_shrd

    def get_datasets(
        self,
        run_lhs=False,
        shuffle_doe=True,
        scaler=None,
        shards=False,
        save_shards=False,
        shard_size=500,
        weather_list=None,
        samp_per_weath=None,
        load_to_gpu=False,
        cross_val_indcs=None,
    ):
        # start timer for run time of DoE generation
        start_t = time.time()

        # copy building attribute of ann_config object for code clarity
        building = self.cfg.building

        if self.cfg.train_perc == 0 and scaler is None:
            # if gathering just testing data, the scaler must be defined
            raise ValueError("if train_perc == 0, scaler must be provided")

        # get current working directory with python project datafiles
        datafiles_dir = Path("pred_elem_seq/datafiles")

        # if the design of experiment is a univariate-focused one
        if self.cfg.n_lhs_uni != 0:
            # if doing a k-folds cross validation:
            if isinstance(cross_val_indcs, dict):
                # get path of base model univariate scaler
                scaler_path = datafiles_dir / (
                    f"scaler/scaler_uni - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - seq_{self.cfg.seq_len} - par_{building.n_unknown_params} - "
                    f"fold_{cross_val_indcs['fold']} - time_{building.sim_time_freq}.sav"
                )
                # get path of univariate sim_energies and doe_params parquet files
                sim_parq_path = datafiles_dir / (
                    f"xy/xy_uni_sim - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - par_{building.n_unknown_params} - fold_{cross_val_indcs['fold']} - "
                    f"time_{building.sim_time_freq} - {self.sim_precision}.parquet"
                )
                params_parq_path = datafiles_dir / (
                    f"xy/xy_uni_params - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - par_{building.n_unknown_params} - fold_{cross_val_indcs['fold']} - "
                    f"time_{building.sim_time_freq}.parquet"
                )
            else:
                # get path of base model univariate scaler
                scaler_path = datafiles_dir / (
                    f"scaler/scaler_uni - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - seq_{self.cfg.seq_len} - par_{building.n_unknown_params} - "
                    f"time_{building.sim_time_freq}.sav"
                )
                # get path of univariate sim_energies and doe_params parquet files
                sim_parq_path = datafiles_dir / (
                    f"xy/xy_uni_sim - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - par_{building.n_unknown_params} - time_{building.sim_time_freq} - "
                    f"{self.sim_precision}.parquet"
                )
                params_parq_path = datafiles_dir / (
                    f"xy/xy_uni_params - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - par_{building.n_unknown_params} - time_{building.sim_time_freq}.parquet"
                )

        else:
            # if doing a k-folds cross validation:
            if isinstance(cross_val_indcs, dict):
                # get path of base model orthogonal scaler
                scaler_path = datafiles_dir / (
                    f"scaler/scaler_orth - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"seq_{self.cfg.seq_len} - par_{building.n_unknown_params} - "
                    f"fold_{cross_val_indcs['fold']} - time_{building.sim_time_freq}.sav"
                )
                # get path of orthogonal sim_energies and doe_params parquet files
                sim_parq_path = datafiles_dir / (
                    f"xy/xy_orth_sim - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"par_{building.n_unknown_params} - fold_{cross_val_indcs['fold']} - "
                    f"time_{building.sim_time_freq} - {self.sim_precision}.parquet"
                )
                params_parq_path = datafiles_dir / (
                    f"xy/xy_orth_params - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"par_{building.n_unknown_params} - fold_{cross_val_indcs['fold']} - "
                    f"time_{building.sim_time_freq}.parquet"
                )
            else:
                # get path of base model orthogonal scaler
                scaler_path = datafiles_dir / (
                    f"scaler/scaler_orth - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"seq_{self.cfg.seq_len} - par_{building.n_unknown_params} - time_{building.sim_time_freq}.sav"
                )
                # get path of orthogonal sim_energies and doe_params parquet files
                sim_parq_path = datafiles_dir / (
                    f"xy/xy_orth_sim - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"par_{building.n_unknown_params} - time_{building.sim_time_freq} - {self.sim_precision}.parquet"
                )
                params_parq_path = datafiles_dir / (
                    f"xy/xy_orth_params - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"par_{building.n_unknown_params} - time_{building.sim_time_freq}.parquet"
                )

        if run_lhs:
            # simulate EnergyPlus BEM to get training/validation/testing output and input data
            if self.cfg.n_lhs_uni != 0:
                # run univariate sample design for convolutional neural network
                sim_energies, doe_params = self.uni_lhs()
            else:
                # run orthogonal-array-based LHS design for convolutional neural network
                sim_energies, doe_params = self.orth_lhs(weather_list, samp_per_weath
                )
            print(
                f"done running EnergyPlus simulations in {(time.time() - start_t)/60:.1f} minutes"
            )

            # write DoE to disk
            sim_energies.to_parquet(
                sim_parq_path,
                compression="gzip",
            )
            doe_params.to_parquet(
                params_parq_path,
                compression="gzip",
            )
        else:
            # read sim_energies pandas dataframe
            sim_parquet_f = pq.ParquetFile(sim_parq_path)
            sim_energies = sim_parquet_f.read(use_pandas_metadata=True).to_pandas()
            # cast to sim_precision
            sim_energies = sim_energies.astype(self.sim_precision)
            # read doe_params pandas dataframe
            params_parquet_f = pq.ParquetFile(params_parq_path)
            doe_params = params_parquet_f.read(use_pandas_metadata=True).to_pandas()
            # create default dictionary for dtypes in doe_params
            def_dict_dtypes = defaultdict(lambda: "float32")
            if "weather_path" in doe_params.columns:
                def_dict_dtypes["weather_path"] = "string"
            # pandas reads as float64, so cast to float32
            doe_params = doe_params.astype(def_dict_dtypes)

        print(
            "done getting EnergyPlus simulation results and design of experiment parameter values"
        )

        # if dividing DoE into shards
        if shards:
            self.get_shard_datasets(
                sim_energies,
                doe_params,
                shuffle_doe,
                scaler,
                scaler_path,
                save_shards,
                shard_size,
            )
        # otherwise, create TensorDatasets
        else:
            self.get_tensor_datasets(
                sim_energies,
                doe_params,
                cross_val_indcs,
                shuffle_doe,
                scaler,
                scaler_path,
                load_to_gpu,
            )

        print(f"done getting datasets in {(time.time() - start_t)/60:.1f} minutes")

        # record time it took to get datasets if run_lhs==True
        if run_lhs:
            if self.cfg.n_lhs_uni != 0:
                run_time_path = datafiles_dir / (
                    f"xy/time_xy_uni - {building.name} - n_lhs_{self.cfg.n_lhs_uni} - "
                    f"n_uni_{self.cfg.n_smp_uni} - par_{building.n_unknown_params} - time_{building.sim_time_freq} - "
                    f"{self.sim_precision}.json"
                )
            else:
                run_time_path = datafiles_dir / (
                    f"xy/time_xy_orth - {building.name} - n_lhs_{self.cfg.n_lhs} - "
                    f"par_{building.n_unknown_params} - time_{building.sim_time_freq} - {self.sim_precision}.json"
                )
            write_dic_to_file(
                {"run_lhs_time (h)": (time.time() - start_t) / 3600},
                run_time_path,
            )


class ShardedIterableDataset(IterableDataset):
    def __init__(
        self, input_shard_paths, label_shard_paths, shuffle_shards=False, seed=0
    ):
        self.input_shard_paths = input_shard_paths
        self.label_shard_paths = label_shard_paths
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        # get total number of samples across all shards
        # number of samples per file
        self.n_samples = 0
        for f_p in self.input_shard_paths:
            self.n_samples += int(f_p.stem.split("_")[-1])

    def set_epoch(self, epoch):
        # Deterministic seed per epoch
        self.seed = self.seed + epoch

    def __iter__(self):
        worker_info = get_worker_info()

        # Split shards across workers
        input_shard_paths = self.input_shard_paths[
            worker_info.id :: worker_info.num_workers
        ]
        label_shard_paths = self.label_shard_paths[
            worker_info.id :: worker_info.num_workers
        ]

        if self.shuffle_shards:
            # Create a worker-local RNG
            rng = random.Random(self.seed + worker_info.id)
            indcs = list(range(len(input_shard_paths)))
            # shuffle input and label lists
            rng.shuffle(indcs)
            input_shard_paths = [input_shard_paths[idx] for idx in indcs]
            label_shard_paths = [label_shard_paths[idx] for idx in indcs]

        for in_shrd_pth, lab_shrd_pth in zip(input_shard_paths, label_shard_paths):
            x_all = torch.load(in_shrd_pth, map_location="cpu")
            y_all = torch.load(lab_shrd_pth, map_location="cpu")

            for i in range(len(x_all)):
                yield x_all[i], y_all[i]


def k_fold_cross_val(ann, n_folds=5, train=False, run_ann=False):
    # instantiate AnnDatasets for all folds
    ann_datasets_fold = AnnDatasets(ann.cfg, ["heat", "cool"])
    # instantiate dataframe to hold cross validation results
    cross_val_res = pd.DataFrame(
        index=range(1, n_folds + 1),
        columns=[
            "$R_{heat}^2$",
            "$$R_{cool}^2$$",
            "$$R_{avg}^2$$",
            "${CVRMSE}_{heat}$",
            "${CVRMSE}_{cool}$",
            "${CVRMSE}_{avg}$",
            "Train time [h]",
        ],
    )
    cross_val_res.index.name = "fold"
    # if predicting individual hours
    if ann.cfg.pred_len == 1:
        # get number of samples for each building simulation (accounts for rolled sequences)
        n_samp_sim = len(ann_datasets_fold.get_sim_time_idx_rolls())
    # if predicting time series
    else:
        # get number of samples for each building simulation (simply the number of hours in the simulation year)
        n_samp_sim = len(ann.cfg.building.sim_time_idx)
    # get total number of training + testing samples
    n_samples = ann.cfg.n_lhs * n_samp_sim
    # seed random number generator (different seed than 0, to not have an identical shuffle with regular training)
    rng = np.random.default_rng(1)
    # get integer indices of DoE
    all_idcs = np.arange(n_samples)
    # shuffle indices, keeping shuffle groups together
    rng.shuffle(all_idcs.reshape(-1, ann_datasets_fold.shuffle_group))
    # segment the shuffled indices into n_folds (these will be testing indices)
    test_idcs_folds = np.split(all_idcs, n_folds)
    # get training indices (the indices in "all indices" that are not in the testing indices)
    train_idcs_folds = [
        all_idcs[~np.isin(all_idcs, test_idx)] for test_idx in test_idcs_folds
    ]

    # for each fold, train and test the surrogate model
    for fold, (train_idx, test_idx) in enumerate(
        zip(train_idcs_folds, test_idcs_folds)
    ):
        print(f"fold: {fold + 1} out of {n_folds}")
        # create cross_val_idcs dictionary that has training and validation indices
        cross_val_idcs = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "fold": fold,
        }
        # get training and testing dataset
        ann_datasets_fold.get_datasets(
            shuffle_doe=False, cross_val_indcs=cross_val_idcs
        )
        if train:
            # train
            run_time = ann.train_test(ann_datasets_fold, trial_name=f"fold{fold}")[
                "run_time"
            ]
        # get ann_model path
        ann_model_path = next(
            pth
            for pth in Path("pred_elem_seq/datafiles/ann").iterdir()
            if pth.stem.endswith(f"fold{fold}")
        )

        # get ann model attributes from JSON file
        ann_attr_path = ann_model_path.parent / f"res_{ann_model_path.stem}.json"
        with open(ann_attr_path, "r") as fr:
            ann_attr = read_dic_from_file(ann_attr_path)
        # get results dictionary from previous training run
        if not train:
            run_time = ann_attr["run_time"]
        # get predictions for testing dataset
        test_res = get_model_preds_prqt(
            ann_model_path, ann_datasets=ann_datasets_fold, run_ann=run_ann
        )
        metrics = get_metrics(test_res)
        # assign results to cross validation dataframe
        cross_val_res.iloc[fold] = [
            metrics["r2"][0],
            metrics["r2"][1],
            np.mean(metrics["r2"]),
            metrics["cvrmse"][0],
            metrics["cvrmse"][1],
            np.mean(metrics["cvrmse"]),
            run_time,
        ]
        print("R^2", metrics["r2"])
        # write cross validation dataframe to csv file
        cross_val_res.to_csv(
            Path(
                "pred_elem_seq/datafiles/results/cross_validation_results - just fold3.csv"
            )
        )


def get_metrics(res, uni_perf=False):
    # sum over all axes except for axis=1
    sum_axes = tuple(ax for ax in range(res["labels"].ndim) if ax != 1)
    # get number of objectives
    n_objs = res["labels"].shape[1]
    if uni_perf:
        # reshape predictions and labels
        labs_3d = res["labels"].reshape((-1, 2, n_objs)).transpose((0, 2, 1))
        preds_3d = res["preds"].reshape((-1, 2, n_objs)).transpose((0, 2, 1))
        # calculate univariate coefficient of determination
        r2_uni = 1 - np.sum(
            (np.diff(labs_3d) - np.diff(preds_3d)) ** 2, axis=(0, 2)
        ) / np.sum(
            (np.diff(labs_3d) - np.mean(np.diff(labs_3d), axis=(0, 2))) ** 2,
            axis=(0, 2),
        )
    else:
        r2_uni = [np.nan, np.nan]
    # get coefficient of determination
    r2 = 1 - np.sum((res["labels"] - res["preds"]) ** 2, axis=0) / np.sum(
        (res["labels"] - np.mean(res["labels"], axis=0)) ** 2, axis=0
    )
    dif_lab_pred = (res["labels"] - res["preds"]) ** 2
    rmse = np.sqrt(np.sum(dif_lab_pred, axis=0) / (len(res["labels"]) - 1))
    # get coefficient of root mean square error
    cvrmse = rmse / np.mean(res["labels"], axis=0) * 100
    # get mean absolute error
    mae = np.mean(np.abs(res["preds"] - res["labels"]), axis=sum_axes)
    # get mean bias error
    mbe = np.mean(res["preds"] - res["labels"], axis=sum_axes)
    # get mean absolute percentage error
    mape = [
        np.mean(
            np.abs(
                (res["preds"][:, i] - res["labels"][:, i])
                / (res["labels"][:, i] + 1e-20)
            ),
            axis=sum_axes,
        )
        * 100
        for i in range(res["labels"].shape[1])
    ]
    # get mean squared error
    mse = np.mean((res["labels"] - res["preds"]) ** 2, axis=0)
    metrics = {
        "r2": r2,
        "r2_uni": r2_uni,
        "cvrmse": cvrmse,
        "mbe": mbe,
        "mae": mae,
        "mape": mape,
        "mse": mse,
    }
    return metrics


def year_sum_res(ann_model_path, ann_datasets, run_ann=False):
    # get predictions from ann_model
    res = get_model_preds_prqt(
        ann_model_path, ann_datasets=ann_datasets, run_ann=run_ann
    )
    # get number of objectives
    n_objs = ann_datasets.cfg.out_dim0
    # reshape numpy arrays and sum predictions over year
    res_year = {
        key: arr.reshape(-1, 8760, n_objs).transpose(0, 2, 1).sum(axis=2)
        for key, arr in res.items()
    }
    # instantiate dataframe of performance metrics
    res_metrics_df = pd.DataFrame(
        index=["Heating", "Cooling", "Average"],
        columns=["$MBE$", "$MAE$", "$R^2$", "$R_{uni}^2$", r"$CVRMSE\ \%$"],
    )
    res_metrics_df.index.name = "Load type"

    if ann_datasets.cfg.n_lhs_uni != 0:
        uni_perf = True
        start = f"test_lhs_uni_{ann_datasets.cfg.n_lhs_uni}"
    else:
        uni_perf = False
        start = f"test_lhs_{ann_datasets.cfg.n_lhs}"

    metrics = get_metrics(res_year, uni_perf)
    res_metrics_df["$MBE$"] = np.append(metrics["mbe"], np.mean(metrics["mbe"]))
    res_metrics_df["$MAE$"] = np.append(metrics["mae"], np.mean(metrics["mae"]))
    res_metrics_df["$R^2$"] = np.append(metrics["r2"], np.mean(metrics["r2"]))
    res_metrics_df["$R_{uni}^2$"] = np.append(
        metrics["r2_uni"], np.mean(metrics["r2_uni"])
    )
    res_metrics_df[r"$CVRMSE\ \%$"] = np.append(
        metrics["cvrmse"], np.mean(metrics["cvrmse"])
    )
    res_metrics_df.to_csv(
        Path(
            f"pred_elem_seq/datafiles/results/{start} - {ann_model_path.stem} - annual sum metrics.csv"
        )
    )


def get_ann_test_res(ann_datasets_list=None, run_ann=False):
    # get ann model paths
    ann_model_paths = [
        pth
        for pth in Path("pred_elem_seq/datafiles/ann").iterdir()
        if pth.suffix == ".sav"
    ]

    # instantiate table with test results
    res_table = pd.DataFrame(
        index=range(len(ann_model_paths)),
        columns=[
            "LHS samples",
            "Training epochs",
            "Prediction length",
            r"seq\_len",
            "Batch size",
            "Starting learning rate",
            "Max learning rate",
            "Weight decay",
            "Moment coefficients",
            "Activation function",
            "Conv blocks",
            "Conv layers per block",
            "Kernel sizes per block",
            "Channels per conv layer",
            "FC blocks",
            "FC layers per block",
            "Nodes per FC layer",
            "Number of parameters",
            "${MAE}_{heat}$",
            "${MAE}_{cool}$",
            "${MAE}_{avg}$",
            "${MBE}_{heat}$",
            "${MBE}_{cool}$",
            "${MBE}_{avg}$",
            r"${MAPE}_{heat}\ \%$",
            r"${MAPE}_{cool}\ \%$",
            r"${MAPE}_{avg}\ \%$",
            "$R_{heat}^2$",
            "$R_{cool}^2$",
            "$R_{avg}^2$",
            "$R_{uni,heat}^2$",
            "$R_{uni,cool}^2$",
            "$R_{uni,avg}^2$",
            r"${CVRMSE}_{heat}\ \%$",
            r"${CVRMSE}_{cool}\ \%$",
            r"${CVRMSE}_{avg}\ \%$",
            "Train time [h]",
        ],
    )

    for i, ann_model_path in enumerate(ann_model_paths):
        print(ann_model_path)
        # get results attributes from pretrained ann model
        ann_attr = read_dic_from_file(
            Path(f"pred_elem_seq/datafiles/ann/res_{ann_model_path.stem}.json")
        )
        # get first dataset in the list (in order to check whether the list has univariate or orthogonal datasets)
        first_ds = list(ann_datasets_list.values())[0]
        # if the first dataset in the list is a univariate dataset
        if first_ds.cfg.n_lhs_uni != 0:
            # all univariate datasets in the list should have the same n_lhs
            if (
                len(
                    set(
                        [
                            dataset.cfg.n_lhs_uni
                            for dataset in ann_datasets_list.values()
                        ]
                    )
                )
                > 1
            ):
                raise ValueError(
                    "All univariate datasets must have the same n_lhs_uni value"
                )
            n_lhs_test = first_ds.cfg.n_lhs_uni
            seq_len_test = ann_attr["seq_len"]
            nn_type = ann_attr["nn_type"]
            uni_perf = True
        else:
            # all orthogonal datasets in the list should have the same n_lhs
            if (
                len(set([dataset.cfg.n_lhs for dataset in ann_datasets_list.values()]))
                > 1
            ):
                raise ValueError(
                    "All orthogonal datasets must have the same n_lhs value"
                )
            n_lhs_test = first_ds.cfg.n_lhs
            seq_len_test = ann_attr["seq_len"]
            nn_type = ann_attr["nn_type"]
            uni_perf = False
        # get AnnDatasets object based on n_lhs_tes, seq_len, and nn_type value
        ann_datasets = ann_datasets_list[(n_lhs_test, seq_len_test, nn_type)]
        # get predictions with ann_model
        res = get_model_preds_prqt(ann_model_path, ann_datasets, run_ann)
        # get performance metrics
        metrics = get_metrics(res, uni_perf)
        # capitalize letters of prelu and relu
        if ann_attr["activation"] == "prelu":
            act_func = "PReLU"
        elif ann_attr["activation"] == "relu":
            act_func = "ReLU"
        else:
            act_func = ann_attr["activation"]

        # get moment coefficients (opt_betas)
        if "opt_betas" in ann_attr.keys():
            opt_betas = ann_attr["opt_betas"]
        else:
            opt_betas = [0.9, 0.999]

        # get number of model parameters
        n_model_params = get_n_model_params(ann_model_path)
        # write ann_outputs to table
        res_table.iloc[i] = [
            ann_attr["n_lhs"],
            ann_attr["epochs"],
            ann_attr["pred_len"],
            ann_attr["seq_len"],
            ann_attr["batch_size"],
            ann_attr["max_lr"] / ann_attr["div_f_lr"],
            ann_attr["max_lr"],
            ann_attr["weight_decay"],
            opt_betas,
            act_func,
            ann_attr["n_conv_blocks"],
            ann_attr["n_conv_layers"],
            ann_attr["kernels"],
            ann_attr["n_channels"],
            ann_attr["n_lin_blocks"],
            ann_attr["n_lin_layers"],
            ann_attr["n_nodes"],
            n_model_params,
            metrics["mae"][0],
            metrics["mae"][1],
            np.mean(metrics["mae"]),
            metrics["mbe"][0],
            metrics["mbe"][1],
            np.mean(metrics["mbe"]),
            metrics["mape"][0],
            metrics["mape"][1],
            np.mean(metrics["mape"]),
            metrics["r2"][0],
            metrics["r2"][1],
            np.mean(metrics["r2"]),
            metrics["r2_uni"][0],
            metrics["r2_uni"][1],
            np.mean(metrics["r2_uni"]),
            metrics["cvrmse"][0],
            metrics["cvrmse"][1],
            np.mean(metrics["cvrmse"]),
            ann_attr["run_time"],
        ]
        print(f"done {i + 1} of {len(res_table)}")

        if uni_perf:
            res_table.to_csv(
                Path("pred_elem_seq/datafiles/results/ann_test_results_uni.csv")
            )
        else:
            res_table.to_csv(
                Path(
                    "pred_elem_seq/datafiles/results/ann_test_results_with_n_params.csv"
                )
            )


def get_model_preds_prqt(ann_model_path, ann_datasets, run_ann=False):

    # define parquet path to store predictions
    if ann_datasets.cfg.n_lhs_uni != 0:
        parquet_path = Path(
            f"pred_elem_seq/datafiles/ann/ann_outputs/test_lhs_uni_{ann_datasets.cfg.n_lhs_uni} - {ann_model_path.stem}.parquet"
        )
    else:
        parquet_path = Path(
            f"pred_elem_seq/datafiles/ann/ann_outputs/test_lhs_{ann_datasets.cfg.n_lhs} - {ann_model_path.stem}.parquet"
        )

    # get ann model attributes from JSON file
    ann_attr_path = ann_model_path.parent / f"res_{ann_model_path.stem}.json"
    with open(ann_attr_path, "r") as fr:
        ann_attr = read_dic_from_file(ann_attr_path)
    # get ann objectives (prediction target(s))
    obj_names_ann = ann_attr.get("obj_names", ["heat", "cool"])

    # get ann_datasets objectives
    obj_names_ds = ann_datasets.dataset_obj_names

    if run_ann:
        # load torch model from disk
        with open(ann_model_path, "rb") as f:
            ann_model = pickle.load(f)

        if ann_datasets is None:
            raise ValueError("if run_ann==True, ann_datasets must be defined")
        # get results for testing ann_datasets
        res = get_model_preds(ann_model, ann_datasets, obj_names_ann)

        if res["labels"].ndim == 3:
            res = {
                key: val.transpose(0, 2, 1).reshape(-1, len(obj_names_ann))
                for key, val in res.items()
            }
        # instantiate dataframe to save to disk
        df = pd.DataFrame(
            data=np.concat((res["preds"], res["labels"]), axis=1),
            columns=[f"preds {obj}" for obj in obj_names_ds]
            + [f"labels {obj}" for obj in obj_names_ds],
        )
        df.to_parquet(parquet_path)
    else:
        # read results dataframe
        res_parquet_f = pq.ParquetFile(parquet_path)
        res_df = res_parquet_f.read(use_pandas_metadata=True).to_pandas()
        res = {
            "preds": res_df[[f"preds {obj}" for obj in obj_names_ds]].to_numpy(),
            "labels": res_df[[f"labels {obj}" for obj in obj_names_ds]].to_numpy(),
        }

    # # remove nan values (because of IterableDataset sometimes has incomplete shards)
    # res["preds"] = res["preds"][~np.isnan(res["preds"]).any(axis=1)]
    # res["labels"] = res["labels"][~np.isnan(res["labels"]).any(axis=1)]

    return res


def get_model_preds(ann_model, ann_datasets, obj_names_ann ):

    if ann_datasets.cfg.seq_len == 0:
        # don't pin memory if predicting yearly energy (data is already loaded on GPU)
        pin_memory = False
    else:
        pin_memory = True

    if isinstance(ann_datasets.test_set, ShardedIterableDataset):
        n_samples = ann_datasets.test_set.n_samples
        persistent_workers = True
        drop_last = True
    else:
        n_samples = len(ann_datasets.test_set.tensors[0])
        persistent_workers = False
        drop_last = False

    batch_size = len(ann_datasets.cfg.building.sim_time_idx)

    # make DataLoader based on testing dataset
    dataloader = DataLoader(
        ann_datasets.test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=0,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    # get objectives for ann_dataset
    obj_names_ds = ann_datasets.dataset_obj_names
    # get number of ann_datasets objectives
    n_objs_ds = ann_datasets.n_dataset_objs
    # get objective indices for ann
    obj_indcs = [obj_names_ds.index(obj) for obj in obj_names_ann]

    # start the time counter
    start_t = time.time()

    # if predicting time series data
    if ann_datasets.cfg.pred_len > 1:
        # instantiate nan torch tensors to store predictions and labels
        results = {
            "preds": torch.full(
                (n_samples, n_objs_ds, ann_datasets.cfg.pred_len),
                torch.nan,
                device=device,
            ),
            "labels": torch.full(
                (n_samples, n_objs_ds, ann_datasets.cfg.pred_len), torch.nan
            ),
        }
    # if predicting individual hours
    else:
        # instantiate nan torch tensors to store predictions and labels
        results = {
            "preds": torch.full((n_samples, n_objs_ds), torch.nan, device=device),
            "labels": torch.full((n_samples, n_objs_ds), torch.nan),
        }

    # set model to evaluation mode
    ann_model.eval()
    # Start testing
    start_idx = 0
    for i, (inputs, labels) in enumerate(dataloader):
        if ann_datasets.cfg.seq_len > 0:
            # send batches to GPU
            inputs = inputs.to(device)        
        # run forward pass with autocasting
        with autocast(device_type="cuda", dtype=torch.float32), torch.no_grad():
            # forward
            outs = ann_model(inputs)
        # get the end_idx for results assignment
        end_idx = start_idx + len(outs)
        # we populate the predictions tensor with the outputs from the
        # PyTorch model in the batch.
        results["preds"][start_idx : end_idx, obj_indcs] = outs
        results["labels"][start_idx : end_idx] = labels
        # set start_idx to end_idx
        start_idx = end_idx
        print(f"Finished Batch {i + 1} out of {len(dataloader)}")

    # convert tensors to numpy arrays
    # torch.cuda.synchronize()
    results["preds"] = results["preds"].detach().cpu().numpy()
    results["labels"] = results["labels"].detach().cpu().numpy()

    print(f"completion time = {(time.time() - start_t) / 60:.3f}min")
    return results


def configure_optimizers(ann_model, min_lr, weight_decay, opt_betas):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        nn.Linear,
        nn.Conv1d,
    )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.BatchNorm1d, nn.PReLU)
    for mn, m in ann_model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in ann_model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=min_lr, betas=opt_betas)
    return optimizer


if __name__ == "__main__":
    pass
