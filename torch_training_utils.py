# PYTORCH MODEL TRAINING FUNCTIONS
"""Contains functions for training, validating, and testing a pytorch model.

"""

####################################
## Packages
####################################
import os
import sys
import glob
import random
import time
import math
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler


######################################################
## Helper function for model/optimizer saving/loading
######################################################
def save_model_and_optimizer_hdf5(model, optimizer, epoch, filepath):
    """Saves the state of a model and optimizer in portable hdf5 format. Model and
    optimizer should be moved to the CPU prior to using this function.

    Args:
        model (torch model): Pytorch model to save
        optimizer (torch optimizer: Pytorch optimizer to save
        epoch (int): Epoch associated with training
        filepath (str): Where to save

    """
    with h5py.File(filepath, 'w') as h5f:
        # Save epoch number
        h5f.attrs['epoch'] = epoch
        
        # Save model parameters and buffers
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy()
            if data.ndim == 0:  # It's a scalar!
                h5f.attrs['model/parameters/' + name] = data
            else:
                h5f.create_dataset('model/parameters/' + name,
                                   data=data)

        for name, buffer in model.named_buffers():
            data = buffer.cpu().numpy()
            if data.ndim == 0:  # It's a scalar!
                h5f.attrs['model/buffers/' + name] = data
            else:
                h5f.create_dataset('model/buffers/' + name,
                                   data=data)
            
        # Save optimizer state
        optimizer_state = optimizer.state_dict()
        for idx, group in enumerate(optimizer_state['param_groups']):
            group_name = f'optimizer/group{idx}'
            for k, v in group.items():
                #print('group_name:', group_name, k)
                if isinstance(v, (int, float)):
                    h5f.attrs[group_name + '/' + k] = v
                elif isinstance(v, list):
                    h5f.create_dataset(group_name + '/' + k, data=v)

        # Save state values, like momentums
        for idx, state in enumerate(optimizer_state['state'].items()):
            state_name = f'optimizer/state{idx}'
            for k, v in state[1].items():
                #print('state_name:', state_name, k)
                if isinstance(v, torch.Tensor):
                    h5f.create_dataset(state_name + '/' + k,
                                       data=v.detach().cpu().numpy())


def load_model_and_optimizer_hdf5(model, optimizer, filepath):
    """Loads state of model and optimizer stored in an hdf5 format.

    Args:
        model (torch model): Pytorch model to save
        optimizer (torch optimizer: Pytorch optimizer to save
        filepath (str): Where to save

    Returns:
        epoch (int): Epoch associated with training

    """
    with h5py.File(filepath, 'r') as h5f:
        # Get epoch number
        epoch = h5f.attrs['epoch']
        
        # Load model parameters and buffers
        for name in h5f.get('model/parameters', []):  # Get the group
            if isinstance(h5f['model/parameters/' + name], h5py.Dataset):
                data = torch.from_numpy(h5f['model/parameters/' + name][:])
            else:
                data = torch.tensor(h5f.attrs['model/parameters/' + name])

            name_list = name.split('.')
            param_name = name_list.pop()
            submod_name = '.'.join(name_list)

            model.get_submodule(submod_name)._parameters[param_name].data.copy_(data)

        for name in h5f.get('model/buffers', []):
            if isinstance(h5f['model/buffers/' + name], h5py.Dataset):
                buffer = torch.from_numpy(h5f['model/buffers/' + name][:])
            else:
                buffer = torch.tensor(h5f.attrs['model/buffers/' + name])

            name_list = name.split('.')
            param_name = name_list.pop()
            submod_name = '.'.join(name_list)
            model.get_submodule(submod_name)._buffers[param_name].data.copy_(buffer)

        # Rebuild optimizer state (need to call this before loading state)
        optimizer_state = optimizer.state_dict()

        # Load optimizer parameter groups
        for k in h5f.attrs:
            if 'optimizer/group' in k:
                #print('k-string:', k)
                idx, param = k.split('/')[1:]
                optimizer_state['param_groups'][int(idx.lstrip('group'))][param] = h5f.attrs[k]

        # Load state values, like momentums
        for name, group in h5f.items():
            if 'optimizer/state' in name:
                state_idx = int(name.split('state')[1])
                param_idx, param_state = list(optimizer_state['state'].items())[state_idx]
                for k in group:
                    optimizer_state['state'][param_idx][k] = torch.from_numpy(group[k][:])

        # Load optimizer state
        optimizer.load_state_dict(optimizer_state)

    return epoch


####################################
## Make Dataloader form DataSet
####################################
def make_dataloader(dataset: torch.utils.data.Dataset,
                    batch_size: int=8,
                    num_batches: int=100,
                    num_workers: int=4):
    """Function to create a pytorch dataloader from a pytorch dataset
    **https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader** 
    Each dataloader has batch_size*num_batches samples randomly selected 
    from the dataset

    num_workers: behavior training on dodona
        =0 if not specified, data is loaded in the main process;
           trains slower if multiple models being trained on the same node
        =1 seperates the data from the main process;
           training speed unaffected by multiple models being trained
        =2 splits data across 2 processors;
           cuts training time in half from num_workers=1
        >2 diminishing returns on training time

    persistant_workers:
        training time seems minimally affected, slight improvement when =True

    Args:
        dataset(torch.utils.data.Dataset): dataset to sample from for data loader
        batch_size (int): batch size
        num_batches (int): number of batches to include in data loader

    Returns:
        dataloader (torch.utils.data.DataLoader): pytorch dataloader 

    """

    randomsampler = RandomSampler(dataset, num_samples=batch_size*num_batches)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=randomsampler,
                            num_workers=num_workers,
                            persistent_workers=True)

    return dataloader

    
####################################
## Saving Results
####################################
def save_append_df(path: str, df: pd.DataFrame, START: bool):
    """Function to save/append dataframe contents to a csv file

        Args:
            path (str): path of csv file
            df (pd.DataFrame): pandas dataframe to save
            START (bool): indicates if the file path needs to be initiated

        Returns:
            No Return Objects

    """
    
    if START:
        assert not os.path.isfile(path), 'If starting training, '+path+' should not exist.'
        df.to_csv(path, header=True, index=True, mode='x')
    else:
        assert os.path.isfile(path), 'If continuing training, '+path+' should exist.'
        df.to_csv(path, header=False, index=True, mode='a')


def append_to_dict(dictt: dict, batch_ID: int, truth, pred, loss):
    """Function to appending sample information to a dictionary Dictionary must 
    be initialized with correct keys

    Args:
        dictt (dict): dictionary to append sample information to
        batch_ID (int): batch ID number for samples
        truth (): array of truth values for batch of samples
        pred (): array of prediction values for batch of samples
        loss (): array of loss values for batch of samples

    Returns:
        dictt (dict): dictionary with appended sample information

    """
    
    batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]
    
    for i in range(batchsize):
        dictt["epoch"].append(0)  # To be easily identified later
        dictt["batch"].append(batch_ID)
        dictt["truth"].append(truth.cpu().detach().numpy().flatten()[i])
        dictt["prediction"].append(pred.cpu().detach().numpy().flatten()[i])
        dictt["loss"].append(loss.cpu().detach().numpy().flatten()[i])

    return dictt


####################################
## Continue Slurm Study
####################################
def continuation_setup(checkpointpath, studyIDX, last_epoch):
    """Function to generate the training.input and training.slurm files for
    continuation of model training

     Args:
         checkpointpath (str): path to model checkpoint to load in model from
         studyIDX (int): study ID to include in file name
         last_epoch (int): numer of epochs completed at this checkpoint

     Returns:
         new_training_slurm_filepath (str): Name of slurm file to submit job for 
                                            continued training

    """
    
    ## Identify Template Files
    training_input_tmpl = "./training_input.tmpl"
    training_slurm_tmpl = "./training_slurm.tmpl"

    ## Make new training.input file
    with open(training_input_tmpl, 'r') as f:
        training_input_data = f.read()
        
    new_training_input_data = training_input_data.replace('<CHECKPOINT>',
                                                          checkpointpath)
    
    input_str = 'study{0:03d}_restart_training_epoch{1:04d}.input'
    new_training_input_filepath = input_str.format(studyIDX, last_epoch+1)
    
    with open(os.path.join('./', new_training_input_filepath), 'w') as f:
        f.write(new_training_input_data)

    with open(training_slurm_tmpl, 'r') as f:
        training_slurm_data = f.read()

    slurm_str = 'study{0:03d}_restart_training_epoch{1:04d}.slurm'
    new_training_slurm_filepath = slurm_str.format(studyIDX, last_epoch+1)
    
    new_training_slurm_data = training_slurm_data.replace('<INPUTFILE>',
                                                          new_training_input_filepath)
    
    new_training_slurm_data = new_training_slurm_data.replace('<epochIDX>',
                                                              '{0:04d}'.format(last_epoch+1))
    
    with open(os.path.join('./', new_training_slurm_filepath), 'w') as f:
        f.write(new_training_slurm_data)

    return new_training_slurm_filepath


####################################
## Training on a Datastep
####################################
def train_scalar_datastep(data: tuple, 
                          model,
                          optimizer,
                          loss_fn,
                          device: torch.device):
    """Function to complete a training step on a single sample in which the
    network's output is a scalar.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    
    ## Set model to train
    model.train()

    ## Extract data
    (inpt, truth) = data
    inpt = inpt.to(device)
    # Unsqueeze is necessary for scalar ground-truth output
    truth = truth.to(torch.float32).unsqueeze(-1).to(device)
    
    ## Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    ## Perform backpropagation and update the weights
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    return truth, pred, loss


def train_array_datastep(data: tuple, 
                         model,
                         optimizer,
                         loss_fn,
                         device: torch.device):
    """Function to complete a training step on a single sample in which the
    network's output is an array.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    
    ## Set model to train
    model.train()

    ## Extract data
    (inpt, truth) = data
    inpt = inpt.to(device)
    truth = truth.to(device)
    
    ## Perform a forward pass
    # NOTE: If training on GPU model should have already been moved to GPU
    # prior to initalizing optimizer.
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    ## Perform backpropagation and update the weights
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    return truth, pred, loss


####################################
## Evaluating on a Datastep
####################################
def eval_scalar_datastep(data: tuple, 
                         model,
                         loss_fn,
                         device: torch.device):
    """Function to complete a validation step on a single sample for which the
    network output is a scalar.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model evaluate
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    
    ## Set model to eval
    model.eval()

    ## Extract data
    (inpt, truth) = data
    inpt = inpt.to(device)
    truth = truth.to(torch.float32).unsqueeze(-1).to(device)

    ## Perform a forward pass
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    return truth, pred, loss


def eval_array_datastep(data: tuple, 
                        model,
                        loss_fn,
                        device: torch.device):
    """Function to complete a validation step on a single sample in which network
    output is an array.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model evaluate
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select

    Returns:
        loss (): evaluated loss for the data sample

    """
    
    ## Set model to eval
    model.eval()

    ## Extract data
    (inpt, truth) = data
    inpt = inpt.to(device)
    truth = truth.to(device)

    ## Perform a forward pass
    pred = model(inpt)
    loss = loss_fn(pred, truth)

    return truth, pred, loss


######################################
## Training & Validation for an Epoch
######################################
def train_scalar_dict_epoch(training_data,
                            validation_data, 
                            model,
                            optimizer,
                            loss_fn,
                            summary_dict: dict,
                            train_sample_dict: dict,
                            val_sample_dict: dict,
                            device: torch.device):
    """Function to complete a training step on a single sample for a network in
    which the output is a single scalar. Training, Validation, and Summary
    information are saved to dictionaries.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        summary_dict (dict): dictionary to save epoch stats to
        train_sample_dict (dict): dictionary to save training sample stats to
        val_sample_dict (dict): dictionary to save validation sample stats to
        device (torch.device): device index to select

    Returns:
        summary_dict (dict): dictionary with epoch stats
        train_sample_dict (dict): dictionary with training sample stats
        val_sample_dict (dict): dictionary with validation sample stats

    """
    
    ## Initialize things to save
    startTime = time.time()
    trainbatches = len(training_data)
    valbatches = len(validation_data)
    trainbatch_ID = 0
    valbatch_ID = 0

    ## Train on all training samples
    for traindata in training_data:
        trainbatch_ID += 1
        truth, pred, train_loss = train_scalar_datastep(traindata, 
                                                        model,
                                                        optimizer,
                                                        loss_fn,
                                                        device)

        train_sample_dict = append_to_dict(train_sample_dict,
                                           trainbatch_ID,
                                           truth,
                                           pred,
                                           train_loss)
        
    train_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

    ## Calcuate the Epoch Average Loss
    train_samples = train_batchsize * trainbatches
    avgTrainLoss = np.sum(train_sample_dict["loss"][-train_samples:]) / train_samples
    summary_dict["train_loss"].append(avgTrainLoss)
    
    ## Evaluate on all validation samples
    with torch.no_grad():
        for valdata in validation_data:
            valbatch_ID += 1
            truth, pred, val_loss = eval_scalar_datastep(valdata, 
                                                         model,
                                                         loss_fn,
                                                         device)

            val_sample_dict = append_to_dict(val_sample_dict,
                                             valbatch_ID,
                                             truth,
                                             pred,
                                             val_loss)

    val_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

    ## Calcuate the Epoch Average Loss
    val_samples = val_batchsize * valbatches
    avgValLoss = np.sum(val_sample_dict["loss"][-val_samples:]) / val_samples

    summary_dict["val_loss"].append(avgValLoss)

    ## Calculate Time
    endTime = time.time()
    epoch_time = (endTime - startTime) / 60
    summary_dict["epoch_time"].append(epoch_time)

    return summary_dict, train_sample_dict, val_sample_dict


def train_scalar_csv_epoch(training_data,
                           validation_data, 
                           model,
                           optimizer,
                           loss_fn,
                           epochIDX,
                           train_per_val,
                           train_rcrd_filename: str,
                           val_rcrd_filename: str,
                           device: torch.device):
    """Function to complete a training epoch on a network which has a single scalar
    as output. Training and validation information is saved to successive CSV files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation 
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select

    """
    
    ## Initialize things to save
    trainbatches = len(training_data)
    valbatches = len(validation_data)
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size


    train_rcrd_filename = train_rcrd_filename.replace(f'<epochIDX>',
                                                      '{:04d}'.format(epochIDX))
    ## Train on all training samples
    with open(train_rcrd_filename, 'a') as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1
            truth, pred, train_loss = train_scalar_datastep(traindata, 
                                                            model,
                                                            optimizer,
                                                            loss_fn,
                                                            device)

            template = "{}, {}, {}"
            for i in range(train_batchsize):
                print(template.format(epochIDX,
                                      trainbatch_ID,
                                      train_loss.cpu().detach().numpy().flatten()[i]),
                      file=train_rcrd_file)
            
    ## Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print('Validating...', epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace(f'<epochIDX>',
                                                      '{:04d}'.format(epochIDX))
        with open(val_rcrd_filename, 'a') as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_scalar_datastep(valdata, 
                                                                 model,
                                                                 loss_fn,
                                                                 device)

                    template = "{}, {}, {}"
                    for i in range(val_batchsize):
                        print(template.format(epochIDX,
                                              valbatch_ID,
                                              val_loss.cpu().detach().numpy().flatten()[i]),
                              file=val_rcrd_file)

    return


def train_array_csv_epoch(training_data,
                          validation_data, 
                          model,
                          optimizer,
                          loss_fn,
                          epochIDX,
                          train_per_val,
                          train_rcrd_filename: str,
                          val_rcrd_filename: str,
                          device: torch.device):
    """Function to complete a training epoch on a network which has an array
    as output. Training and validation information is saved to successive CSV 
    files.

    Args:
        training_data (torch.dataloader): dataloader containing the training samples
        validation_data (torch.dataloader): dataloader containing the validation samples
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation 
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select

    """
    
    ## Initialize things to save
    trainbatches = len(training_data)
    valbatches = len(validation_data)
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace(f'<epochIDX>',
                                                      '{:04d}'.format(epochIDX))
    ## Train on all training samples
    with open(train_rcrd_filename, 'a') as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1
            truth, pred, train_loss = train_array_datastep(traindata, 
                                                           model,
                                                           optimizer,
                                                           loss_fn,
                                                           device)

            template = "{}, {}, {}"
            for i in range(train_batchsize):
                print(template.format(epochIDX,
                                      trainbatch_ID,
                                      train_loss.cpu().detach().numpy().flatten()[i]),
                      file=train_rcrd_file)
            
    ## Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print('Validating...', epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace(f'<epochIDX>',
                                                      '{:04d}'.format(epochIDX))
        with open(val_rcrd_filename, 'a') as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    truth, pred, val_loss = eval_array_datastep(valdata, 
                                                                model,
                                                                loss_fn,
                                                                device)

                    template = "{}, {}, {}"
                    for i in range(val_batchsize):
                        print(template.format(epochIDX,
                                              valbatch_ID,
                                              val_loss.cpu().detach().numpy().flatten()[i]),
                              file=val_rcrd_file)

    return


if __name__ == '__main__':
    """For testing and debugging.

    """

    sys.path.insert(0, os.getenv('YOKE_DIR'))
    from models.CNN_modules import PVI_SingleField_CNN

    # Excercise model setup, save, and load
    # NOTE: Model takes (BatchSize, Channels, Height, Width) tensor.
    pvi_input = torch.rand(1, 1, 1700, 500)
    pvi_CNN = PVI_SingleField_CNN(img_size=(1, 1700, 500),
                                  size_threshold=(8, 8),
                                  kernel=5,
                                  features=12, 
                                  interp_depth=15,
                                  conv_onlyweights=True,
                                  batchnorm_onlybias=True,
                                  act_layer=nn.GELU,
                                  hidden_features=20)

    optimizer = torch.optim.AdamW(pvi_CNN.parameters(),
                                  lr=0.001,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.01)

    new_h5_path = os.path.join('./', 'TEST_MODEL_SAVE.hdf5')
    epochIDX = 5
    save_model_and_optimizer_hdf5(pvi_CNN, optimizer, epochIDX, new_h5_path)
    print('MODEL SAVED...')
    starting_epoch = load_model_and_optimizer_hdf5(pvi_CNN, optimizer, new_h5_path)
