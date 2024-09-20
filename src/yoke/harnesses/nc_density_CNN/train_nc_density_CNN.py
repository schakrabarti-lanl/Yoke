"""Actual training workhorse for CNN network mapping image to PTW-scale.

"""
#############################################
## Packages
#############################################
import sys
import os
import typing
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from yoke.models.CNNmodules import PVI_SingleField_CNN
from yoke.datasets.nestedcyl_dataset import PVI_SingleField_DataSet
import yoke.torch_training_utils as tr

#############################################
## Inputs
#############################################
descr_str = 'Trains CNN to estimate PTW strength scale from density for the nested cylinder'
parser = argparse.ArgumentParser(prog='NC CNN Training',
                                 description=descr_str,
                                 fromfile_prefix_chars='@')

#############################################
## Learning Problem
#############################################
parser.add_argument('--studyIDX',
                    action='store',
                    type=int,
                    default=1,
                    help='Study ID number to match hyperparameters')

parser.add_argument('--input_field',
                    action='store',
                    type=str,
                    default='hr_MOICyl',
                    help='Data field the models will train on')

#############################################
## File Paths
#############################################
parser.add_argument('--design_file',
                    action='store',
                    type=str,
                    default='/data2/design_nc231213_Sn_MASTER.csv',
                    help='.csv file that contains the truth values for data files')

parser.add_argument('--train_filelist',
                    action='store',
                    type=str,
                    default='nc231213_train_80pct.txt',
                    help='Path to list of files to train on.')

parser.add_argument('--validation_filelist',
                    action='store',
                    type=str,
                    default='nc231213_val_10pct.txt',
                    help='Path to list of files to validate on.')

parser.add_argument('--test_filelist',
                    action='store',
                    type=str,
                    default='nc231213_test_10pct.txt',
                    help='Path to list of files to test on.')

#############################################
## Model Parameters
#############################################
parser.add_argument('--size_threshold_W',
                    action='store',
                    type=int,
                    default=8,
                    help='Upper limit for width of reduced image')

parser.add_argument('--size_threshold_H',
                    action='store',
                    type=int,
                    default=8,
                    help='Upper limit for height of reduced image')

parser.add_argument('--kernel',
                    action='store',
                    type=int,
                    default=5,
                    help='Size of square convolutional kernel')

parser.add_argument('--features',
                    action='store',
                    type=int,
                    default=12,
                    help='Number of features (channels) in a convolution')

parser.add_argument('--interp_depth',
                    action='store',
                    type=int,
                    default=12,
                    help='Number of interpertability blocks in the model')

parser.add_argument('--conv_onlyweights',
                    action='store',
                    type=bool,
                    default=False,
                    help=('Determines if convolutional layers learn only weights '
                          'or weights and bias'))

parser.add_argument('--batchnorm_onlybias',
                    action='store',
                    type=bool,
                    default=False,
                    help=('Determines if the batch normalization layers learn only '
                          'bias or weights and bias'))

parser.add_argument('--act_layer',
                    action='store',
                    type=str,
                    default='nn.GELU',
                    help='Torch layer to use as activation; of the form nn.LAYERNAME')

parser.add_argument('--hidden_features',
                    action='store',
                    type=int,
                    default=20,
                    help='Number of features (channels) the dense layers')

#############################################
## Training Parameters
#############################################
parser.add_argument('--init_learnrate',
                    action='store',
                    type=float,
                    default=1e-3,
                    help='Initial learning rate')

parser.add_argument('--batch_size',
                    action='store',
                    type=int,
                    default=64,
                    help='Batch size')

#############################################
## Epoch Parameters
#############################################
parser.add_argument('--total_epochs',
                    action='store',
                    type=int,
                    default=10,
                    help='Total training epochs')

parser.add_argument('--cycle_epochs',
                    action='store',
                    type=int,
                    default=5,
                    help=('Number of epochs between saving the model and re-queueing '
                          'training process; must be able to be completed in the '
                          'set wall time'))

parser.add_argument('--train_batches',
                    action='store',
                    type=int,
                    default=250,
                    help='Number of batches to train on in a given epoch')

parser.add_argument('--val_batches',
                    action='store',
                    type=int,
                    default=25,
                    help='Number of batches to validate on in a given epoch')

parser.add_argument('--TRAIN_PER_VAL',
                    action='store',
                    type=int,
                    default=10,
                    help='Number of training epochs between each validation epoch')

parser.add_argument('--trn_rcrd_filename',
                    action='store',
                    type=str,
                    default='./default_training.csv',
                    help='Filename for text file of training loss and metrics on each batch')

parser.add_argument('--val_rcrd_filename',
                    action='store',
                    type=str,
                    default='./default_validation.csv',
                    help='Filename for text file of validation loss and metrics on each batch')

parser.add_argument('--continuation',
                    action='store_true',
                    help='Indicates if training is being continued or restarted')

parser.add_argument('--checkpoint',
                    action='store',
                    type=str,
                    default='None',
                    help='Path to checkpoint to continue training from')

#############################################
#############################################
if __name__ == '__main__':

    #############################################
    ## Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ## Study ID
    studyIDX = args.studyIDX

    # YOKE env variables
    YOKE_DIR = os.getenv('YOKE_DIR')
    NC_NPZ_DIR = os.getenv('NC_NPZ_DIR')
    NC_DESIGN_DIR = os.getenv('NC_DESIGN_DIR')

    ## Data Paths
    input_field = args.input_field
    design_file = os.path.abspath(NC_DESIGN_DIR+args.design_file)
    train_filelist = YOKE_DIR + 'filelists/' + args.train_filelist
    validation_filelist = YOKE_DIR + 'filelists/' + args.validation_filelist
    test_filelist = YOKE_DIR + 'filelists/' + args.test_filelist

    ## Model Parameters
    thresholdW = args.size_threshold_W
    thresholdH = args.size_threshold_H
    size_threshold = (thresholdH, thresholdW)
    kernel = args.kernel
    features= args.features
    interp_depth = args.interp_depth
    conv_onlyweights = args.conv_onlyweights
    batchnorm_onlybias = args.batchnorm_onlybias
    act_layer = eval(args.act_layer)
    hidden_features = args.hidden_features

    ## Training Parameters
    initial_learningrate = args.init_learnrate
    batch_size = args.batch_size
    num_workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    train_per_val = args.TRAIN_PER_VAL
    
    ## Epoch Parameters
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    #############################################
    ## Check Devices
    #############################################
    print('\n')
    print('Slurm & Device Information')
    print('=========================================')
    print('Slurm Job ID:', os.environ['SLURM_JOB_ID'])
    print('Pytorch Cuda Available:', torch.cuda.is_available())
    print('GPU ID:', os.environ['SLURM_JOB_GPUS'])
    print('Number of System CPUs:', os.cpu_count())
    print('Number of CPUs per GPU:', os.environ['SLURM_JOB_CPUS_PER_NODE'])

    print('\n')
    print('Model Training Information')
    print('=========================================')

    #############################################
    ## Initialize Model
    #############################################
    model = PVI_SingleField_CNN(img_size=(1, 1700, 500),
                                size_threshold=size_threshold,
                                kernel=kernel,
                                features=features, 
                                interp_depth=interp_depth,
                                conv_onlyweights=conv_onlyweights,
                                batchnorm_onlybias=batchnorm_onlybias,
                                act_layer=act_layer,
                                hidden_features=hidden_features)
    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    ## Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=initial_learningrate,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.01)

    #############################################
    ## Initialize Loss
    #############################################
    loss_fn = nn.MSELoss(reduction='none')

    print('Model initialized.')

    #############################################
    ## Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model,
                                                          optimizer,
                                                          checkpoint)
        print('Model state loaded for continuation.')
    else:
        starting_epoch = 0

    #############################################
    ## Move model and optimizer state to GPU
    #############################################
    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    #############################################
    ## Initialize Data
    #############################################
    train_dataset = PVI_SingleField_DataSet(NC_NPZ_DIR,
                                            train_filelist,
                                            input_field=input_field,
                                            design_file=design_file)
    val_dataset = PVI_SingleField_DataSet(NC_NPZ_DIR,
                                          validation_filelist,
                                          input_field=input_field,
                                          design_file=design_file)
    test_dataset = PVI_SingleField_DataSet(NC_NPZ_DIR,
                                           test_filelist,
                                           input_field=input_field,
                                           design_file=design_file)

    print('Datasets initialized.')

    #############################################
    ## Training Loop
    #############################################
    ## Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch+cycle_epochs, total_epochs+1)

    for epochIDX in range(starting_epoch, ending_epoch):
        ## Setup Dataloaders
        train_dataloader = tr.make_dataloader(train_dataset,
                                              batch_size,
                                              train_batches,
                                              num_workers=num_workers)
        val_dataloader = tr.make_dataloader(val_dataset,
                                            batch_size,
                                            val_batches,
                                            num_workers=num_workers)

        ## Train an Epoch
        tr.train_scalar_csv_epoch(training_data=train_dataloader,
                                  validation_data=val_dataloader, 
                                  model=model,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn,
                                  epochIDX=epochIDX,
                                  train_per_val=train_per_val,
                                  train_rcrd_filename=trn_rcrd_filename,
                                  val_rcrd_filename=val_rcrd_filename,
                                  device=device)

        ## Print Summary Results
        print('Completed epoch '+str(epochIDX)+'...')

    ## Save Model Checkpoint
    print("Saving model checkpoint at end of epoch "+ str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    model.to('cpu')  
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to('cpu')

    # Save model and optimizer state in hdf5
    h5_name_str = 'study{0:03d}_modelState_epoch{1:04d}.hdf5'
    new_h5_path = os.path.join('./', h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(model, optimizer, epochIDX, new_h5_path)

    #############################################
    ## Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX+1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(new_h5_path,
                                               studyIDX,
                                               last_epoch=epochIDX)
        os.system(f'sbatch {new_slurm_file}')

    #############################################
    ## Run Test Set When Training is Complete
    #############################################
    if FINISHED_TRAINING:
        print("Testing Model . . .")
        test_dataloader = tr.make_dataloader(test_dataset, batch_size=batch_size)
        testbatch_ID = 0
        testing_dict = {"epoch": [],
                        "batch": [],
                        "truth": [],
                        "prediction": [],
                        "loss": []}

        # Move model back to GPU for final evaluation
        model.to(device)
        
        with torch.no_grad():
            for testdata in test_dataloader:
                testbatch_ID += 1
                truth, pred, loss = tr.eval_scalar_datastep(testdata, 
                                                            model,
                                                            loss_fn,
                                                            device)
                testing_dict = tr.append_to_dict(testing_dict,
                                                 testbatch_ID,
                                                 truth,
                                                 pred,
                                                 loss)

        ## Save Testing Info
        del testing_dict["epoch"]
        testingdf = pd.DataFrame.from_dict(testing_dict, orient='columns')
        test_csv_filename = 'study{0:03d}_test_results.csv'
        testingdf.to_csv(os.path.join('./', test_csv_filename.format(studyIDX)))
        print('Model testing results saved.')

        print('STUDY{0:03d} COMPLETE'.format(studyIDX))
