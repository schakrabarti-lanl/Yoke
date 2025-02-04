"""Training for TCNN mapping LSC geometry to density image.

Actual training workhorse for Transpose CNN network mapping layered shaped
charge geometry parameters to density image.

We modify the training procedure to use multiple CUDA streams to load multiple
simultaneous batches onto the GPU.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn

from yoke.models.surrogateCNNmodules import tCNNsurrogate
from yoke.datasets.lsc_dataset import LSC_cntr2rho_DataSet
import yoke.torch_training_utils as tr
from src.yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = (
    "Trains Transpose-CNN to reconstruct density field of LSC simulation "
    "from contours and simulation time. This training uses network with "
    "no interpolation."
)
parser = argparse.ArgumentParser(
    prog="LSC Surrogate Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser.add_argument(
    "--LSC_DESIGN_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../../data_examples/"),
    help="Directory in which LSC design.txt file lives.",
)
parser.add_argument(
    "--design_file",
    action="store",
    type=str,
    default="design_lsc240420_SAMPLE.csv",
    help=".csv file that contains the truth values for data files",
)
parser.add_argument(
    "--LSC_NPZ_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../../data_examples/lsc240420/"),
    help="Directory in which LSC *.npz files lives.",
)
parser.add_argument(
    "--train_filelist",
    action="store",
    type=str,
    default="lsc240420_train_sample.txt",
    help="Path to list of files to train on.",
)
parser.add_argument(
    "--validation_filelist",
    action="store",
    type=str,
    default="lsc240420_val_sample.txt",
    help="Path to list of files to validate on.",
)
parser.add_argument(
    "--test_filelist",
    action="store",
    type=str,
    default="lsc240420_test_sample.txt",
    help="Path to list of files to test on.",
)


#############################################
#############################################
if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    # Model Parameters
    featureList = args.featureList
    linearFeatures = args.linearFeatures

    # Training Parameters
    initial_learningrate = args.init_learnrate
    batch_size = args.batch_size
    # Leave one CPU out of the worker queue. Not sure if this is necessary.
    num_workers = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])  # - 1
    train_per_val = args.TRAIN_PER_VAL

    # Epoch Parameters
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
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])

    print("\n")
    print("Model Training Information")
    print("=========================================")

    #############################################
    # Initialize Model
    #############################################

    model = tCNNsurrogate(
        input_size=29,
        linear_features=(7, 5, linearFeatures),
        initial_tconv_kernel=(5, 5),
        initial_tconv_stride=(5, 5),
        initial_tconv_padding=(0, 0),
        initial_tconv_outpadding=(0, 0),
        initial_tconv_dilation=(1, 1),
        kernel=(3, 3),
        nfeature_list=featureList,
        output_image_size=(1120, 800),
        act_layer=nn.GELU,
    )

    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    # Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_learningrate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # Move model and optimizer state to GPU
    #############################################
    if args.multigpu:
        model = nn.DataParallel(model)
    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    #############################################
    # Script and compile model on device
    #############################################
        if args.multigpu:
            compiled_model = model # jit compilation disabled in multi-gpu scenarios.
        else:
            scripted_model = torch.jit.script(model)

            # Model compilation has some interesting parameters to play with.
            #
            # NOTE: Compiled model is not able to be loaded from checkpoint for some
            # reason.
            compiled_model = torch.compile(
                scripted_model,
                fullgraph=True,  # If TRUE, throw error if
                # whole graph is not
                # compileable.
                mode="reduce-overhead",
            )  # Other compile
            # modes that may
            # provide better
            # performance

    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_cntr2rho_DataSet(args.LSC_NPZ_DIR, train_filelist, design_file)
    val_dataset = LSC_cntr2rho_DataSet(
        args.LSC_NPZ_DIR, validation_filelist, design_file
    )
    test_dataset = LSC_cntr2rho_DataSet(args.LSC_NPZ_DIR, test_filelist, design_file)

    print("Datasets initialized.")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset, batch_size, train_batches, num_workers=num_workers
    )
    val_dataloader = tr.make_dataloader(
        val_dataset, batch_size, val_batches, num_workers=num_workers
    )

    # Create multiple CUDA streams
    #
    # Note: There is a weird behavior when using cuda streams. Once the first
    # cycle of epochs completes the next cycle starts on a second GPU and the
    # next job uses 2 GPUs.
    Nstreams = 8
    streams = [torch.cuda.Stream() for _ in range(Nstreams)]

    # Iterate over epochs
    for epochIDX in range(starting_epoch, ending_epoch):
        # Time each epoch and print to stdout
        startTime = time.time()

        # Explicitly run training and validation loop so CUDA stream and graph
        # can be modified.
        trainbatches = len(train_dataloader)
        valbatches = len(val_dataloader)
        trainbatch_ID = 0
        valbatch_ID = 0

        train_batchsize = train_dataloader.batch_size
        val_batchsize = val_dataloader.batch_size

        trn_rcrd_filename = trn_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        # Train on all training samples
        with open(trn_rcrd_filename, "a") as train_rcrd_file:
            # Set model to train
            compiled_model.train()
            stream_idx = 0

            for traindata in train_dataloader:
                trainbatch_ID += 1

                # Extract data
                (inpt, truth) = traindata
                inpt = inpt.to(device, non_blocking=True)
                truth = truth.to(device, non_blocking=True)

                # Use alternating streams for overlapping data transfer and computation
                stream = streams[stream_idx]
                stream_idx = (stream_idx + 1) % len(streams)

                # Perform a forward pass
                # NOTE: If training on GPU model should have already been moved to GPU
                # prior to initalizing optimizer.
                with torch.cuda.stream(stream):
                    pred = compiled_model(inpt)
                    train_loss = loss_fn(pred, truth)

                    # Perform backpropagation and update the weights
                    optimizer.zero_grad(set_to_none=True)  # Possible speed-up
                    train_loss.mean().backward()
                    optimizer.step()

                # Synchronize the stream to ensure computation is finished
                # before the next iteration
                stream.synchronize()

                # truth, pred, train_loss = train_array_datastep(traindata,
                #                                                compiled_model,
                #                                                optimizer,
                #                                                loss_fn,
                #                                                device)

                template = "{}, {}, {}"
                for i in range(train_batchsize):
                    print(
                        template.format(
                            epochIDX,
                            trainbatch_ID,
                            train_loss.cpu().detach().numpy().flatten()[i],
                        ),
                        file=train_rcrd_file,
                    )

        # Evaluate on all validation samples
        if epochIDX % train_per_val == 0:
            print("Validating...", epochIDX)
            val_rcrd_filename = val_rcrd_filename.replace(
                "<epochIDX>", f"{epochIDX:04d}"
            )
            with open(val_rcrd_filename, "a") as val_rcrd_file:
                # Set model to eval
                compiled_model.eval()

                with torch.no_grad():
                    for valdata in val_dataloader:
                        valbatch_ID += 1

                        # Extract data
                        (inpt, truth) = valdata
                        inpt = inpt.to(device, non_blocking=True)
                        truth = truth.to(device, non_blocking=True)

                        # Perform a forward pass
                        pred = compiled_model(inpt)
                        val_loss = loss_fn(pred, truth)

                        # truth, pred, val_loss = eval_array_datastep(valdata,
                        #                                             compiled_model,
                        #                                             loss_fn,
                        #                                             device)

                        template = "{}, {}, {}"
                        for i in range(val_batchsize):
                            print(
                                template.format(
                                    epochIDX,
                                    valbatch_ID,
                                    val_loss.cpu().detach().numpy().flatten()[i],
                                ),
                                file=val_rcrd_file,
                            )

        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print("Completed epoch " + str(epochIDX) + "...")
        print("Epoch time:", epoch_time)

    # Save Model Checkpoint
    print("Saving model checkpoint at end of epoch " + str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    compiled_model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state in hdf5
    h5_name_str = "study{0:03d}_modelState_epoch{1:04d}.hdf5"
    new_h5_path = os.path.join("./", h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(
        compiled_model, optimizer, epochIDX, new_h5_path, compiled=True
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            new_h5_path, studyIDX, last_epoch=epochIDX
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
