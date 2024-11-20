"""Script to start training a study."""

####################################
# Packages
####################################
import os
import shutil
import argparse
import numpy as np
import pandas as pd


####################################
# Helper Function
####################################
def replace_keys(study_dict: dict, data: str) -> str:
    """Function to replace "key" values in a string with dictionary values.

    Args:
        study_dict (dict): dictonary of keys and values to replace
        data (str): data to replace keys in

    Returns:
        data (str): data with keys replaced

    """
    for key, value in study_dict.items():
        if key == "studyIDX":
            data = data.replace(f"<{key}>", f"{value:03d}")
        elif isinstance(value, np.float64) or isinstance(value, float):
            data = data.replace(f"<{key}>", f"{value:5.4f}")
        elif isinstance(value, np.int64) or isinstance(value, int):
            data = data.replace(f"<{key}>", f"{value:d}")
        elif isinstance(value, str):
            data = data.replace(f"<{key}>", f"{value}")
        elif isinstance(value, np.bool_) or isinstance(value, bool):
            data = data.replace(f"<{key}>", f"{str(value)}")
        else:
            print("Key is", key, "with value of", value, "with type", type(value))
            raise ValueError("Unrecognized datatype in hyperparameter list.")

    return data


####################################
# Process Hyperparameters
####################################
# .csv argparse argument
descr_str = "Starts execution of training harness"
parser = argparse.ArgumentParser(prog="HARNESS START", description=descr_str)

parser.add_argument(
    "--csv",
    action="store",
    type=str,
    default="./hyperparameters.csv",
    help="CSV file containing study hyperparameters",
)

parser.add_argument(
    "--rundir",
    action="store",
    type=str,
    default="./runs",
    help=("Directory to create study directories within. This should be a softlink to "
          "somewhere with a lot of drive space."),
)

parser.add_argument(
    "--cpFile",
    action="store",
    type=str,
    default="./cp_files.txt",
    help=("Name of text file containing local files that should be copied to the "
          "study directory."),
)

args = parser.parse_args()

training_input_tmpl = "./training_input.tmpl"
training_slurm_tmpl = "./training_slurm.tmpl"
training_START_input = "./training_START.input"
training_START_slurm = "./training_START.slurm"

# List of files to copy
with open(args.cpFile, 'r') as cp_text_file:
    cp_file_list = [line.strip() for line in cp_text_file]

# Process Hyperparmaeters File
studyDF = pd.read_csv(
    args.csv, sep=",", header=0, index_col=0, comment="#", engine="python"
)
varnames = studyDF.columns.values
idxlist = studyDF.index.values

# Save Hyperparameters to list of dictionaries
studylist = []
for i in idxlist:
    studydict = {}
    studydict["studyIDX"] = int(i)

    for var in varnames:
        studydict[var] = studyDF.loc[i, var]

    studylist.append(studydict)

####################################
# Run Studies
####################################
# Iterate Through Dictionary List to Run Studies
for k, study in enumerate(studylist):
    # Make Study Directory
    studydirname = args.rundir + "/study_{:03d}".format(study["studyIDX"])

    if not os.path.exists(studydirname):
        os.makedirs(studydirname)

    # Make new training_input.tmpl file
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    training_input_data = replace_keys(study, training_input_data)
    training_input_filepath = os.path.join(studydirname, "training_input.tmpl")

    with open(training_input_filepath, "w") as f:
        f.write(training_input_data)

    # Make new training_slurm.tmpl file
    with open(training_slurm_tmpl) as f:
        training_slurm_data = f.read()

    training_slurm_data = replace_keys(study, training_slurm_data)
    training_slurm_filepath = os.path.join(studydirname, "training_slurm.tmpl")

    with open(training_slurm_filepath, "w") as f:
        f.write(training_slurm_data)

    # Make new training_START.input file
    with open(training_START_input) as f:
        START_input_data = f.read()

    START_input_data = replace_keys(study, START_input_data)
    START_input_name = "study{:03d}_START.input".format(study["studyIDX"])
    START_input_filepath = os.path.join(studydirname, START_input_name)

    with open(START_input_filepath, "w") as f:
        f.write(START_input_data)

    # Make a new training_START.slurm file
    with open(training_START_slurm) as f:
        START_slurm_data = f.read()

    START_slurm_data = replace_keys(study, START_slurm_data)
    START_slurm_name = "study{:03d}_START.slurm".format(study["studyIDX"])
    START_slurm_filepath = os.path.join(studydirname, START_slurm_name)

    with open(START_slurm_filepath, "w") as f:
        f.write(START_slurm_data)

    # Copy files to study directory from list
    for f in cp_file_list:
        shutil.copy(f, studydirname)
    
    # Submit Job
    os.system((f"cd {studydirname}; sbatch {START_slurm_name}; "
               f"cd {os.path.dirname(__file__)}"))
