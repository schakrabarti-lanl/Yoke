"""Collection of helper functions related to yoke command line interface (CLI)."""

import numpy as np


####################################
# Miscellaneous helper functions.
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
