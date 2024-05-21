"""Functions and classes for torch DataSets which sample the Layered Shaped
Charge data, *lsc240420*.

"""

####################################
## Packages
####################################
import os
import sys
import glob
import random
import typing
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NoneStr = typing.Union[None, str]


################################################
## Functions for returning the run *key* from
## the npz-file name
################################################
def LSCnpz2key(npz_file: str):
    """Function to extract simulation *key* from the name of an .npz file. 

    A study key looks like **lsc240420_id00001** and a NPZ filename is like 
    **lsc240420_id00001_pvi_idx00000.npz**

    Args:
        npz_file (str): file path from working directory to .npz file
    
    Returns: 
        key (str): The correspond simulation key for the NPZ file.

    """

    key = npz_file.split('/')[-1].split('_pvi_')[0]

    return key


def LSCcsv2bspline_pts(design_file: str, key: str):
    """Function to extract the B-spline nodes from the design .csv file given the
    study key.
        
    Args:
        csv_file (str): file path from working directory to the .csv design file
        key (str): The study information for a given simulation; of the 
                   form *lsc240420_id?????*
        
    Returns:
        bspline_pts (numpy array): The B-spline nodes defining the geometry of 
                                   the Layered Shaped Charge

    """
    
    design_df = pd.read_csv(design_file,
                            sep=',',
                            header=0,
                            index_col=0,
                            engine='python')
    
    # removed spaces from headers
    for col in design_df.columns:
        design_df.rename(columns={col: col.strip()}, inplace=True)

    bspline_pts = design_df.loc[key, 'sa1':'ct7'].values

    return bspline_pts.astype(float)


def LSCread_npz(npz: np.lib.npyio.NpzFile, field: str):
    """Function to extract a value corresponding to an NPZ key.

    Args:
        npz (np.lib.npyio.NpzFile): a loaded .npz file
        field (str): name of field to extract

    """

    return npz[field]


####################################
## DataSet Class
####################################
class LSC_cntr2rho_DataSet(Dataset):
    def __init__(self,
                 filelist: str,
                 design_file: str):
        """The definition of a dataset object for the *Layered Shaped Charge* data
        which produces pairs of B-spline contour-node vectors and simulation times
        together with an average density field.

        Args:
            filelist (str): Text file listing file names to read
            design_file (str): .csv file with master design study parameters

        """

        ## Model Arguments 
        self.filelist = filelist
        self.design_file = design_file

        ## Create filelist
        with open(filelist, 'r') as f:
            self.filelist = [line.rstrip() for line in f]
            
        self.Nsamples = len(self.filelist)

    def __len__(self):
        """Return number of samples in dataset.

        """

        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given index.

        """

        ## Get the input image
        filepath = self.filelist[index]
        npz = np.load(filepath)
        true_image = npz['av_density']
        true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
        nY, nX = true_image.shape
        true_image = true_image.reshape((1, nY, nX))
        print('type true_image:', type(true_image))
        true_image = torch.tensor(true_image).to(torch.float32)

        ## Get the contours and sim_time
        sim_key = LSCnpz2key(filepath)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        sim_time = npz['sim_time']
        npz.close()

        sim_params = np.append(Bspline_nodes, sim_time)
        nSP = sim_params.shape[0]
        sim_params = sim_params.reshape((1, nSP))
        sim_params = torch.from_numpy(sim_params).to(torch.float32)
        
        return sim_params, true_image

    # def __getitems__(self, indices):
    #     """A mysterious method that PyTorch may have support for and documentation
    #     implies may produce a speed up that returns a batch as a list of tensors
    #     instead of a single tensor.

    #     """


if __name__ == '__main__':
    """For testing and debugging.

    """

    # Imports for plotting
    # To view possible matplotlib backends use
    # >>> import matplotlib
    # >>> bklist = matplotlib.rcsetup.interactive_bk
    # >>> print(bklist)
    import matplotlib
    #matplotlib.use('MacOSX')
    matplotlib.use('TkAgg')
    # Get rid of type 3 fonts in figures
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    # Ensure LaTeX font
    font = {'family': 'serif'}
    plt.rc('font', **font)
    plt.rcParams['figure.figsize'] = (6, 6)
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    # Test key
    npz_filename = '/data2/lsc240420/lsc240420_id00001_pvi_idx00000.npz'
    print('LSC NPZ filename:', npz_filename)
    LSCkey = LSCnpz2key(npz_filename)
    print('LSC key:', LSCkey)

    # Test B-spline retrieval
    csv_filename = '/data2/design_lsc240420_MASTER.csv'
    bspline_pts = LSCcsv2bspline_pts(csv_filename, LSCkey)
    print('Shape of B-spline points:', bspline_pts.shape)
    print('B-spline points:', bspline_pts)
    
    filelist = '/data2/yoke/filelists/lsc240420_test_10pct.txt'
    LSC_ds = LSC_cntr2rho_DataSet(filelist,
                                  csv_filename)
    sampIDX = 150
    sim_params, true_image = LSC_ds.__getitem__(sampIDX)
    
    print('Shape of true_image tensor: ', true_image.shape)
    print('Shape of sim_params tensor: ', sim_params.shape)
    print('sim_params:', sim_params)
    
    sim_params = sim_params.numpy()
    true_image = np.squeeze(true_image.numpy())
    
    # Plot normalized radiograph and density field for diagnostics.
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    img1 = ax1.imshow(true_image,
                      aspect='equal',
                      origin='lower',
                      cmap='jet')
    ax1.set_ylabel("Z-axis", fontsize=16)                 
    ax1.set_xlabel("R-axis", fontsize=16)
    ax1.set_title('Time={:.3f}us'.format(sim_params[0, -1]), fontsize=18)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='10%', pad=0.1)
    fig1.colorbar(img1,
                  cax=cax1).set_label('Density',
                                      fontsize=14)

    plt.show()
    
