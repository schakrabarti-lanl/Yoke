"""Program to read in an NPZ file and print the keys within. Can also create a
plot of one of the arrays stored within the file.

"""

import os
import argparse
import numpy as np

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib
#matplotlib.use('MacOSX')
#matplotlib.use('pdf')
#matplotlib.use('QtAgg')
# Get rid of type 3 fonts in figures
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
# Ensure LaTeX font
font = {'family': 'serif'}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (6, 6)
from mpl_toolkits.axes_grid1 import make_axes_locatable


###################################################################
# Define command line argument parser
descr_str = 'Read in single NPZ file and plot field.'
parser = argparse.ArgumentParser(prog='Plot field from NPZ',
                                 description=descr_str)

# Example: lsc_nonconvex_pvi_idx00115.npz
# indir
parser.add_argument('--indir', '-D',
                    action='store',
                    type=str,
                    default='/data2/lsc240420',
                    help='Directory to find NPZ files.')

# outdir
parser.add_argument('--outdir', '-O',
                    action='store',
                    type=str,
                    default='./',
                    help='Directory to output images to.')

# run index
parser.add_argument('--runID', '-R',
                    action='store',
                    type=str,
                    default='lsc240420_id_00101_pvi',
                    help='Run identifier.')

# PVI index
parser.add_argument('--pviIDX', '-I',
                    action='store',
                    type=int,
                    default=0,
                    help='PVI index to plot, [0-100]')

# Hydro-dynamic field from PVI
parser.add_argument('--field', '-F',
                    action='store',
                    type=str,
                    default=None,
                    help='Depends on keys stored in file. Use -K option to print keys.')
                                
parser.add_argument('--keys', '-K',
                    action='store_true',
                    help='Flag to print keys of NPZ file.')

parser.add_argument('--save', '-S',
                    action='store_true',
                    help='Flag to save image.')


def print_NPZ_keys(npzfile='./lsc_nonconvex_pvi_idx00115.npz'):
    """Print keys of NPZ file.

    """

    NPZ = np.load(npzfile)
    print('NPZ file keys:')
    for key in NPZ.keys():
        print(key)

    NPZ.close()
    
    return


def singlePVIarray(npzfile='./lsc_nonconvex_pvi_idx00115.npz',
                   FIELD='rho'):
    """Function to grab single array from NPZ.

    Args:
       indir (str): File name for NPZ.
       FIELD (str): Field to return array for.

    Returns:
       field (np.array): Array of hydro-dynamic field for plotting

    """

    NPZ = np.load(npzfile)
    arrays_dict = dict()
    for key in NPZ.keys():
        arrays_dict[key] = NPZ[key]

    NPZ.close()

    return arrays_dict[FIELD]


if __name__ == '__main__':
    # Parse commandline arguments
    args_ns = parser.parse_args()

    # Assign command-line arguments
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    pviIDX = args_ns.pviIDX
    FIELD = args_ns.field
    KEYS = args_ns.keys
    SAVEFIG = args_ns.save

    # Assemble filename
    # Example: lsc_nonconvex_pvi_idx00115.npz
    npzfile = os.path.join(indir, runID+'_idx%05d.npz' % pviIDX)
    print('filename:', npzfile)
    
    if KEYS:
        print_NPZ_keys(npzfile=npzfile)
    else:
        # Get the fields
        Hfield = singlePVIarray(npzfile=npzfile,
                                FIELD=FIELD)
        simtime = singlePVIarray(npzfile=npzfile,
                                 FIELD='sim_time')
        Rcoord = singlePVIarray(npzfile=npzfile,
                                FIELD='Rcoord')
        Zcoord = singlePVIarray(npzfile=npzfile,
                                FIELD='Zcoord')

        Hfield = np.concatenate((np.fliplr(Hfield), Hfield), axis=1)
        print('Shape of Hfield: ', Hfield.shape)

        print(Hfield.shape)

        # Plot normalized radiograph and density field for diagnostics.
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        img1 = ax1.imshow(Hfield,
                          aspect='equal',
                          extent=[-Rcoord.max(),
                                  Rcoord.max(),
                                  Zcoord.min(),
                                  Zcoord.max()],
                          origin='lower',
                          cmap='cividis' if FIELD=='pRad' else 'jet')
        ax1.set_ylabel("Z-axis (um)", fontsize=16)                 
        ax1.set_xlabel("R-axis (um)", fontsize=16)
        ax1.set_title('T={:.2f}us'.format(float(simtime)), fontsize=18)

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='10%', pad=0.1)
        fig1.colorbar(img1,
                      cax=cax1).set_label(f'{FIELD}',
                                          fontsize=14)

        if SAVEFIG:
            fig1.savefig(os.path.join(outdir,
                                      f'{runID}_idx{pviIDX:05d}_{FIELD}.png'), 
                         bbox_inches='tight')
        else:
            plt.show()

