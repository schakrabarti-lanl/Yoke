"""Definition of CNN interpretability network parts. Full NC CNN network is
built from these.

"""
####################################
## Packages
####################################
import math
import torch
import torch.nn as nn

from cnn_utils import conv2d_shape


####################################
## Interpretability Module
####################################
class CNN_Interpretability_Module(nn.Module):
    def __init__(self, 
                 img_size: tuple[int, int, int]=(1, 1700, 500),
                 kernel: int=5,
                 features: int=12, 
                 depth: int=12,
                 conv_onlyweights: bool=True,
                 batchnorm_onlybias: bool=True,
                 act_layer=nn.GELU):
        """Convolutional Neural Network Module that creates the "interpretability
        layers" Sequence of Conv2D, Batch Normalization, and Activation

        Args:
            img_size (tuple[int, int, int]): size of input (channels, height, width)
            kernel (int): size of square convolutional kernel
            features (int): number of features in the convolutional layers
            depth (int): number of interpretability blocks
            conv_onlyweights (bool): determines if convolutional layers learn 
                                     only weights or weights and bias
            batchnorm_onlybias (bool): determines if the batch normalization 
                                       layers learn only bias or weights and bias
            act_layer(nn.modules.activation): torch neural network layer class 
                                              to use as activation

        """

        super().__init__()

        self.img_size = img_size
        C, H, W = self.img_size
        self.kernel = kernel
        self.features = features
        self.depth = depth
        self.conv_weights = True 
        self.conv_bias = not conv_onlyweights
        self.batchnorm_weights = not batchnorm_onlybias
        self.batchnorm_bias = True

        ## Input Layers
        self.inConv = nn.Conv2d(in_channels=C,
                                out_channels=self.features, 
                                kernel_size=self.kernel, 
                                stride=1, 
                                padding='same',  # pads the input so the output
                                                 # has the shape as the input,
                                                 # stride=1 only
                                bias=self.conv_bias)

        normLayer = nn.BatchNorm2d(features)

        if not self.batchnorm_weights:
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False
            
        self.inNorm = normLayer
        self.inActivation = act_layer()

        ## Interpretability Layers 
        self.interpConv = nn.ModuleList()
        self.interpNorms = nn.ModuleList()
        self.interpActivations = nn.ModuleList()

        for i in range(self.depth - 1):
            interpLayer = nn.Conv2d(in_channels=self.features,
                                    out_channels=self.features, 
                                    kernel_size=self.kernel, 
                                    stride=1, 
                                    padding='same',  # pads the input so the
                                                     # output has the shape of
                                                     # the input, stride=1 only
                                    bias=self.conv_bias)
            
            self.interpConv.append(interpLayer)

            normLayer = nn.BatchNorm2d(features)

            # Necessary step to turn off only the scaling.
            if not self.batchnorm_weights:
                nn.init.constant_(normLayer.weight, 1)
                normLayer.weight.requires_grad = False

            self.interpNorms.append(normLayer)
            self.interpActivations.append(act_layer())

    def forward(self, x):
        ## Input Layers
        x = self.inConv(x)
        x = self.inNorm(x)
        x = self.inActivation(x)

        ## Interpretability Layers
        for i in range(self.depth - 1):
            x = self.interpConv[i](x)
            x = self.interpNorms[i](x)
            x = self.interpActivations[i](x)

        return x

    
####################################
## Reduction Module
####################################
class CNN_Reduction_Module(nn.Module):
    def __init__(self, 
                 img_size: tuple[int, int, int]=(1, 1700, 500),
                 size_threshold: tuple[int, int]=(8, 8),
                 kernel: int=5,
                 stride: int=2,
                 features: int=12, 
                 conv_onlyweights: bool=True,
                 batchnorm_onlybias: bool=True,
                 act_layer=nn.GELU):
        """Convolutional Neural Network Module that creates the "reduction layers"
        Sequence of Conv2D, Batch Normalization, and Activation

        Args:
            img_size (tuple[int, int, int]): size of input 
                                             (channels, height, width)
            size_threshold (tuple[int, int]): (approximate) size of final, 
                                              reduced image (height, width)
            kernel (int): size of square convolutional kernel
            stride (int): size of base stride for convolutional kernel
            features (int): number of features in the convolutional layers
            conv_onlyweights (bool): determines if convolutional layers learn 
                                     only weights or weights and bias
            batchnorm_onlybias (bool): determines if the batch normalization layers 
                                       learn only bias or weights and bias
            act_layer(nn.modules.activation): torch neural network layer class to 
                                              use as activation

        """
        
        super().__init__()
        
        self.img_size = img_size
        C, H, W = self.img_size
        self.size_threshold = size_threshold
        H_lim, W_lim = self.size_threshold
        self.kernel = kernel
        self.stride = stride
        self.features = features
        self.depth = 0  # initialize depth
        self.conv_weights = True 
        self.conv_bias = not conv_onlyweights
        self.batchnorm_weights = not batchnorm_onlybias
        self.batchnorm_bias = True

        ## Input Layers
        self.inConv = nn.Conv2d(in_channels=C,
                                out_channels=self.features, 
                                kernel_size=self.kernel, 
                                stride=self.stride, 
                                padding=self.stride,
                                padding_mode='zeros',
                                bias=self.conv_bias)
        
        normLayer = nn.BatchNorm2d(features)

        # Necessary step to turn off only the scaling.
        if not self.batchnorm_weights:
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

        self.inNorm = normLayer
        self.inActivation = act_layer()
        
        W, H, _ = conv2d_shape(w=W,
                               h=H,
                               k=self.kernel,
                               s_w=self.stride,
                               s_h=self.stride,
                               p_w=self.stride,
                               p_h=self.stride)

        self.depth += 1 

        ## Setup Reduction Layers
        self.reductionConv = nn.ModuleList()
        self.reductionNorms = nn.ModuleList()
        self.reductionActivations = nn.ModuleList()

        ## Reduction Layers
        while (W > W_lim or H > H_lim):
            ## Set Stride & Padding
            if W > W_lim:
                w_stride = self.stride
            else:
                w_stride = 1
            if H > H_lim:
                h_stride = self.stride
            else: 
                h_stride = 1
                
            w_pad = 2 * w_stride
            h_pad = 2 * h_stride

            ## Define Layers
            reduceLayer = nn.Conv2d(in_channels=self.features,
                                    out_channels=self.features, 
                                    kernel_size=self.kernel, 
                                    stride=(h_stride, w_stride), 
                                    padding=(h_pad, w_pad),
                                    padding_mode='zeros', 
                                    bias=self.conv_bias)
            
            self.reductionConv.append(reduceLayer)
            normLayer = nn.BatchNorm2d(features)

            if not self.batchnorm_weights:
                nn.init.constant_(normLayer.weight, 1)
                normLayer.weight.requires_grad = False

            self.reductionNorms.append(normLayer)
            self.reductionActivations.append(act_layer())

            ## Recalculate Size
            W, H, _ = conv2d_shape(w=W,
                                   h=H,
                                   k=self.kernel,
                                   s_w=w_stride,
                                   s_h=h_stride,
                                   p_w=w_pad,
                                   p_h=h_pad)

            self.depth += 1

        ## Define final size
        self.finalW = W
        self.finalH = H

    def forward(self, x):
        ## Input Layers
        x = self.inConv(x)
        x = self.inNorm(x)
        x = self.inActivation(x)

        ## Interpretability Layers
        for i in range(self.depth - 1):
            x = self.reductionConv[i](x)
            x = self.reductionNorms[i](x)
            x = self.reductionActivations[i](x)

        return x


class PVI_SingleField_CNN(nn.Module):
    def __init__(self, 
                 img_size: tuple[int, int, int]=(1, 1700, 500),
                 size_threshold: tuple[int, int]=(8, 8),
                 kernel: int=5,
                 features: int=12, 
                 interp_depth: int=12,
                 conv_onlyweights: bool=True,
                 batchnorm_onlybias: bool=True,
                 act_layer = nn.GELU,
                 hidden_features: int=20):
        """Convolutional Neural Network Model that uses a single PVI field to predict
        one scalar value. Constructed using both an interpretability block,
        defined above, and a reduction block.

        Args:
            img_size (tuple[int, int, int]): size of input (channels, height, width)
            size_threshold (tuple[int, int]): (approximate) size of reduced image 
                                              (height, width)
            kernel (int): size of square convolutional kernel
            features (int): number of features in the convolutional layers
            interp_depth (int): number of interpretability blocks
            conv_onlyweights (bool): determines if convolutional layers learn only 
                                     weights or weights and bias
            batchnorm_onlybias (bool): determines if the batch normalization layers 
                                       learn only bias or weights and bias
            act_layer (nn.modules.activation): torch neural network layer class to 
                                               use as activation
            hidden_features (int): number of hidden features in the fully connected 
                                   dense layer

        """
        super().__init__()
        self.img_size = img_size
        C, H, W = self.img_size
        self.size_threshold = size_threshold
        self.kernel = kernel
        self.features = features
        self.interp_depth = interp_depth
        self.hidden_features = hidden_features

        self.conv_onlyweights = conv_onlyweights
        self.conv_weights = True 
        self.conv_bias = not self.conv_onlyweights
        self.batchnorm_onlybias = batchnorm_onlybias
        self.batchnorm_weights = not self.batchnorm_onlybias
        self.batchnorm_bias = True

        self.interp_module = CNN_Interpretability_Module(img_size=self.img_size,
                                                         kernel=self.kernel,
                                                         features=self.features, 
                                                         depth=self.interp_depth,
                                                         conv_onlyweights=self.conv_onlyweights,
                                                         batchnorm_onlybias=self.batchnorm_onlybias,
                                                         act_layer=act_layer)

        self.reduction_module = CNN_Reduction_Module(img_size=(self.features, H, W),
                                                     size_threshold=self.size_threshold,
                                                     kernel=self.kernel,
                                                     features=self.features, 
                                                     conv_onlyweights=self.conv_onlyweights,
                                                     batchnorm_onlybias=self.batchnorm_onlybias,
                                                     act_layer=act_layer)
        
        self.reduction_depth = self.reduction_module.depth
        self.finalW = self.reduction_module.finalW
        self.finalH = self.reduction_module.finalH

        self.endConv = nn.Conv2d(in_channels=self.features,
                                 out_channels=self.features, 
                                 kernel_size=self.kernel, 
                                 stride=1, 
                                 padding='same',  # pads the input so the output
                                                  # has the shape as the input,
                                                  # stride=1 only
                                 bias=self.conv_bias)
        self.endConvActivation = act_layer()
        
        # Hidden Layer (Equivalent to a matched convolutional layer?)
        self.hidden = nn.Linear(self.finalH * self.finalW * self.features, self.hidden_features)
        self.hiddenActivation = act_layer()
        
        # Linear Output Layer
        self.linOut = nn.Linear(self.hidden_features, 1)

    def forward(self, x):
        x = self.interp_module(x)
        x = self.reduction_module(x)
        
        ## Final Convolution
        x = self.endConv(x)
        x = self.endConvActivation(x)
        x = torch.flatten(x, start_dim=1)
        
        ## Hidden Layer
        x = self.hidden(x)
        x = self.hiddenActivation(x)
        
        ## Linear Output Layer 
        x = self.linOut(x)
        x = torch.flatten(x, start_dim=1)
        
        return(x)


if __name__ == '__main__':
    """For testing and debugging.

    """

    # Excercise model setup
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

    pvi_CNN.eval()
    pvi_pred = pvi_CNN(pvi_input)

    
