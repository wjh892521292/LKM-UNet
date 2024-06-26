#2p,2w
from re import S
from xml.dom import xmlbuilder
import numpy as np
import torch
from torch import nn
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
import copy

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out
    
class BiPixelMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)


        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)


        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      
        # adjust the window size here to fit the feature map
        self.p = p*5
        self.p1 = 5*p
        self.p2 = 7*p
        self.p3 = 6*p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        if ll == 5: #3d
         
            Z,H,W = x.shape[2:]

            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_div = x.reshape(B, C, Z//self.p1, self.p1, H//self.p2, self.p2, W//self.p3, self.p3)
                x_div = x_div.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous().view(B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
            else:
                x_div = x

        elif ll == 4: #2d
            H,W = x.shape[2:]

            if H%self.p==0 and W%self.p==0:                
                x_div = x.reshape(B, C, H//self.p, self.p, W//self.p, self.p).permute(0, 3, 5, 1, 2, 4).contiguous().view(B*self.p*self.p, C, H//self.p, W//self.p)            
            else:
                x_div = x
        

        NB = x_div.shape[0]
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p1, self.p2, self.p3, C, NZ, NH, NW).permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if H%self.p==0 and W%self.p==0:
                x_out = x_out.transpose(-1, -2).reshape(B, self.p, self.p, C, NH, NW).permute(0, 3, 4, 1, 5, 2).contiguous().reshape(B, C, *img_dims)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        out = x_out + x

        return out


class BiWindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      

       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       


    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        ll = len(x.shape)
     
       

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]



        if ll == 5: #3d
            
            Z,H,W = x.shape[2:]

            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool3d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x

        elif ll == 4: #2d

            H,W = x.shape[2:]
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                pool_layer = nn.AvgPool2d(self.p, stride=self.p)
                x_div = pool_layer(x)
            else:
                x_div = x
        

      
        if ll == 5: #3d
            NZ,NH,NW = x_div.shape[2:]
        else:
            NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if ll == 5:
            if Z%self.p==0 and H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        if ll == 4:
            if self.p==0:
                self.p = 1
            if H%self.p==0 and W%self.p==0:
                unpool_layer = nn.Upsample(scale_factor=self.p, mode='nearest')
                x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW)
                x_out = unpool_layer(x_out)
            else:
                x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
                
        out = x_out + x

        return out


class ResidualBiMambaEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        w_mamba_layers = []
        mamba_layers = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            stage = StackedResidualBlocks(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]
            
            mamba_layers.append(BiPixelMambaLayer(input_channels, 2**( (n_stages-s+1)//2-1) ))
            w_mamba_layers.append(BiWindowMambaLayer(input_channels, 2**((n_stages-s+1)//2)//2 ))
            


        #self.stages = nn.Sequential(*stages)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.w_mamba_layers = nn.ModuleList(w_mamba_layers)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        #for s in self.stages:
        for s in range(len(self.stages)):
            #x = s(x)
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            x = self.w_mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output

class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualBiMambaEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks = n_conv_per_stage[s-1],
                conv_op = encoder.conv_op,
                input_channels = 2 * input_features_skip,
                output_channels = input_features_skip,
                kernel_size = encoder.kernel_sizes[-(s + 1)],
                initial_stride = 1,
                conv_bias = encoder.conv_bias,
                norm_op = encoder.norm_op,
                norm_op_kwargs = encoder.norm_op_kwargs,
                dropout_op = encoder.dropout_op,
                dropout_op_kwargs = encoder.dropout_op_kwargs,
                nonlin = encoder.nonlin,
                nonlin_kwargs = encoder.nonlin_kwargs,
            ))
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

class LKMUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualBiMambaEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_lkmunet_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'LKMUNet'
    network_class = LKMUNet
    kwargs = {
        'LKMUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == LKMUNet:
        model.apply(init_last_bn_before_add_to_0)

    return model
