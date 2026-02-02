'''     
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/TPAMI2025-MSBlock.png
论文链接：https://arxiv.org/abs/2308.05480
'''    

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')

import warnings  
warnings.filterwarnings('ignore')
from calflops import calculate_flops 

import torch     
import torch.nn as nn
   
from engine.extre_module.ultralytics_nn.conv import Conv

__all__ = ['MS_Stage']

class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k) -> None:    
        super().__init__()
        
        self.in_conv = Conv(inc, ouc, 1)   
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)   
     
    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))  

class MSBlock(nn.Module):   
    def __init__(self, inc, ouc, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3, in_down_ratio=2.) -> None:
        super().__init__()
        
        in_channel = int(inc * in_expand_ratio // in_down_ratio)  
        self.mid_channel = in_channel // len(kernel_sizes) 
        groups = int(self.mid_channel * mid_expand_ratio) 
        self.in_conv = Conv(inc, in_channel)
   
        self.mid_convs = []   
        for kernel_size in kernel_sizes:    
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(int(layers_num))]    
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)    
 
        self.attention = None    
    
    def forward(self, x):     
        out = self.in_conv(x)
        channels = []  
        for i,mid_conv in enumerate(self.mid_convs):     
            channel = out[:,i * self.mid_channel:(i+1) * self.mid_channel,...]
            if i >= 1:  
                channel = channel + channels[i-1]
            channel = mid_conv(channel)
            channels.append(channel)    
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)  
        return out
  
class MS_Stage(nn.Module):   
    def __init__(self, inc, ouc, downsample=True, deep_conv=False, layers_num=3, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., in_down_ratio=2.) -> None:
        super().__init__()
        
        # 1. 下采样层（可选）    
        if downsample: 
            if deep_conv:
                self.downsample = Conv(inc, inc, k=3, s=2, p=1, g=inc)
            else:
                self.downsample = Conv(inc, inc, k=3, s=2, p=1)
        else:
            self.downsample = nn.Identity()  # 如果不下采样，则直接返回输入  
        self.blocks = MSBlock(inc, ouc, kernel_sizes, in_expand_ratio, mid_expand_ratio, layers_num, in_down_ratio)
    def forward(self, x):     
        out = self.downsample(x)
        out = self.blocks(out)
        return out

if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)     

    module = MSBlock(in_channel, out_channel, kernel_sizes=[1, 3, 3], layers_num=3).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)