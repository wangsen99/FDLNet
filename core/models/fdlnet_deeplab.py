"""
 @Time    : 22/9/2
 @Author  : WangSen
 @Email   : wangsen@shu.edu.cn
 
 @Project : FDLNet
 @File    : fdlnet_deeplab.py
 @Function: FDLNet 
 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .segbase import SegBaseModel
from .fcn import _FCNHead
from .frelayer import LFE

from ..nn import _ConvBNReLU
__all__ = ['FDLNet', 'get_fdlnet', 'get_fdlnet_resnet101_citys']

class FDLNet(SegBaseModel):
    def __init__(self, nclass, criterion=None, backbone='resnet50', aux=False, pretrained_base=False, **kwargs):
        super(FDLNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.criterion = criterion
        self.fcm = _FDLHead(2048, 2048, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None)

        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['fcm', 'auxlayer'] if aux else ['fcm'])

    def forward(self, x, gts=None, segSize=None):
        size = x.size()[2:]
        outputs = []
        c1, c2, c3, c4 = self.base_forward(x)
        fcm= self.fcm(c4, c1)
        seg_out_final = F.interpolate(fcm, size, mode='bilinear', align_corners=True)
        
        outputs.append(seg_out_final)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        if self.training:
            return self.criterion(outputs, gts)
        else:
            return tuple(outputs)

class _FDLHead(nn.Module):
    def __init__(self, in_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FDLHead, self).__init__()
        c1_channels=256
        self.att = LFE(in_channels, dct_h=8, dct_w=8, frenum=8)
        self.ppm = _DeepLabHead(c1_channels=256, out_channels=512, **kwargs)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.fam =  _SFFHead(in_channels=2048, inter_channels=512, **kwargs)

        self.final_seg = nn.Sequential(
            _ConvBNReLU(512+48, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        fre = self.att(x) #B 2048 1 1

        f = self.ppm(x)
        fa = self.fam(f, fre)

        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        fa = F.interpolate(fa, size, mode='bilinear', align_corners=True)

        seg_out = self.final_seg(torch.cat([fa, c1], dim=1))

        return seg_out


class SFF(nn.Module):
    """ spatial frequency fusion module"""

    def __init__(self, in_channels, **kwargs):
        super(SFF, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, fre):
        batch_size, _, height, width = x.size()
        fre = fre.expand_as(x)
        feat_a = x.view(batch_size, -1, height * width) #B C H*W
        feat_f_transpose = fre.view(batch_size, -1, height * width).permute(0, 2, 1) #B H*W C
        attention = torch.bmm(feat_a, feat_f_transpose)  # B C C
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new) # B C C

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width) # B C H*W
        out = self.alpha * feat_e + x
        return out


class _SFFHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_SFFHead, self).__init__()
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.freatt = SFF(inter_channels, **kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
        )

    def forward(self, x, fre):
        feat_x = self.conv_x1(x)
        feat_f = self.conv_f1(fre)

        feat_p = self.freatt(feat_x, feat_f)
        feat_p = self.conv_p2(feat_p)

        return feat_p

class _DeepLabHead(nn.Module):
    def __init__(self, c1_channels=256, out_channels=512, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        
        self.block = nn.Sequential(
            _ConvBNReLU(256, out_channels, 3, padding=1, norm_layer=norm_layer),
            )

    def forward(self, x):
       
        x = self.aspp(x)
        
       
        return self.block(x)

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

def get_fdlnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='./',
            pretrained_base=True, **kwargs):

    from ..data.dataloader import datasets
    model = FDLNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(0)
        checkpoint = torch.load(get_model_file('fdlnet_deeplab', root=root),
                                    map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    return model



def get_fdlnet_resnet101_citys(**kwargs):
    return get_fdlnet('citys', 'resnet101', **kwargs)



if __name__ == '__main__':
    model = get_fdlnet_resnet101_citys()
    img = torch.randn(4, 3, 480, 480)
    output = model(img)
