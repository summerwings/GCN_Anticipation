# implementation adapted from:
# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ResNet50 import ResNet50Model


class MultiStageModel(nn.Module):
    def __init__(self, num_classes_phase=10, num_stages=1, num_layers=5, num_f_maps=256, dim=2048, causal_conv =True, feature_extractor_pretrain=True):
        super(MultiStageModel, self).__init__()

        if feature_extractor_pretrain==True:
            # load model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_path = r'best_CNN_epoch.pth'
            self.CNN = ResNet50Model().to(self.device)
            self.CNN.load_state_dict(torch.load(model_path))
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.CNN = ResNet50Model().to(self.device)
        '''
        for p in self.parameters():
            p.requires_grad = False
        '''
        self.num_classes_phase = num_classes_phase  # 10
        self.num_stages = num_stages  # 2
        self.num_layers = num_layers  # 10
        self.num_f_maps = num_f_maps  # 32
        self.dim = dim  # 2048
        self.causal_conv = causal_conv

        self.Relu = nn.ReLU(inplace=True)

        self.stages_phase = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes_phase,
                                 self.num_classes_phase,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])





    def forward(self, x):

        x, _, _, _, _, _ = self.CNN(x)

        x = self.Relu(x)

        out_classes_phase = self.stage1_phase(x)
        feature = out_classes_phase.squeeze(-1).clone()


        outputs_classes_phase = out_classes_phase.unsqueeze(0)

        for s in self.stages_phase:
            out_classes_phase = s(F.softmax(out_classes_phase, dim=1))
            outputs_classes_phase = out_classes_phase


        return feature



class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_f_maps2,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.fcn_feature = nn.Linear(dim,num_f_maps)
        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_f_maps2, 1)


    def forward(self, x):
        T, C, M = x.size()
        x = x.permute(0, 2, 1).contiguous() # change the size to N, C, T
        out = self.conv_1x1(x)


        for layer in self.layers:
            out = layer(out)
        # out_classes = self.conv_out_classes(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
