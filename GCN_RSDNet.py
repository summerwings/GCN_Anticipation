# implementation adapted from:
# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from MSTCN import MultiStageModel
from STGCN import STGCN


class GCNProgressNet(nn.Module):

    def __init__(self, num_classes_phase=6,
                 num_stages=2,
                 num_layers=4,
                 num_f_maps=256,
                 dim=2048,
                 causal_conv =True,
                 in_channels=4,
                 num_class=5,
                 graph_args={'max_hop': 1, 'strategy': 'uniform', 'dilation': 1},
                 edge_importance_weighting=False,
                 attention= False,
                 fc_output_channel=256,
                 feature_extractor_pretrain=False,
                 stream = 'Graph',
                 GCN_mode = 'STGCN',
                 FPS = 2.5,
                 oFps = 25,
                 **kwargs,
                 ):

        super(GCNProgressNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stream = stream
        if self.stream == 'Both':
            self.MSTCN = MultiStageModel(num_classes_phase, num_stages, num_layers, num_f_maps, dim, causal_conv, feature_extractor_pretrain)

        if self.stream == 'BB':
            self.fc_bb_feature = nn.Linear(32 ,self.GCN.channel_n_3)

        if GCN_mode == 'STGCN':
            self.GCN = STGCN(in_channels, num_class, graph_args, edge_importance_weighting, attention)

        if self.stream == 'Both':
            self.fc_input_channel = self.GCN.channel_n_3 * 2
        else:
            self.fc_input_channel = self.GCN.channel_n_3

        self.GCN.channel_n_3 = fc_output_channel # the channel before FC layers

        self.FPS = FPS
        self.oFPS = oFps
        self.step =self.oFPS/self.FPS





        self.fc_stage_RSD = nn.Sequential(nn.Linear(self.fc_input_channel,
                                                    self.GCN.channel_n_3),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.GCN.channel_n_3, num_classes_phase*3))

        self.fc_tool_stage_RSD = nn.Sequential(nn.Linear(self.fc_input_channel,self.GCN.channel_n_3),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.GCN.channel_n_3, num_class*3), )





    def forward(self, x, bb):

        if self.stream == 'Both':
            output_feature1 = self.MSTCN(x)
            output_feature2= self.GCN(bb)

            # change the size from N, C, T to N, T, C
            if len(output_feature1.size()) < 3:
                output_feature1 = output_feature1.unsqueeze(-1)

            output_feature1 = output_feature1.permute(0, 2, 1).contiguous()

            # because there is the attention mech in stgcn so here we could just use mean
            output_feature2 = torch.mean(output_feature2, dim=-1)
            out_stem = torch.cat((output_feature1, output_feature2), dim=2)


        if self.stream == 'Visual':
            out_stem = self.MSTCN(x)
            out_stem = out_stem.permute(0, 2, 1).contiguous()

        if self.stream == 'Graph':
            out_stem = self.GCN(bb)
            out_stem = torch.mean(out_stem, dim=-1)

        if self.stream == 'BB':
            N, T, V, C = bb.size()
            bb = bb.view(N , T, V * C).contiguous()
            out_stem = self.fc_bb_feature(bb)

        N, T, C = out_stem.size()
        # fc to the cat output
        feature = out_stem.clone()


        stage_RSD = self.fc_stage_RSD(out_stem).squeeze(-1)
        tool_stage_RSD = self.fc_tool_stage_RSD(out_stem).squeeze(-1)






        return feature, stage_RSD, tool_stage_RSD


class attention_m(nn.Module):

    def __init__(self, n_channel):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(n_channel, n_channel),
            nn.ReLU(inplace=True),
            nn.Linear(n_channel, n_channel),
            nn.Sigmoid(),)

    def forward(self, x):
        N, T, C = x.size()
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = self.m(x)
        x = x.permute(0, 1, 2, 4, 3).contiguous()

        return x
