import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime
import torch.nn.functional as F

class ResNet50Model(nn.Module):
    def __init__(self, hparams=None, phase_num = 7, tool_num = 5):
        super(ResNet50Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hparams == None:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        else:
            self.model = models.resnet50(pretrained=hparams)
        # replace final layer with number of labels
        self.phase_num = phase_num
        self.tool_num = tool_num


        self.model.fc = Identity()
        self.fc1_c = 2048
        self.fc2_c = 512
        self.fc3_c = 128

        self.fc_pro = nn.Sequential(nn.Linear(self.fc1_c,
                                              self.fc2_c),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.fc2_c, 1), )

        self.fc_RSD = nn.Sequential(nn.Linear(self.fc1_c,
                                              self.fc2_c),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.fc2_c, 1), )

        self.fc_stage = nn.Sequential(nn.Linear(self.fc1_c,
                                                    self.fc2_c),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.fc2_c, self.phase_num), )

        self.fc_stage_RSD = nn.Sequential(nn.Linear(self.fc1_c,
                                                    self.fc2_c),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.fc2_c, self.phase_num*3),)

        self.fc_tool_stage_RSD = nn.Sequential(nn.Linear(self.fc1_c,
                                                    self.fc2_c),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.fc2_c, self.tool_num*3), )


    def forward(self, x):
        visual_features = []
        for b in range(x.size(0)):
            v_f = self.model(x[b,:,:,:,:])
            visual_features.append(v_f.cpu().detach().numpy())
        visual_features = torch.tensor(np.array(visual_features)).float().to(self.device)

        out_stem = visual_features
        feature = out_stem.clone()
        # output
        Progress = self.fc_pro(out_stem).squeeze(-1)
        stage = self.fc_stage(out_stem).squeeze(-1)
        stage_RSD = self.fc_stage_RSD(out_stem).squeeze(-1)
        tool_stage_RSD = self.fc_tool_stage_RSD(out_stem).squeeze(-1)
        RSD = self.fc_RSD(out_stem).squeeze(-1)

        return feature, stage, stage_RSD, tool_stage_RSD, RSD, Progress


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x