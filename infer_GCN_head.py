import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from C80_video_dataloader import  SurgeryVideoDataset
from GCN_RSDNet import GCNProgressNet
from Loss import attention_loss, framewise_ce, anticipation_mae,class_wise_anticipation_mae

from natsort import natsorted
import glob

class infer_GCN(nn.Module):

    def __init__(self,
                 test_dir = r'Sample/Test',
                 ):

        super(infer_GCN, self).__init__()
        self.test_dir = test_dir

    def infer(self):

        # load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GCNProgressNet().to(device)
        model_path = r'best_PNet_epoch.pth'
        model.load_state_dict(torch.load(model_path))

        loss_RSD = nn.L1Loss()

        loss_anticipation_train = anticipation_mae().to(device)
        loss_anticipation = anticipation_mae().to(device)
        loss_anticipation_2 = anticipation_mae(h=3000).to(device)
        loss_anticipation_3 = anticipation_mae(h=4500).to(device)

        class_wise_loss_anticipation_loss = class_wise_anticipation_mae().to(device)
        class_wise_loss_anticipation_loss_2 = class_wise_anticipation_mae(h=3000).to(device)
        class_wise_loss_anticipation_loss_3 = class_wise_anticipation_mae(h=4500).to(device)

        tmp_RSD_val = []

        tmp_tool_stageRSD_loss_values_val = []
        tmp_stageRSD_loss_values_val = []
        tmp_total_loss_val = []
        # validation
        wtmp_stageRSD_loss_values_2_val = []
        wtmp_tool_stageRSD_loss_values_2_val = []

        wtmp_stageRSD_loss_values_3_val = []
        wtmp_tool_stageRSD_loss_values_3_val = []

        wtmp_stageRSD_loss_values_5_val = []
        wtmp_tool_stageRSD_loss_values_5_val = []


        tmp_stageRSD_loss_values_2_val = []
        tmp_tool_stageRSD_loss_values_2_val = []

        tmp_stageRSD_loss_values_3_val = []
        tmp_tool_stageRSD_loss_values_3_val = []

        tmp_stageRSD_loss_values_5_val = []
        tmp_tool_stageRSD_loss_values_5_val = []


        ptmp_stageRSD_loss_values_2_val = []
        ptmp_tool_stageRSD_loss_values_2_val = []

        ptmp_stageRSD_loss_values_3_val = []
        ptmp_tool_stageRSD_loss_values_3_val = []

        ptmp_stageRSD_loss_values_5_val = []
        ptmp_tool_stageRSD_loss_values_5_val = []

        etmp_tool_stageRSD_loss_values_2_val = []
        etmp_stageRSD_loss_values_2_val = []

        etmp_tool_stageRSD_loss_values_3_val = []
        etmp_stageRSD_loss_values_3_val = []

        etmp_tool_stageRSD_loss_values_5_val = []
        etmp_stageRSD_loss_values_5_val = []


        tool_wise_loss_2  = [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]
        tool_wise_loss_3 =  [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]
        tool_wise_loss  =  [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]


        phase_wise_loss_2  =  [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
        phase_wise_loss_3 = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
        phase_wise_loss  =  [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]

        # action record
        action_tool_list = []
        action_phase_list = []

        t_data_loader = DataLoader(SurgeryVideoDataset(self.test_dir), shuffle=False)

        time_start = time.time()
        for t, t_data in enumerate(t_data_loader):
            logging.info('batch {}'.format(t))
            model.eval()
            t_data = t_data

            # define and squeeze the dim from 1 b n n n to b n n n n
            data = t_data['video'].to(device)
            bb = t_data['boxes'].to(device)
            tmp_stage_RSD = t_data['stage_RSD'][:, :, 1:].to(device)
            tmp_tool_RSD = t_data['tool_RSD'].to(device)
            tmp_RSD = t_data['RSD'].to(device)

            output_feature, tmp_output_stg_RSD, tmp_output_tool_stg_RSD, tmp_output_RSD, frames = self.infer_one_clip(model, data, bb)

            if t == 0:
                output_stg_RSD = tmp_output_stg_RSD
                output_tool_stg_RSD = tmp_output_tool_stg_RSD
                output_RSD = tmp_output_RSD


                stage_RSD = tmp_stage_RSD
                tool_RSD = tmp_tool_RSD
                RSD = tmp_RSD


                frames_total = frames

            else:
                output_stg_RSD = torch.cat((output_stg_RSD, tmp_output_stg_RSD), dim=1)
                output_tool_stg_RSD = torch.cat((output_tool_stg_RSD, tmp_output_tool_stg_RSD), dim=1)
                output_RSD = torch.cat((output_RSD, tmp_output_RSD), dim=1)

                stage_RSD = torch.cat((stage_RSD, tmp_stage_RSD), dim=1)
                tool_RSD = torch.cat((tool_RSD, tmp_tool_RSD), dim=1)
                RSD = torch.cat((RSD, tmp_RSD), dim=1)

                frames_total += frames

        time_end = time.time()

        time_cost_total = time_end - time_start
        logging.info('time cost {}'.format(time_cost_total / frames_total))

        # RSD
        loss_allRSD = loss_RSD(output_RSD, RSD)
        loss_allRSD_minute = loss_allRSD / 25 / 60
        tmp_RSD_val.append(loss_allRSD_minute.item())

        # tool stage rsd
        _, loss_tool_stg_rsd_2, _, _ = loss_anticipation_2(output_tool_stg_RSD[:, :, :5], tool_RSD)
        _, loss_tool_stg_rsd_3, _, _ = loss_anticipation_3(output_tool_stg_RSD[:, :, 5:10], tool_RSD)
        _, loss_tool_stg_rsd, _, _ = loss_anticipation_train(output_tool_stg_RSD[:, :, 10:], tool_RSD)

        loss_tool_stg_all = loss_tool_stg_rsd + loss_tool_stg_rsd_2 + loss_tool_stg_rsd_3
        loss_tool_stg_minute = loss_tool_stg_all / 25 / 60
        tmp_tool_stageRSD_loss_values_val.append(loss_tool_stg_minute.item())
        # stage rsd
        _, loss_stg_rsd_2, _, _ = loss_anticipation_train(output_stg_RSD[:, :, :6], stage_RSD)
        _, loss_stg_rsd_3, _, _ = loss_anticipation_train(output_stg_RSD[:, :, 6:12], stage_RSD)
        _, loss_stg_rsd, _, _ = loss_anticipation_train(output_stg_RSD[:, :, 12:], stage_RSD)
        loss_stg_rsd_all = loss_stg_rsd + loss_stg_rsd_2 + loss_stg_rsd_3
        loss_stg_minute = loss_stg_rsd_all / 25 / 60
        tmp_stageRSD_loss_values_val.append(loss_stg_minute.item())

        # total loss
        loss_total = loss_tool_stg_minute + loss_stg_minute  # + loss_stg + loss_Elapsed_minute #att_loss+ loss_stg_minute + att_loss
        tmp_total_loss_val.append(loss_total.item())

        # 2min
        # tool stage rsd
        wloss_tool_stg_rsd_2, loss_tool_stg_rsd_2, ploss_tool_stg_rsd_2, eloss_tool_stg_rsd_2 = loss_anticipation_2(
            output_tool_stg_RSD[:, :, :5], tool_RSD)

        wloss_tool_stg_rsd_minute_2 = wloss_tool_stg_rsd_2/ 25 / 60
        loss_tool_stg_minute_2 = loss_tool_stg_rsd_2 / 25 / 60
        ploss_tool_stg_rsd_minute_2 = ploss_tool_stg_rsd_2 / 25 / 60
        eloss_tool_stg_rsd_minute_2 = eloss_tool_stg_rsd_2 / 25 / 60

        wtmp_tool_stageRSD_loss_values_2_val.append(wloss_tool_stg_rsd_minute_2.item())
        tmp_tool_stageRSD_loss_values_2_val.append(loss_tool_stg_minute_2.item())
        ptmp_tool_stageRSD_loss_values_2_val.append(ploss_tool_stg_rsd_minute_2.item())
        etmp_tool_stageRSD_loss_values_2_val.append(eloss_tool_stg_rsd_minute_2.item())

        # stage rsd
        wloss_stg_rsd_2, loss_stg_rsd_2, ploss_stg_rsd_2, eloss_stg_rsd_2 = loss_anticipation_2(output_stg_RSD[:, :, :6], stage_RSD)

        wloss_stg_rsd_minute_2 = wloss_stg_rsd_2 / 25 / 60
        loss_stg_minute_2 = loss_stg_rsd_2 / 25 / 60
        ploss_stg_rsd_minute_2 = ploss_stg_rsd_2 / 25 / 60
        eloss_stg_rsd_minute_2 = eloss_stg_rsd_2 / 25 / 60

        wtmp_stageRSD_loss_values_2_val.append(wloss_stg_rsd_minute_2.item())
        tmp_stageRSD_loss_values_2_val.append(loss_stg_minute_2.item())
        ptmp_stageRSD_loss_values_2_val.append(ploss_stg_rsd_minute_2.item())
        etmp_stageRSD_loss_values_2_val.append(eloss_stg_rsd_minute_2.item())
        # 3min
        # tool stage rsd
        wloss_tool_stg_rsd_3, loss_tool_stg_rsd_3, ploss_tool_stg_rsd_3, eloss_tool_stg_rsd_3 = loss_anticipation_3(
            output_tool_stg_RSD[:, :, 5:10], tool_RSD)

        wloss_tool_stg_rsd_minute_3 = wloss_tool_stg_rsd_3/ 25 / 60
        loss_tool_stg_minute_3 = loss_tool_stg_rsd_3 / 25 / 60
        ploss_tool_stg_rsd_minute_3 = ploss_tool_stg_rsd_3 / 25 / 60
        eloss_tool_stg_rsd_minute_3 = eloss_tool_stg_rsd_3 / 25 / 60

        wtmp_tool_stageRSD_loss_values_3_val.append(wloss_tool_stg_rsd_minute_3.item())
        tmp_tool_stageRSD_loss_values_3_val.append(loss_tool_stg_minute_3.item())
        ptmp_tool_stageRSD_loss_values_3_val.append(ploss_tool_stg_rsd_minute_3.item())
        etmp_tool_stageRSD_loss_values_3_val.append(eloss_tool_stg_rsd_minute_3.item())

        # stage rsd
        wloss_stg_rsd_3, loss_stg_rsd_3, ploss_stg_rsd_3, eloss_stg_rsd_3 = loss_anticipation_3(output_stg_RSD[:, :, 6:12], stage_RSD)

        wloss_stg_rsd_minute_3 = wloss_stg_rsd_3 / 25 / 60
        loss_stg_minute_3 = loss_stg_rsd_3 / 25 / 60
        ploss_stg_rsd_minute_3 = ploss_stg_rsd_3 / 25 / 60
        eloss_stg_rsd_minute_3 = eloss_stg_rsd_3 / 25 / 60

        wtmp_stageRSD_loss_values_3_val.append(wloss_stg_rsd_minute_3.item())
        tmp_stageRSD_loss_values_3_val.append(loss_stg_minute_3.item())
        ptmp_stageRSD_loss_values_3_val.append(ploss_stg_rsd_minute_3.item())
        etmp_stageRSD_loss_values_3_val.append(eloss_stg_rsd_minute_3.item())

        # 5min
        # tool stage rsd 5
        wloss_tool_stg_rsd_5, loss_tool_stg_rsd_5, ploss_tool_stg_rsd_5, eloss_tool_stg_rsd_5 = loss_anticipation(
            output_tool_stg_RSD[:, :, 10:], tool_RSD)

        wloss_tool_stg_rsd_minute_5 = wloss_tool_stg_rsd_5 / 25 / 60
        loss_tool_stg_minute_5 = loss_tool_stg_rsd_5 / 25 / 60
        ploss_tool_stg_rsd_minute_5 = ploss_tool_stg_rsd_5 / 25 / 60
        eloss_tool_stg_rsd_minute_5 = eloss_tool_stg_rsd_5 / 25 / 60

        wtmp_tool_stageRSD_loss_values_5_val.append(wloss_tool_stg_rsd_minute_5.item())
        tmp_tool_stageRSD_loss_values_5_val.append(loss_tool_stg_minute_5.item())
        ptmp_tool_stageRSD_loss_values_5_val.append(ploss_tool_stg_rsd_minute_5.item())
        etmp_tool_stageRSD_loss_values_5_val.append(eloss_tool_stg_rsd_minute_5.item())
        # stage rsd 5
        wloss_stg_rsd_5, loss_stg_rsd_5, ploss_stg_rsd_5, eloss_stg_rsd_5 = loss_anticipation(output_stg_RSD[:, :, 12:], stage_RSD)

        wloss_stg_rsd_minute_5 = wloss_stg_rsd_5 / 25 / 60
        loss_stg_minute_5 = loss_stg_rsd_5 / 25 / 60
        ploss_stg_rsd_minute_5 = ploss_stg_rsd_5 / 25 / 60
        eloss_stg_rsd_minute_5 = eloss_stg_rsd_5 / 25 / 60

        wtmp_stageRSD_loss_values_5_val.append(wloss_stg_rsd_minute_5.item())
        tmp_stageRSD_loss_values_5_val.append(loss_stg_minute_5.item())
        ptmp_stageRSD_loss_values_5_val.append(ploss_stg_rsd_minute_5.item())
        etmp_stageRSD_loss_values_5_val.append(eloss_stg_rsd_minute_5.item())

        # tool_wise_2
        output_loss_list = class_wise_loss_anticipation_loss_2(output_tool_stg_RSD[:, :, :5], tool_RSD)
        tool_wise_loss_2 = self.app_item(tool_wise_loss_2, output_loss_list)
        # tool_wise_3
        output_loss_list = class_wise_loss_anticipation_loss_3(output_tool_stg_RSD[:, :, 5:10], tool_RSD)
        tool_wise_loss_3 = self.app_item(tool_wise_loss_3, output_loss_list)
        # tool_wise_5
        output_loss_list = class_wise_loss_anticipation_loss(output_tool_stg_RSD[:, :, 10:], tool_RSD)
        tool_wise_loss = self.app_item(tool_wise_loss, output_loss_list)

        # p_wise_2
        output_loss_list = class_wise_loss_anticipation_loss_2(output_stg_RSD[:, :, :6], stage_RSD)
        phase_wise_loss_2 = self.app_item(phase_wise_loss_2, output_loss_list)
        # p_wise_3
        output_loss_list = class_wise_loss_anticipation_loss_3(output_stg_RSD[:, :, 6:12], stage_RSD)
        phase_wise_loss_3 = self.app_item(phase_wise_loss_3, output_loss_list)
        # p_wise_5
        output_loss_list = class_wise_loss_anticipation_loss(output_stg_RSD[:, :, 12:], stage_RSD)
        phase_wise_loss = self.app_item(phase_wise_loss, output_loss_list)

        # stat
        logging.debug('gpu {}'.format(torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024))
        logging.info('Whole Surgery RSD {}'.format(np.mean(tmp_RSD_val)))
        logging.info('wMAE 2min tool {}'.format(np.mean(wtmp_tool_stageRSD_loss_values_2_val)))
        logging.info('inMAE 2min tool {}'.format(np.mean(tmp_tool_stageRSD_loss_values_2_val)))
        logging.info('pMAE 2min tool {}'.format(np.mean(ptmp_tool_stageRSD_loss_values_2_val)))
        logging.info('eMAE 2min tool {}'.format(np.mean(etmp_tool_stageRSD_loss_values_2_val)))

        logging.info('wMAE 2min stage {}'.format(np.mean(wtmp_stageRSD_loss_values_2_val)))
        logging.info('inMAE 2min stage {}'.format(np.mean(tmp_stageRSD_loss_values_2_val)))
        logging.info('pMAE 2min stage {}'.format(np.mean(ptmp_stageRSD_loss_values_2_val)))
        logging.info('eMAE 2min stage {}'.format(np.mean(etmp_stageRSD_loss_values_2_val)))

        logging.info('wMAE 3min tool {}'.format(np.mean(wtmp_tool_stageRSD_loss_values_3_val)))
        logging.info('inMAE 3min tool {}'.format(np.mean(tmp_tool_stageRSD_loss_values_3_val)))
        logging.info('pMAE 3min tool {}'.format(np.mean(ptmp_tool_stageRSD_loss_values_3_val)))
        logging.info('eMAE 3min tool {}'.format(np.mean(etmp_tool_stageRSD_loss_values_3_val)))

        logging.info('wMAE 3min stage {}'.format(np.mean(wtmp_stageRSD_loss_values_3_val)))
        logging.info('inMAE 3min stage {}'.format(np.mean(tmp_stageRSD_loss_values_3_val)))
        logging.info('pMAE 3min stage {}'.format(np.mean(ptmp_stageRSD_loss_values_3_val)))
        logging.info('eMAE 3min stage {}'.format(np.mean(etmp_stageRSD_loss_values_3_val)))

        logging.info('wMAE 5min tool {}'.format(np.mean(wtmp_tool_stageRSD_loss_values_5_val)))
        logging.info('inMAE 5min tool {}'.format(np.mean(tmp_tool_stageRSD_loss_values_5_val)))
        logging.info('pMAE 5min tool {}'.format(np.mean(ptmp_tool_stageRSD_loss_values_5_val)))
        logging.info('eMAE 5min tool {}'.format(np.mean(etmp_tool_stageRSD_loss_values_5_val)))

        logging.info('wMAE 5min stage {}'.format(np.mean(wtmp_stageRSD_loss_values_5_val)))
        logging.info('inMAE 5min stage {}'.format(np.mean(tmp_stageRSD_loss_values_5_val)))
        logging.info('pMAE 5min stage {}'.format(np.mean(ptmp_stageRSD_loss_values_5_val)))
        logging.info('eMAE 5min stage {}'.format(np.mean(etmp_stageRSD_loss_values_5_val)))

        logging.info('tool 2min')
        self.stat_info(tool_wise_loss_2)
        logging.info('tool 3min')
        self.stat_info(tool_wise_loss_3)
        logging.info('tool 5min')
        self.stat_info(tool_wise_loss)

        logging.info('phase 2min')
        self.stat_info(phase_wise_loss_2)
        logging.info('phase 3min')
        self.stat_info(phase_wise_loss_3)
        logging.info('phase 5min')
        self.stat_info(phase_wise_loss)


    def app_item(self,save_list, output_loss_list):
        print(save_list)
        print(output_loss_list)

        for i in range(len(save_list)):
            print(i)
            for j in range(len(save_list[i])):
                print(j)
                save_list[i][j].append(output_loss_list[i][j].item()/60/25)

        return save_list

    def infer_one_clip(self,model, data, bb):
        clip_length = bb.size(1)
        f = 0
        for t in range(clip_length):
            with torch.no_grad():
                output_feature, tmp_output_stg_RSD, tmp_output_tool_stg_RSD, tmp_RSD = model(data, bb[:, :t + 1, :, :])
                tmp_RSD = tmp_RSD
            f += 1
            if t == 0:
                output_stg_RSD = tmp_output_stg_RSD[:, t:t + 1, :]
                output_tool_stg_RSD = tmp_output_tool_stg_RSD[:, t:t + 1, :]
                output_RSD = tmp_RSD[:, t:t + 1]


            else:
                output_stg_RSD = torch.cat((output_stg_RSD, tmp_output_stg_RSD[:, t:t + 1, :]), dim=1)
                output_tool_stg_RSD = torch.cat((output_tool_stg_RSD, tmp_output_tool_stg_RSD[:, t:t + 1, :]), dim=1)
                output_RSD = torch.cat((output_RSD, tmp_RSD[:, t:t + 1]), dim=1)

        return output_feature, output_stg_RSD, output_tool_stg_RSD, output_RSD, f



    # stat for class-wise

    def stat_info(self, class_loss_lists):

        for i in range(len(class_loss_lists[0])):
            logging.info('class {} wMAE {}'.format(i,np.mean(class_loss_lists[0][i])))
            logging.info('class {} inMAE {}'.format(i,np.mean(class_loss_lists[1][i])))
            logging.info('class {} pMAE {}'.format(i,np.mean(class_loss_lists[2][i])))
            logging.info('class {} eMAE {}'.format(i,np.mean(class_loss_lists[3][i])))







