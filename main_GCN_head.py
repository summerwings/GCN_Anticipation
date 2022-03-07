import random
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from C80_video_dataloader import  SurgeryVideoDataset
from GCN_RSDNet import GCNProgressNet
from Loss import anticipation_mae
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class train_GCN(nn.Module):

    def __init__(self, random_seed,
                 train_dir=r'Sample\Train',
                 val_dir = r'Sample\Val',
                 ):

        super(train_GCN, self).__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.seed = random_seed

    def train(self, epochs = 40, val_step =1):
        # set seed
        self.setup_seed(self.seed)

        # load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))
        model = GCNProgressNet(stream='Graph').to(device)

        # train setting
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
        loss_anticipation_train = anticipation_mae().to(device)
        loss_anticipation = anticipation_mae().to(device)
        loss_anticipation_2 = anticipation_mae(h=3000).to(device)
        loss_anticipation_3 = anticipation_mae(h=4500).to(device)

        # set epoch
        epochs = epochs
        val_step = val_step

        # set metrics
        stage_RSD_loss_values_train = []
        tool_RSD_loss_values_train = []

        stage_RSD_loss_values_val = []
        tool_RSD_loss_values_val = []

        total_loss_train = []
        total_loss_val = []

        min_RSD_MAE = 100000

        # train
        for i in range(epochs):

            tmp_stageRSD_loss_values_train = []
            tmp_tool_stageRSD_loss_values_train = []

            tmp_total_loss_train = []

            num = 0

            train_data_loader = DataLoader(SurgeryVideoDataset(self.train_dir), batch_size=32, shuffle=True)

            for b, b_data in enumerate(train_data_loader):
                logging.info('epoch {} batch {}'.format(i, b))
                data = b_data['video'].to(device)
                bb = b_data['boxes'].to(device)
                stage_RSD = b_data['stage_RSD'][:, :, 1:].to(device)
                tool_RSD = b_data['tool_RSD'].to(device)

                model.train()
                optimizer.zero_grad()
                # feed data
                output_feature, output_stg_RSD, output_tool_stg_RSD = model(data, bb)

                # tool stage rsd
                loss_tool_stg_rsd_2, inloss_tool_stg_rsd_2, _, _ = loss_anticipation_2(output_tool_stg_RSD[:, :, :5],
                                                                                       tool_RSD)
                loss_tool_stg_rsd_3, inloss_tool_stg_rsd_3, _, _ = loss_anticipation_3(output_tool_stg_RSD[:, :, 5:10],
                                                                                       tool_RSD)
                loss_tool_stg_rsd, inloss_tool_stg_rsd, _, _ = loss_anticipation_train(output_tool_stg_RSD[:, :, 10:],
                                                                                       tool_RSD)

                loss_tool_stg_all = inloss_tool_stg_rsd + inloss_tool_stg_rsd_2 + inloss_tool_stg_rsd_3
                loss_tool_stg_minute = loss_tool_stg_all / 25 / 60
                tmp_tool_stageRSD_loss_values_train.append(loss_tool_stg_minute.item())
                # stage rsd
                loss_stg_rsd_2, inloss_stg_rsd_2, _, _ = loss_anticipation_2(output_stg_RSD[:, :, :6], stage_RSD)
                loss_stg_rsd_3, inloss_stg_rsd_3, _, _ = loss_anticipation_3(output_stg_RSD[:, :, 6:12], stage_RSD)
                loss_stg_rsd, inloss_stg_rsd, _, _ = loss_anticipation_train(output_stg_RSD[:, :, 12:], stage_RSD)
                loss_stg_rsd_all = inloss_stg_rsd+inloss_stg_rsd_2+inloss_stg_rsd_3
                loss_stg_minute = loss_stg_rsd_all / 25 / 60
                tmp_stageRSD_loss_values_train.append(loss_stg_minute.item())

                # total loss
                loss_total = loss_tool_stg_minute + loss_stg_minute  # + loss_stg #+ loss_sim# +loss_minute + loss_stg + loss_Elapsed_minute #att_loss+ loss_stg_minute + att_loss
                tmp_total_loss_train.append(loss_total.item())

                # optimize
                loss_total.backward()
                optimizer.step()
                num += 1

            logging.info('EPOCH:{},NUM:{},loss={}'.format(i, num, loss_total))

            # stat
            stage_RSD_loss_values_train.append(np.nanmean(tmp_stageRSD_loss_values_train))
            tool_RSD_loss_values_train.append(np.nanmean(tmp_tool_stageRSD_loss_values_train))
            total_loss_train.append(np.nanmean(tmp_total_loss_train))

            if (i + 1) % val_step == 0:

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

                val_data_loader = DataLoader(SurgeryVideoDataset(self.val_dir), shuffle=False)

                for v, v_data in enumerate(val_data_loader):
                    logging.info('epoch {} batch {}'.format(i, v))
                    model.eval()
                    v_data = v_data

                    # define and squeeze the dim from 1 b n n n to b n n n n
                    data = v_data['video'].to(device)
                    bb = v_data['boxes'].to(device)
                    stage_RSD = v_data['stage_RSD'][:, :, 1:].to(device)
                    tool_RSD = v_data['tool_RSD'].to(device)

                    with torch.no_grad():
                        output_feature, output_stg_RSD, output_tool_stg_RSD = model(data, bb)

                    # tool stage rsd
                    loss_tool_stg_rsd_2, inloss_tool_stg_rsd_2, _, _ = loss_anticipation_2(
                        output_tool_stg_RSD[:, :, :5],
                        tool_RSD)
                    loss_tool_stg_rsd_3, inloss_tool_stg_rsd_3, _, _ = loss_anticipation_3(
                        output_tool_stg_RSD[:, :, 5:10],
                        tool_RSD)
                    loss_tool_stg_rsd, inloss_tool_stg_rsd, _, _ = loss_anticipation_train(
                        output_tool_stg_RSD[:, :, 10:],
                        tool_RSD)

                    loss_tool_stg_all = inloss_tool_stg_rsd + inloss_tool_stg_rsd_2 + inloss_tool_stg_rsd_3
                    loss_tool_stg_minute = loss_tool_stg_all / 25 / 60
                    tmp_tool_stageRSD_loss_values_val.append(loss_tool_stg_minute.item())
                    # stage rsd
                    loss_stg_rsd_2, inloss_stg_rsd_2, _, _ = loss_anticipation_2(output_stg_RSD[:, :, :6], stage_RSD)
                    loss_stg_rsd_3, inloss_stg_rsd_3, _, _ = loss_anticipation_3(output_stg_RSD[:, :, 6:12], stage_RSD)
                    loss_stg_rsd, inloss_stg_rsd, _, _ = loss_anticipation_train(output_stg_RSD[:, :, 12:], stage_RSD)
                    loss_stg_rsd_all = inloss_stg_rsd+inloss_stg_rsd_2+inloss_stg_rsd_3
                    loss_stg_minute = loss_stg_rsd_all / 25 / 60
                    tmp_stageRSD_loss_values_val.append(loss_stg_minute.item())

                    # total loss
                    loss_total = loss_tool_stg_minute + loss_stg_minute  # + loss_stg #+ loss_sim #+ loss_stg + loss_Elapsed_minute #att_loss+ loss_stg_minute + att_loss
                    tmp_total_loss_val.append(loss_total.item())

                    # 2min
                    # tool stage rsd
                    wloss_tool_stg_rsd_2, loss_tool_stg_rsd_2, ploss_tool_stg_rsd_2, eloss_tool_stg_rsd_2 = loss_anticipation_2(
                        output_tool_stg_RSD[:, :, :5], tool_RSD)

                    wloss_tool_stg_rsd_2 = wloss_tool_stg_rsd_2 / 25 / 60
                    loss_tool_stg_minute_2 = loss_tool_stg_rsd_2 / 25 / 60
                    ploss_tool_stg_rsd_minute_2 = ploss_tool_stg_rsd_2 / 25 / 60
                    eloss_tool_stg_rsd_minute_2 = eloss_tool_stg_rsd_2 / 25 / 60

                    wtmp_tool_stageRSD_loss_values_2_val.append(wloss_tool_stg_rsd_2.item())
                    tmp_tool_stageRSD_loss_values_2_val.append(loss_tool_stg_minute_2.item())
                    ptmp_tool_stageRSD_loss_values_2_val.append(ploss_tool_stg_rsd_minute_2.item())
                    etmp_tool_stageRSD_loss_values_2_val.append(eloss_tool_stg_rsd_minute_2.item())
                    # stage rsd
                    wloss_stg_rsd_2, loss_stg_rsd_2, ploss_stg_rsd_2, eloss_stg_rsd_2 = loss_anticipation_2(
                        output_stg_RSD[:, :, :6], stage_RSD)

                    wloss_stg_rsd_2 = wloss_stg_rsd_2 / 25 / 60
                    loss_stg_minute_2 = loss_stg_rsd_2 / 25 / 60
                    ploss_stg_rsd_minute_2 = ploss_stg_rsd_2 / 25 / 60
                    eloss_stg_rsd_minute_2 = eloss_stg_rsd_2 / 25 / 60

                    wtmp_stageRSD_loss_values_2_val.append(wloss_stg_rsd_2.item())
                    tmp_stageRSD_loss_values_2_val.append(loss_stg_minute_2.item())
                    ptmp_stageRSD_loss_values_2_val.append(ploss_stg_rsd_minute_2.item())
                    etmp_stageRSD_loss_values_2_val.append(eloss_stg_rsd_minute_2.item())
                    # 3min
                    # tool stage rsd
                    wloss_tool_stg_rsd_3, loss_tool_stg_rsd_3, ploss_tool_stg_rsd_3, eloss_tool_stg_rsd_3 = loss_anticipation_3(
                        output_tool_stg_RSD[:, :, 5:10], tool_RSD)

                    wloss_tool_stg_rsd_3 = wloss_tool_stg_rsd_3 / 25 / 60
                    loss_tool_stg_minute_3 = loss_tool_stg_rsd_3 / 25 / 60
                    ploss_tool_stg_rsd_minute_3 = ploss_tool_stg_rsd_3 / 25 / 60
                    eloss_tool_stg_rsd_minute_3 = eloss_tool_stg_rsd_3 / 25 / 60

                    wtmp_tool_stageRSD_loss_values_3_val.append(wloss_tool_stg_rsd_3.item())
                    tmp_tool_stageRSD_loss_values_3_val.append(loss_tool_stg_minute_3.item())
                    ptmp_tool_stageRSD_loss_values_3_val.append(ploss_tool_stg_rsd_minute_3.item())
                    etmp_tool_stageRSD_loss_values_3_val.append(eloss_tool_stg_rsd_minute_3.item())
                    # stage rsd
                    wloss_stg_rsd_3, loss_stg_rsd_3, ploss_stg_rsd_3, eloss_stg_rsd_3 = loss_anticipation_2(
                        output_stg_RSD[:, :, 6:12], stage_RSD)

                    wloss_stg_rsd_3 = wloss_stg_rsd_3 / 25 / 60
                    loss_stg_minute_3 = loss_stg_rsd_3 / 25 / 60
                    ploss_stg_rsd_minute_3 = ploss_stg_rsd_3 / 25 / 60
                    eloss_stg_rsd_minute_3 = eloss_stg_rsd_3 / 25 / 60

                    wtmp_stageRSD_loss_values_3_val.append(wloss_stg_rsd_3.item())
                    tmp_stageRSD_loss_values_3_val.append(loss_stg_minute_3.item())
                    ptmp_stageRSD_loss_values_3_val.append(ploss_stg_rsd_minute_3.item())
                    etmp_stageRSD_loss_values_3_val.append(eloss_stg_rsd_minute_3.item())

                    # 5min
                    # tool stage rsd 5
                    wloss_tool_stg_rsd_5, loss_tool_stg_rsd_5, ploss_tool_stg_rsd_5, eloss_tool_stg_rsd_5 = loss_anticipation(
                        output_tool_stg_RSD[:, :, 10:], tool_RSD)

                    wloss_tool_stg_rsd_5 = wloss_tool_stg_rsd_5 / 25 / 60
                    loss_tool_stg_minute_5 = loss_tool_stg_rsd_5 / 25 / 60
                    ploss_tool_stg_rsd_minute_5 = ploss_tool_stg_rsd_5 / 25 / 60
                    eloss_tool_stg_rsd_minute_5 = eloss_tool_stg_rsd_5 / 25 / 60

                    wtmp_tool_stageRSD_loss_values_5_val.append(wloss_tool_stg_rsd_5.item())
                    tmp_tool_stageRSD_loss_values_5_val.append(loss_tool_stg_minute_5.item())
                    ptmp_tool_stageRSD_loss_values_5_val.append(ploss_tool_stg_rsd_minute_5.item())
                    etmp_tool_stageRSD_loss_values_5_val.append(eloss_tool_stg_rsd_minute_5.item())
                    # stage rsd 5
                    wloss_stg_rsd_5, loss_stg_rsd_5, ploss_stg_rsd_5, eloss_stg_rsd_5 = loss_anticipation(
                        output_stg_RSD[:, :, 12:], stage_RSD)

                    wloss_stg_rsd_5 = wloss_stg_rsd_5 / 25 / 60
                    loss_stg_minute_5 = loss_stg_rsd_5 / 25 / 60
                    ploss_stg_rsd_minute_5 = ploss_stg_rsd_5 / 25 / 60
                    eloss_stg_rsd_minute_5 = eloss_stg_rsd_5 / 25 / 60

                    wtmp_stageRSD_loss_values_5_val.append(wloss_stg_rsd_5.item())
                    tmp_stageRSD_loss_values_5_val.append(loss_stg_minute_5.item())
                    ptmp_stageRSD_loss_values_5_val.append(ploss_stg_rsd_minute_5.item())
                    etmp_stageRSD_loss_values_5_val.append(eloss_stg_rsd_minute_5.item())

                    logging.debug('gpu {}'.format(torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024))

                if np.nanmean(tmp_total_loss_val) < min_RSD_MAE:
                    logging.info('save model for epoch {}'.format(i))
                    torch.save(model.state_dict(), 'best_PNet_epoch.pth')
                    min_RSD_MAE = np.nanmean(tmp_total_loss_val)

                    '''
                    tmp_mean_RSD_loss = np.mean(tmp_RSD_loss_values_val)
                    tmp_mean_stage_RSD_loss = np.mean(tmp_stageRSD_loss_values_val)
                    tmp_mean_ce_loss = np.mean(tmp_ce_loss_values_val)
                    tmp_mean_total_loss = np.mean(tmp_total_loss_val)
                    '''

                # stat
                stage_RSD_loss_values_val.append(np.nanmean(tmp_stageRSD_loss_values_val))
                tool_RSD_loss_values_val.append(np.nanmean(tmp_tool_stageRSD_loss_values_val))
                total_loss_val.append(np.nanmean(tmp_total_loss_val))
                logging.info('wMAE 2min tool {}'.format(np.nanmean(wtmp_tool_stageRSD_loss_values_2_val)))
                logging.info('inMAE 2min tool {}'.format(np.nanmean(tmp_tool_stageRSD_loss_values_2_val)))
                logging.info('pMAE 2min tool {}'.format(np.nanmean(ptmp_tool_stageRSD_loss_values_2_val)))
                logging.info('eMAE 2min tool {}'.format(np.nanmean(etmp_tool_stageRSD_loss_values_2_val)))

                logging.info('wMAE 2min stage {}'.format(np.nanmean(wtmp_stageRSD_loss_values_2_val)))
                logging.info('inMAE 2min stage {}'.format(np.nanmean(tmp_stageRSD_loss_values_2_val)))
                logging.info('pMAE 2min stage {}'.format(np.nanmean(ptmp_stageRSD_loss_values_2_val)))
                logging.info('eMAE 2min stage {}'.format(np.nanmean(etmp_stageRSD_loss_values_2_val)))

                logging.info('wMAE 3min tool {}'.format(np.nanmean(wtmp_tool_stageRSD_loss_values_3_val)))
                logging.info('inMAE 3min tool {}'.format(np.nanmean(tmp_tool_stageRSD_loss_values_3_val)))
                logging.info('pMAE 3min tool {}'.format(np.nanmean(ptmp_tool_stageRSD_loss_values_3_val)))
                logging.info('eMAE 3min tool {}'.format(np.nanmean(etmp_tool_stageRSD_loss_values_3_val)))

                logging.info('wMAE 3min stage {}'.format(np.nanmean(wtmp_stageRSD_loss_values_3_val)))
                logging.info('inMAE 3min stage {}'.format(np.nanmean(tmp_stageRSD_loss_values_3_val)))
                logging.info('pMAE 3min stage {}'.format(np.nanmean(ptmp_stageRSD_loss_values_3_val)))
                logging.info('eMAE 3min stage {}'.format(np.nanmean(etmp_stageRSD_loss_values_3_val)))

                logging.info('wMAE 5min tool {}'.format(np.nanmean(wtmp_tool_stageRSD_loss_values_5_val)))
                logging.info('inMAE 5min tool {}'.format(np.nanmean(tmp_tool_stageRSD_loss_values_5_val)))
                logging.info('pMAE 5min tool {}'.format(np.nanmean(ptmp_tool_stageRSD_loss_values_5_val)))
                logging.info('eMAE 5min tool {}'.format(np.nanmean(etmp_tool_stageRSD_loss_values_5_val)))

                logging.info('wMAE 5min stage {}'.format(np.nanmean(wtmp_stageRSD_loss_values_5_val)))
                logging.info('inMAE 5min stage {}'.format(np.nanmean(tmp_stageRSD_loss_values_5_val)))
                logging.info('pMAE 5min stage {}'.format(np.nanmean(ptmp_stageRSD_loss_values_5_val)))
                logging.info('eMAE 5min stage {}'.format(np.nanmean(etmp_stageRSD_loss_values_5_val)))

            plt.figure("train", (32, 12))

            plt.subplot(2, 3, 1)
            plt.title("Train Stage RSD MAE Loss")
            x = [(i + 1) for i in range(len(stage_RSD_loss_values_train))]
            plt.xlabel("epoch")
            plt.plot(x, stage_RSD_loss_values_train)

            plt.subplot(2, 3, 2)
            plt.title("Train Tool RSD Time MAE Loss")
            x = [(i + 1) for i in range(len(tool_RSD_loss_values_train))]
            plt.xlabel("epoch")
            plt.plot(x, tool_RSD_loss_values_train)

            plt.subplot(2, 3, 3)
            plt.title("Train Total Loss")
            x = [(i + 1) for i in range(len(total_loss_train))]
            plt.xlabel("epoch")
            plt.plot(x, total_loss_train)

            plt.subplot(2, 3, 4)
            plt.title("Val Stage RSD MAE Loss")
            x = [(i + 1) for i in range(len(stage_RSD_loss_values_val))]
            plt.xlabel("epoch")
            plt.plot(x, stage_RSD_loss_values_val)

            plt.subplot(2, 3, 5)
            plt.title("Val Tool RSD Time MAE Loss")
            x = [(i + 1) for i in range(len(tool_RSD_loss_values_val))]
            plt.xlabel("epoch")
            plt.plot(x, tool_RSD_loss_values_val)

            plt.subplot(2, 3, 6)
            plt.title("Val Total Loss")
            plt.xlabel("epoch")
            x = [(i + 1) * 1 for i in range(len(total_loss_val))]
            plt.plot(x, total_loss_val)

            save_dir = r'PNet_loss.png'
            plt.savefig(save_dir)
            plt.close("train")



    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True






