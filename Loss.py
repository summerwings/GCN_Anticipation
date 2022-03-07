import torch
import torch.nn as nn

class anticipation_mae(nn.Module):

    def __init__(self, h=7500):
        super(anticipation_mae, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h  = torch.tensor(h).float().to(self.device)
        self.lossfunc = nn.L1Loss()

    def forward(self, output_rsd, target_rsd):

        wMAE = torch.tensor(0).float().to(self.device)
        inMAE = torch.tensor(0).float().to(self.device)
        pMAE = torch.tensor(0).float().to(self.device)
        eMAE = torch.tensor(0).float().to(self.device)

        nwMAE = 0
        ninMAE = 0
        npMAE = 0
        neMAE = 0

        for n in range(output_rsd.size(0)):
            for c in range(output_rsd.size(-1)):
                tmp_output = output_rsd[n,:,c]
                tmp_target = target_rsd[n,:,c]

                tmp_output = torch.where((tmp_output > self.h)|(tmp_output <0), self.h, tmp_output)
                tmp_target = torch.where((tmp_target > self.h)|(tmp_target <0), self.h, tmp_target)


                cond_oMAE = torch.where((tmp_target == self.h))
                cond_inMAE = torch.where((tmp_target<self.h) & (tmp_target>0))
                cond_pMAE = torch.where((tmp_output<0.9*self.h) & (tmp_output>0.1*self.h))
                cond_eMAE = torch.where((tmp_target < 0.1 * self.h) & (tmp_target > 0))


                output_oMAE = torch.abs(tmp_output[cond_oMAE] - tmp_target[cond_oMAE])
                output_inMAE = torch.abs(tmp_output[cond_inMAE] - tmp_target[cond_inMAE])

                output_wMAE = torch.nanmean(torch.stack((torch.nanmean(output_oMAE),torch.nanmean(output_inMAE))))
                output_pMAE  = torch.abs(tmp_output[cond_pMAE] - tmp_target[cond_pMAE])
                output_eMAE = torch.abs(tmp_output[cond_eMAE] - tmp_target[cond_eMAE])

                if torch.isnan(output_wMAE).item() == 0:
                    nwMAE += 1
                    wMAE += output_wMAE


                if output_inMAE.size(0) > 0:
                    ninMAE += 1
                    inMAE += torch.nanmean(output_inMAE)

                if output_pMAE.size(0) > 0:
                    npMAE += 1
                    pMAE += torch.nanmean(output_pMAE)


                if output_eMAE.size(0)>0:
                    neMAE += 1
                    eMAE += torch.nanmean(output_eMAE)

        # ensure the zero size would be nan
        wMAE = wMAE/nwMAE
        inMAE = inMAE/ninMAE
        pMAE = pMAE/npMAE
        eMAE = eMAE/neMAE


        return wMAE, inMAE, pMAE, eMAE

class attention_loss(nn.Module):
    def __init__(self, sigma = 0.05):
        super(attention_loss, self).__init__()


        self.sigma = sigma



    def forward(self, a ):

        loss  = self.sigma*torch.sum(a)

        return loss


class framewise_ce(nn.Module):
    def __init__(self, sigma = 0.05):
        super(framewise_ce, self).__init__()


        self.ce = nn.CrossEntropyLoss(size_average=True)



    def forward(self, x, target):

        loss = 0
        for b in range(x.size(0)):

            loss+= self.ce(x[b,:,:],target[b,:])


        return loss


class similarity_loss(nn.Module):
    def __init__(self, sigma = 0.05):
        super(similarity_loss, self).__init__()


        self.similarity_loss = nn.CosineSimilarity(dim=2)



    def forward(self, output,target ):

        loss  = torch.mean(self.similarity_loss(output,target))

        return loss

class class_wise_anticipation_mae(nn.Module):

    def __init__(self, h=7500):
        super(class_wise_anticipation_mae, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h = torch.tensor(h).float().to(self.device)
        self.lossfunc = anticipation_mae(h)
        self.base = [0]

    def forward(self, output_rsd, target_rsd):

        c = output_rsd.size(-1)
        wMAE_loss_list  = self.base*c
        inMAE_loss_list = self.base*c
        pMAE_loss_list = self.base*c
        eMAE_loss_list = self.base*c

        for i in range(c):

            tmp_output_rsd = output_rsd[:,:,i:i+1]
            tmp_target_rsd = target_rsd[:,:,i:i+1]
            wMAE_loss_list[i] , inMAE_loss_list[i],pMAE_loss_list[i], eMAE_loss_list[i] = self.lossfunc(tmp_output_rsd,tmp_target_rsd)

        output_loss_list = [wMAE_loss_list, inMAE_loss_list,pMAE_loss_list,eMAE_loss_list]
        return output_loss_list

