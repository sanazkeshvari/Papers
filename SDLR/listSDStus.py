import numpy as np
import torch

###
import pandas as pd
import os
###

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS


def listSDStus(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE, xb=None, epoch = 0, Parameters_Path = None):
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max # log()

    # Prior Calculation
    ###AA = preds_sorted_by_true_minus_max.exp().flip(dims=[1])
    ###BB = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1)
    Unique_Labels = torch.sort(torch.unique(y_true)).values[1:]
    Sigma_Label_Based = pd.read_csv("./Parameters/One/Sigma_All_Score_" + str(Parameters_Path) + ".csv", index_col = 0)
    Mean_Label_Based = pd.read_csv("./Parameters/One/Mean_All_Score_" + str(Parameters_Path) + ".csv", index_col = 0)
    Prior_Loss = 0
    XX = torch.clone(y_pred)
    PX_XX = torch.clone(y_pred)
    for i in range(len(Unique_Labels)):
        Selected_Indices = torch.where(y_true == Unique_Labels[i])
        Temp = y_pred[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy())]
        Temp = torch.abs(Temp)
        Temp_Mean = torch.from_numpy(Mean_Label_Based.iloc[int(Unique_Labels[i]), :].to_numpy())
        Temp_Sigma = torch.from_numpy(Sigma_Label_Based.iloc[int(Unique_Labels[i]), :].to_numpy())

        Coefficient = 1 # torch.pow((1 / ( 2 * torch.pi * Temp_Sigma)), (len(Selected_Indices) / 2) )
        Temp[torch.where(Temp == 0)] = float("-inf")
        Temp = torch.multiply(Coefficient, torch.exp(-1 * (torch.pow(torch.subtract(Temp, Temp_Mean), 2) / (2 * Temp_Sigma))))

        ###
        PX_XX[list(Selected_Indices[0].numpy().astype(np.int8)), list(Selected_Indices[1].numpy().astype(np.int8))] = torch.multiply(Temp, PX_XX[list(Selected_Indices[0].numpy()), list(Selected_Indices[1].numpy())]).type(torch.float32)
        ###

        Temp = torch.log(Temp + eps)
        Prior_Loss += torch.sum(Temp)

    Prior_Loss /= xb.shape[0] # Mean Of Priors

    """
    xb = torch.abs(xb)
    sigma = ((torch.max(xb, dim=1).values - torch.min(xb, dim=1).values) / 6.0) + 1e-10
    #prior6 = torch.multiply(1 / (sigma + eps), torch.exp(((xb - torch.min(xb)) / (torch.max(xb) - torch.min(xb))) / (2 * sigma) ))
    prior6 = torch.multiply(1 / (sigma + eps), torch.exp( -1 * (torch.pow(xb, 2) / (2 * sigma)) ) )

    temp = torch.linalg.norm(prior6, dim=-1)
    ###prior7 = torch.cumsum(temp.flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = observation_loss - torch.log(prior6)

    """

    if not os.path.isdir("./Parameters/One/"): os.makedirs("./Parameters/One/")
    torch.save(Temp_Sigma, 'Temp_Sigma.pt')
    XX = pd.DataFrame(XX.detach().numpy())
    XX.to_csv("./Parameters/One/XX_" + str(epoch).zfill(4) + ".csv")
    PX_XX = pd.DataFrame(PX_XX.detach().numpy())
    PX_XX.to_csv("./Parameters/One/PX_XX_" + str(epoch).zfill(4) + ".csv")

    Mean_Label_Based = pd.DataFrame(Mean_Label_Based, index = Unique_Labels.to(torch.int8).tolist())
    Mean_Label_Based.to_csv("./Parameters/One/Mean_" + str(epoch).zfill(4) + ".csv")
    Sigma_Label_Based = pd.DataFrame(Sigma_Label_Based, index = Unique_Labels.to(torch.int8).tolist())
    Sigma_Label_Based.to_csv("./Parameters/One/Sigma_" + str(epoch).zfill(4) + ".csv")

    observation_loss[mask] = 0.0

    observation_loss = torch.sum(observation_loss, dim=1) - Prior_Loss

    return torch.mean(observation_loss)
