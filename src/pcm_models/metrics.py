import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

def auc_score_taskwise(predictions, labels, target_ids):
    AUCs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows]

        if torch.unique(y).shape[0] == 2:
            auc = roc_auc_score(y, preds)
            AUCs.append(auc)
            target_id_list.append(target_idx.item())
        else:
            AUCs.append(np.nan)
            target_id_list.append(target_idx.item())
    return np.nanmean(AUCs), AUCs, target_id_list


def concordance_index_taskwise(predictions, labels, target_ids):
    CIs = list()
    target_id_list = list()

    for target_idx in torch.unique(target_ids):
        rows = torch.where(target_ids == target_idx)
        preds = predictions[rows].detach()
        y = labels[rows]

        if torch.unique(y).shape[0] >= 2:
            #print(f'y: {y}')
            #print(f'preds: {preds}')
            ci = concordance_index(y, preds)
            #print(f'ci: {ci}')
            CIs.append(ci)
            target_id_list.append(target_idx.item())
        else:
            CIs.append(np.nan)
            target_id_list.append(target_idx.item())
    return np.nanmean(CIs), CIs, target_id_list
