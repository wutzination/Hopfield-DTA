import numpy as np
import torch
import torch.nn as nn
from src.pcm_models.losses import MSE

if __name__ == "__main__":
    criterion = torch.nn.MSELoss()

    label_list = [9, 10, 11, 12, 13, 14]
    pred_list = [9.3, 9.6, 11.9, 12.5, 12.8, 14.5]
    label = torch.tensor(label_list, dtype=torch.float)
    pred = torch.tensor(pred_list, dtype=torch.float)

    print(f"Test loss; should be 0:\t{criterion(label, label)}")
    print(f"Loss not standardized:\t{criterion(pred, label)}")

    print(f"Labels mean {torch.mean(label)} and  std {torch.std(label)}")

    mean = np.mean(label_list)
    std = np.std(label_list)

    label_scaled = (label - mean) / std
    pred_scaled = (pred - mean) / std
    print(f"Scaled Labels: {label_scaled}")
    print(f"Scaled Pred: {pred_scaled}")
    print(f"Labels new mean {torch.mean(label_scaled)} and  std {torch.std(label_scaled)}")
    print(f"Loss new scaled:\t{criterion(pred_scaled, label_scaled)}")

    label_un_scaled = label_scaled * std + mean
    pred_un_scaled = pred_scaled * std + mean
    print(f"Un-Scaled Labels: {label_un_scaled}")
    print(f"Un-Scaled Pred: {pred_un_scaled}")
    print(f"Loss un-scaled:\t{criterion(pred_un_scaled, label_un_scaled)}")

    pred_de_std = torch.mul(pred_scaled, std)
    pred_de_std = torch.add(pred_de_std, mean)

    label_de_std = torch.mul(label_scaled, std)
    label_de_std = torch.add(label_de_std, mean)

    print(f"Un-Scaled Labels: {label_de_std}")
    print(f"Un-Scaled Pred: {pred_de_std}")

    print("\n\n")

    unscaled_pred = torch.tensor([11.7229, 11.7285, 11.7348, 11.7278, 11.7323, 11.7260, 11.7244, 11.7311,
                                 11.7313, 11.7311, 11.7302, 11.7274, 11.7286, 11.7285, 11.7302, 11.7261,
                                 11.7273, 11.7287, 11.7267, 11.7270, 11.7320, 11.7250, 11.7271, 11.7300,
                                 11.7256, 11.7311, 11.7313, 11.7245, 11.7288, 11.7317, 11.7245, 11.7304,
                                 11.7257, 11.7302, 11.7272, 11.7268, 11.7349, 11.7305, 11.7267, 11.7269], dtype=torch.float)

    unscaled_label = torch.tensor([11.4979, 11.8000, 14.7003, 12.1000, 12.2000, 11.8000, 11.9000, 12.7000,
                                  12.5000, 11.3000, 11.8000, 11.8000, 11.9000, 11.2000, 13.1223, 11.2000,
                                  11.3000, 12.3000, 11.6000, 11.7000, 11.2000, 13.7000, 11.5000, 11.1000,
                                  11.2000, 11.7000, 11.7000, 11.2000, 11.7000, 11.7000, 12.5000, 11.1000,
                                  11.8000, 14.0000, 10.6229, 11.2000, 10.7021, 12.0000, 12.0000, 10.6229], dtype=torch.float)

    scaled_pred = torch.tensor([0.0034, 0.0100, 0.0175, 0.0092, 0.0146, 0.0070, 0.0051, 0.0131,
                                0.0134, 0.0131, 0.0121, 0.0087, 0.0101,0.0101, 0.0121, 0.0072,
                                0.0086, 0.0102, 0.0079, 0.0082, 0.0142, 0.0059, 0.0083, 0.0118,
                                0.0066, 0.0132, 0.0134, 0.0052, 0.0104, 0.0139, 0.0053, 0.0123,
                                0.0067, 0.0121, 0.0085, 0.0080, 0.0177, 0.0124, 0.0079, 0.0081], dtype=torch.float)

    scaled_label = torch.tensor([-0.2646,  0.0952,  3.5503,  0.4526,  0.5717,  0.0952,  0.2143,  1.1674,
                                 0.9291, -0.5004,  0.0952,  0.0952,  0.2143, -0.6196,  1.6704, -0.6196,
                                -0.5004,  0.6909, -0.1430, -0.0239, -0.6196,  2.3586, -0.2622, -0.7387,
                                -0.6196, -0.0239, -0.0239, -0.6196, -0.0239, -0.0239,  0.9291, -0.7387,
                                 0.0952,  2.7160, -1.3071, -0.6196, -1.2128,  0.3335,  0.3335, -1.3071], dtype=torch.float)
    print(f"Computed from Values")
    print(f"Loss un-scaled:\t{criterion(unscaled_pred, unscaled_label)}")
    print(f"Loss scaled:\t{criterion(scaled_pred, scaled_label)}")

    pred_de_std2 = torch.mul(scaled_pred, 0.8408341695391968)
    pred_de_std2 = torch.add(pred_de_std2, 11.720703662534172)

    label_de_std2 = torch.mul(scaled_label, 0.8408341695391968)
    label_de_std2 = torch.add(label_de_std2, 11.720703662534172)
    print(f"Unscale label and pred")
    print(f"Loss un-scaled2:\t{criterion(pred_de_std2, label_de_std2)}\n")

    print(f'Unscaled Preds:\n{pred_de_std2}')
    print(f'Unscaled Labels:\n{label_de_std2}')
    print("\n\n")
    scale_pred_man = torch.sub(unscaled_pred, 11.720703662534172)
    scale_pred_man = torch.div(scale_pred_man, 0.8408341695391968)

    scale_label_man = torch.sub(unscaled_label, 11.720703662534172)
    scale_label_man = torch.div(scale_label_man, 0.8408341695391968)

    print(f'Manually scaled Preds:\n{scale_pred_man}')
    print(f'Manually scaled Labels:\n{scale_label_man}')
    print(f"Loss manually scaled:\t{criterion(scale_pred_man, scale_label_man)}\n")
