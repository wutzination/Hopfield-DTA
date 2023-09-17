import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from functools import partial
import numpy as np

# from configs.config_instance import MyConfig

from losses import MSE
from metrics import concordance_index_taskwise
from initialization import init_weights

# Mappings
activation_function_mapping = {
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid()
}
dropout_mapping = {
    'relu': nn.Dropout,
    'selu': nn.AlphaDropout
}


class Fully_Connected_Wurdinger(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super(Fully_Connected_Wurdinger, self).__init__()

        # Config
        self.cfg = cfg
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        self.save_hyperparameters(config)

        # Loss functions
        self.LossFunction = MSE

        # Architecture
        # Input layer
        self.dropout = dropout_mapping[cfg.model.architecture.activation](
            cfg.model.regularization.input_dropout)
        self.fc = nn.Linear(cfg.model.architecture.input_dim,
                            cfg.model.architecture.number_hidden_neurons)
        self.act = activation_function_mapping[cfg.model.architecture.activation]

        # Hidden layer
        self.hidden_linear_layers = nn.ModuleList([])
        self.hidden_dropout_layers = nn.ModuleList([])
        self.hidden_activations = nn.ModuleList([])

        for _ in range(cfg.model.architecture.number_hidden_layers):
            self.hidden_dropout_layers.append(
                dropout_mapping[cfg.model.architecture.activation](
                    cfg.model.regularization.dropout)
            )
            self.hidden_linear_layers.append(
                nn.Linear(cfg.model.architecture.number_hidden_neurons,
                          cfg.model.architecture.number_hidden_neurons)
            )
            self.hidden_activations.append(
                activation_function_mapping[cfg.model.architecture.activation]
            )

        # Output layer
        self.dropout_o = dropout_mapping[cfg.model.architecture.activation](
            cfg.model.regularization.dropout)
        self.fc_o = nn.Linear(cfg.model.architecture.number_hidden_neurons,
                              cfg.model.architecture.output_dim)

        # Initialization
        model_initialization = partial(init_weights, cfg.model.architecture.activation)
        self.apply(model_initialization)

    def forward(self, input_descriptor_mol, input_descriptor_prot):

        input_descriptor = torch.cat((input_descriptor_mol, input_descriptor_prot), 1)

        # Input layer
        x = self.dropout(input_descriptor)
        x = self.fc(x)
        x = self.act(x)

        # Hidden layer
        for hidden_dropout, hidden_layer, hidden_activation_function in zip(self.hidden_dropout_layers,
                                                                            self.hidden_linear_layers,
                                                                            self.hidden_activations):
            x = hidden_dropout(x)
            x = hidden_layer(x)
            x = hidden_activation_function(x)

        # Output layer
        x = self.dropout_o(x)
        x = self.fc_o(x)

        return x

    def training_step(self, batch, batch_idx):

        input_descriptor_mol = batch['x_mol']
        input_descriptor_prot = batch['x_prot']
        labels = batch['y']
        target_idx = batch['target_idx']
        mean = batch["mean"]
        std = batch["std"]

        predictions = self.forward(input_descriptor_mol, input_descriptor_prot)
        predictions = torch.squeeze(predictions)  # This gives a prediction matrix for all targets

        loss = self.LossFunction(predictions, labels)

        #de-standardize predictions and label and compute loss to report
        pred_de_std = torch.mul(predictions, std)
        pred_de_std = torch.add(pred_de_std, mean)

        label_de_std = torch.mul(labels, std)
        label_de_std = torch.add(label_de_std, mean)

        loss_to_report = self.LossFunction(pred_de_std, label_de_std)

        output = {'loss': loss,
                  'loss_to_report': loss_to_report,
                  'predictions': predictions,
                  'labels': labels,
                  'target_idx': target_idx}
        return output

    def validation_step(self, batch, batch_idx):
        input_descriptor_mol = batch['x_mol']
        input_descriptor_prot = batch['x_prot']
        labels = batch['y']
        target_idx = batch['target_idx']
        mean = batch["mean"]
        std = batch["std"]

        predictions = self.forward(input_descriptor_mol, input_descriptor_prot)
        predictions = torch.squeeze(predictions)  # This gives a prediction matrix for all targets

        loss = self.LossFunction(predictions, labels)

        # de-standardize predictions and label and compute loss to report
        pred_de_std = torch.mul(predictions, std)
        pred_de_std = torch.add(pred_de_std, mean)

        label_de_std = torch.mul(labels, std)
        label_de_std = torch.add(label_de_std, mean)

        loss_to_report = self.LossFunction(pred_de_std, label_de_std)
        output = {'loss': loss,
                  'loss_to_report': loss_to_report,
                  'predictions': predictions,
                  'labels': labels,
                  'target_idx': target_idx}
        return output

    def training_epoch_end(self, step_outputs):
        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x['predictions'] for x in step_outputs], 0)
        labels = torch.cat([x['labels'] for x in step_outputs], 0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        epoch_loss_to_report = torch.mean(torch.tensor([x["loss_to_report"] for x in step_outputs]))
        target_ids = torch.cat([x['target_idx'] for x in step_outputs], 0)

        ci, _, _ = concordance_index_taskwise(predictions.cpu(), labels.cpu(), target_ids.cpu())

        epoch_dict = {'loss_train': epoch_loss, 'loss_to_report_train': epoch_loss_to_report, 'ci_train': ci}
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, 'training', on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_validation = dict()

        # Predictions
        predictions = torch.cat([x['predictions'] for x in step_outputs], 0)
        labels = torch.cat([x['labels'] for x in step_outputs], 0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        epoch_loss_to_report = torch.mean(torch.tensor([x["loss_to_report"] for x in step_outputs]))
        target_ids = torch.cat([x['target_idx'] for x in step_outputs], 0)

        ci, _, _ = concordance_index_taskwise(predictions.cpu(), labels.cpu(), target_ids.cpu())

        epoch_dict = {'loss_val': epoch_loss, 'loss_to_report_val': epoch_loss_to_report, 'ci_val': ci}
        log_dict_validation.update(epoch_dict)
        self.log_dict(log_dict_validation, 'validation', on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.cfg.training.lr,
                                     weight_decay=self.cfg.training.weight_decay)
        return optimizer
