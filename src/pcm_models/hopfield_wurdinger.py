import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch.nn as nn
import torch
from functools import partial
import numpy as np
#from run_hopfield.modules import Hopfield
from hflayers import Hopfield

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


class Hopfield_Module_Wurdinger(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super(Hopfield_Module_Wurdinger, self).__init__()

        # Config
        self.cfg = cfg
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        self.save_hyperparameters(config)

        # Loss functions
        self.LossFunction = MSE

        #Hopfield Memory
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mem_prot = torch.from_numpy(np.load(cfg.data.path + cfg.data.mem_prot)).to(device)
        self.mem_prot = self.mem_prot.reshape(1,
                                              self.mem_prot.shape[0],
                                              self.mem_prot.shape[1])

        self.mem_mol = torch.from_numpy(np.load(cfg.data.path + cfg.data.mem_mol)).to(device)
        self.mem_mol = self.mem_mol.reshape(1,
                                            self.mem_mol.shape[0],
                                            self.mem_mol.shape[1])

        # Architecture
        # Input layer
        self.dropout = dropout_mapping[cfg.model.architecture.activation](
            cfg.model.regularization.input_dropout)
        self.fc = nn.Linear(cfg.model.architecture.input_dim_prot + cfg.model.architecture.input_dim_mol,
                            cfg.model.architecture.number_hidden_neurons)
        self.act = activation_function_mapping[cfg.model.architecture.activation]

        # Layer normalization before Hopfield
        self.layerNormProt1 = nn.LayerNorm(normalized_shape=([
                                                              cfg.model.architecture.input_dim_prot]),
                                           elementwise_affine=False)
        self.layerNormMol1 = nn.LayerNorm(normalized_shape=([
                                                             cfg.model.architecture.input_dim_mol]),
                                           elementwise_affine=False)

        #Hopfield layer
        # Hopfield block WITHOUT trainable parameters
        self.hopfield_prot = Hopfield(
            input_size=cfg.model.architecture.input_dim_prot,
            # Input dimension for queries --- here: dim of the protein representation
            hidden_size=cfg.model.architecture.hidden_size_prot,  # Dimension of similarity space --- Querie-, Key-dims
            stored_pattern_size=self.mem_prot.shape[2],  # Input dimension for keys --- here: dim of memory
            pattern_projection_size=self.mem_prot.shape[2],  # Input dimension for values -- here: dim of memory
            output_size=cfg.model.architecture.input_dim_prot,  # Dimension of the retrieval
            num_heads=cfg.model.architecture.num_heads_prot,  # Number of heads
            scaling=cfg.model.architecture.hop_scaling_prot,  # <- important hyperparameter
            dropout=0.5
        )

        # Hopfield block WITH trainable parameters
        self.hopfield_mol = Hopfield(
            input_size=cfg.model.architecture.input_dim_mol,
            # Input dimension for queries --- here: dim of the protein representation
            hidden_size=cfg.model.architecture.hidden_size_mol,  # Dimension of similarity space --- Querie-, Key-dims
            stored_pattern_size=self.mem_mol.shape[2],  # Input dimension for keys --- here: dim of memory
            pattern_projection_size=self.mem_mol.shape[2],  # Input dimension for values -- here: dim of memory
            output_size=cfg.model.architecture.input_dim_mol,  # Dimension of the retrieval
            num_heads=cfg.model.architecture.num_heads_mol,  # Number of heads
            scaling=cfg.model.architecture.hop_scaling_mol,  # <- important hyperparameter
            dropout=0.5
        )

        # Layer normalization after Hopfield
        self.layerNormProt2 = nn.LayerNorm(normalized_shape=([
                                                              cfg.model.architecture.input_dim_prot]),
                                           elementwise_affine=False)
        self.layerNormMol2 = nn.LayerNorm(normalized_shape=([
                                                             cfg.model.architecture.input_dim_mol]),
                                           elementwise_affine=False)

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

        input_descriptor_prot = self.layerNormProt1(input_descriptor_prot)
        input_descriptor_mol = self.layerNormMol1(input_descriptor_mol)

        shape_prot = input_descriptor_prot.shape
        shape_mol = input_descriptor_mol.shape

        input_descriptor_mol = input_descriptor_mol.reshape(1,
                                                            shape_mol[0],
                                                            shape_mol[1])
        input_descriptor_prot = input_descriptor_prot.reshape(1,
                                                              shape_prot[0],
                                                              shape_prot[1])

        #print(input_descriptor_mol.shape)
        #print(input_descriptor_prot.shape)

        #print(self.mem_mol.shape)
        #print(self.mem_prot.shape)

        retrieval_prot = self.hopfield_prot((self.mem_prot, input_descriptor_prot, self.mem_prot))
        #print(retrieval_prot.shape)
        retrieval_mol = self.hopfield_mol((self.mem_mol, input_descriptor_mol, self.mem_mol))
        #print(retrieval_mol.shape)

        input_descriptor_prot = input_descriptor_prot + retrieval_prot
        input_descriptor_mol = input_descriptor_mol + retrieval_mol

        input_descriptor_prot = input_descriptor_prot.reshape(shape_prot[0],
                                                              shape_prot[1])

        input_descriptor_mol = input_descriptor_mol.reshape(shape_mol[0],
                                                            shape_mol[1])
        #print(input_descriptor_prot.shape)
        #print(input_descriptor_mol.shape)

        input_descriptor_prot = self.layerNormProt2(input_descriptor_prot)
        input_descriptor_mol = self.layerNormMol2(input_descriptor_mol)

        input_descriptor = torch.cat((input_descriptor_mol, input_descriptor_prot), 1)
        #print(input_descriptor.shape)
        # Input layer
        x = self.dropout(input_descriptor)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
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
