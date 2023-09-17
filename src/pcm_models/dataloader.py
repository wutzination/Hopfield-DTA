import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from configs.config_instance import MyConfig, WurdingerConfigHopfieldHyper

class Kiba_Data_module_hyper(pl.LightningModule):
    def __init__(self, cfg: WurdingerConfigHopfieldHyper):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        # Load data:
        (mol_ids_train, target_ids_train, labels_train,
         mol_ids_val, target_ids_val, labels_val,
         mol_features_train, target_features_train,
         mol_features_val, target_features_val,
         std_mean) = self._load_triplets_and_split_data()

        self.train_dataset = self._BaseDataClass(mol_ids_train, target_ids_train, labels_train,
                                                 mol_features_train, target_features_train, std_mean)
        self.val_dataset = self._BaseDataClass(mol_ids_val, target_ids_val, labels_val,
                                               mol_features_val, target_features_val, std_mean)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.training.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.ressources.num_workers,
                          )#drop_last=True)

    def val_dataloader(self):
         return DataLoader(self.val_dataset,
                           batch_size=self.cfg.inference.batch_size,
                           shuffle=False,
                           num_workers=self.cfg.ressources.num_workers,
                           )#drop_last=True)

    def _load_triplets_and_split_data(self):
        """
        This function loads the preprocessed data.
        :return:
        """

        #load data
        train_affi = np.load(self.cfg.data.path + self.cfg.data.affi_train)
        val_affi = np.load(self.cfg.data.path + self.cfg.data.affi_val)

        train_ecfp = np.load(self.cfg.data.path + self.cfg.data.train_ecfp)
        val_ecfp = np.load(self.cfg.data.path + self.cfg.data.val_ecfp)

        train_seq = np.load(self.cfg.data.path + self.cfg.data.train_seq)
        val_seq = np.load(self.cfg.data.path + self.cfg.data.val_seq)

        std_mean = np.load(self.cfg.data.path + self.cfg.data.std_mean)

        # Splits
        # Training
        mol_ids_train = train_affi[:, 0]
        target_ids_train = train_affi[:, 1]
        labels_train = train_affi[:, 2].astype('float32')

        # Validation
        mol_ids_val = val_affi[:, 0]
        target_ids_val = val_affi[:, 1]
        labels_val = val_affi[:, 2].astype('float32')

        return mol_ids_train, target_ids_train, labels_train,\
               mol_ids_val, target_ids_val, labels_val,\
               train_ecfp, train_seq,\
               val_ecfp, val_seq, \
               std_mean


    class _BaseDataClass(Dataset):
        def __init__(self, mol_ids, target_ids, labels, features_mol, features_target, std_mean):
            self.mol_ids = mol_ids
            self.target_ids = target_ids
            self.labels = labels
            self.features_mol = features_mol
            self.features_target = features_target
            self.std_mean = std_mean

        def __getitem__(self, index):
            mol_id = self.mol_ids[index]
            target_idx = self.target_ids[index]

            std = self.std_mean[0]
            mean = self.std_mean[1]

            y = self.labels[index]
            x_mol = self.features_mol[index]
            x_prot = self.features_target[index]
            sample = {'x_mol': x_mol, 'x_prot': x_prot, 'y': y, 'target_idx': target_idx, 'std': std, 'mean': mean}
            return sample

        def __len__(self):
            length = self.mol_ids.shape[0]
            return length

class Tox21_Data_Module(pl.LightningDataModule):

    def __init__(self, cfg: MyConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        # Load data:
        (mol_ids_train, target_ids_train, labels_train,
        mol_ids_val, target_ids_val, labels_val,
        mol_ids_test, target_ids_test, labels_test,
        feature_mtx) = self._load_triplets_and_split_data()

        self.train_dataset = self._BaseDataClass(mol_ids_train, target_ids_train, labels_train, feature_mtx)
        self.val_dataset = self._BaseDataClass(mol_ids_val, target_ids_val, labels_val, feature_mtx)
        self.test_dataset = self._BaseDataClass(mol_ids_test, target_ids_test, labels_test, feature_mtx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.training.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.ressources.num_workers,
                          drop_last=True)

    def val_dataloader(self):
         return DataLoader(self.val_dataset,
                          batch_size=self.cfg.inference.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.ressources.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.inference.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.resources.num_workers,
                          drop_last=True)

    def _load_triplets_and_split_data(self, val_fold=3, test_fold=4):
        """
        This function loads the preprocessed data.
        :return:
        """

        split = np.load(self.cfg.data.path + self.cfg.data.split).flatten()
        mol_ids = np.load(self.cfg.data.path + self.cfg.data.mol_id).flatten()
        target_ids = np.load(self.cfg.data.path + self.cfg.data.target_id).flatten()
        labels = np.load(self.cfg.data.path + self.cfg.data.labels).flatten().astype('float32')
        feature_mtx = np.load(self.cfg.data.path + self.cfg.data.features)

        # Splits
        # Training
        mol_ids_train = mol_ids[(split != val_fold) * (split != test_fold)]
        target_ids_train = target_ids[(split != val_fold) * (split != test_fold)]
        labels_train = labels[(split != val_fold) * (split != test_fold)]

        # Validation
        mol_ids_val = mol_ids[split == val_fold]
        target_ids_val = target_ids[split == val_fold]
        labels_val = labels[split == val_fold]

        # Test
        mol_ids_test = mol_ids[split == test_fold]
        target_ids_test = target_ids[split == test_fold]
        labels_test = labels[split == test_fold]

        return mol_ids_train, target_ids_train, labels_train,\
               mol_ids_val, target_ids_val, labels_val, \
               mol_ids_test, target_ids_test, labels_test,\
               feature_mtx

    class _BaseDataClass(Dataset):
        def __init__(self, mol_ids, target_ids, labels, features):
            self.mol_ids = mol_ids
            self.target_ids = target_ids
            self.labels = labels
            self.features = features

        def __getitem__(self, index):
            mol_id = self.mol_ids[index]
            target_idx = self.target_ids[index]
            y = self.labels[index]

            x = self.features[[mol_id], :]

            sample = {'x': x, 'y': y, 'target_idx': target_idx}
            return sample

        def __len__(self):
            length = self.mol_ids.shape[0]
            return length

