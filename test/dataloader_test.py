import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from src.pcm_models.dataloader import Kiba_Data_module_hyper

if __name__ == "__main__":
    initialize(config_path="../src/pcm_models/configs",
               job_name="dataloader_test")

    cfg = compose(config_name="cfg_wurdinger_hyper",
                  overrides=["data.path=/home/laurenz/Downloads/data/folds/hyperparaSearch/",
                             "training.batch_size=1",
                             "inference.batch_size=1",
                             "ressources.num_workers=1"
                             ])
    print(OmegaConf.to_yaml(cfg))

    dm = Kiba_Data_module_hyper(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f'Length train-loader: {len(train_loader)} with bach_size: {cfg.training.batch_size}')
    print(f'Length val-loader: {len(val_loader)} with bach_size: {cfg.inference.batch_size}')

    for batch in train_loader:
        #Control batch_size
        batch_size = cfg.training.batch_size
        size_of_last_batch = len(train_loader.dataset)-(len(train_loader)-1) * batch_size
        number_mol_vector = len(batch["x_mol"])
        assert number_mol_vector == batch_size or number_mol_vector == size_of_last_batch, \
            f"Batch size mol vector should equal batch_size({batch_size}) but got {number_mol_vector}"
        number_prot_vectors = len(batch["x_prot"])
        assert number_prot_vectors == batch_size or number_prot_vectors == size_of_last_batch, \
            f"Batch size prot vector should equal batch_size({batch_size}) but got {number_prot_vectors}"
        number_label_vectors = len(batch["y"])
        assert number_label_vectors == batch_size or number_label_vectors == size_of_last_batch, \
            f"Number feature vectors should equal batch_size({batch_size}) but got {number_label_vectors}"
        number_targetIds_vectors = len(batch["target_idx"])
        assert number_targetIds_vectors == batch_size or number_targetIds_vectors == size_of_last_batch, \
            f"Number feature vectors should equal batch_size({batch_size}) but got {number_targetIds_vectors}"

        #Control vector size and model match
        input_dim = cfg.model.architecture.input_dim
        len_feature_vectors = len(batch["x_mol"][0]) + len(batch["x_prot"][0])
        assert len_feature_vectors == input_dim, \
            f"Len feature vector should equal input_dim({input_dim}) but got {len_feature_vectors}"

        if batch_size == size_of_last_batch:
            output_dim = cfg.model.architecture.output_dim
            len_label_vectors = len(batch["y"])/batch_size
            assert len_label_vectors == output_dim, \
                f"Number feature vectors should equal output_dim({output_dim}) but got {len_label_vectors}"

        #Target ID must not be higher than 229
        max_target_id = int(torch.max(batch["target_idx"]))
        assert max_target_id <= 229, \
            f"Target ID must not be higher than 229 but got {max_target_id}"

    for batch in val_loader:
        # Control batch_size
        batch_size = cfg.inference.batch_size
        size_of_last_batch = len(val_loader.dataset) - (len(val_loader) - 1) * batch_size
        number_mol_vector = len(batch["x_mol"])
        assert number_mol_vector == batch_size or number_mol_vector == size_of_last_batch, \
            f"Batch size mol vector should equal batch_size({batch_size}) but got {number_mol_vector}"
        number_prot_vectors = len(batch["x_prot"])
        assert number_prot_vectors == batch_size or number_prot_vectors == size_of_last_batch, \
            f"Batch size prot vector should equal batch_size({batch_size}) but got {number_prot_vectors}"
        number_label_vectors = len(batch["y"])
        assert number_label_vectors == batch_size or number_label_vectors == size_of_last_batch, \
            f"Number feature vectors should equal batch_size({batch_size}) but got {number_label_vectors}"
        number_targetIds_vectors = len(batch["target_idx"])
        assert number_targetIds_vectors == batch_size or number_targetIds_vectors == size_of_last_batch, \
            f"Number feature vectors should equal batch_size({batch_size}) but got {number_targetIds_vectors}"

        # Control vector size and model match
        input_dim = cfg.model.architecture.input_dim
        len_feature_vectors = len(batch["x_mol"][0]) + len(batch["x_prot"][0])
        assert len_feature_vectors == input_dim, \
            f"Len feature vector should equal input_dim({input_dim}) but got {len_feature_vectors}"

        if batch_size == size_of_last_batch:
            output_dim = cfg.model.architecture.output_dim
            len_label_vectors = len(batch["y"]) / batch_size
            assert len_label_vectors == output_dim, \
                f"Number feature vectors should equal output_dim({output_dim}) but got {len_label_vectors}"

        # Target ID must not be higher than 229
        max_target_id = int(torch.max(batch["target_idx"]))
        assert max_target_id <= 229, \
            f"Target ID must not be higher than 229 but got {max_target_id}"
