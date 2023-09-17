import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import wandb
import numpy as np

from hopfield_wurdinger import Hopfield_Module_Wurdinger
from dataloader import Kiba_Data_module_hyper


@hydra.main(config_path="configs", config_name="cfg_wurdinger_hopfield")
def run_hopfield(cfg: OmegaConf):

    dm = Kiba_Data_module_hyper(cfg)

    model = Hopfield_Module_Wurdinger(cfg)

    logger = pl_loggers.WandbLogger(save_dir='/system/user/publicdata/pcm_wurdinger/masterthesis/hopfield_trainable/',
                                    name=cfg.model.name,
                                    project='wurdinger-hopfield-testset')
    checkpoint_callback = ModelCheckpoint(monitor="ci_val", mode='max', save_top_k=1)

    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback],
                         max_epochs=cfg.training.epochs)

    trainer.fit(model=model, datamodule=dm)
    wandb.finish()


if __name__ == "__main__":
    seeds = np.load('seeds.npy')
    for seed in seeds:
        seed_everything(seed)
        run_hopfield()

