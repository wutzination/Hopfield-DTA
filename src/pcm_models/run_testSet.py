import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import wandb
import numpy as np

from fully_connected_wurdinger import Fully_Connected_Wurdinger
from dataloader import Kiba_Data_module_hyper


@hydra.main(config_path="configs", config_name="cfg_wurdinger_test2")
def run_testset(cfg: OmegaConf):

    dm = Kiba_Data_module_hyper(cfg)

    model = Fully_Connected_Wurdinger(cfg)

    logger = pl_loggers.WandbLogger(save_dir='/system/user/publicwork/pcm_wurdinger/results/wand_testset',
                                    name=cfg.model.name,
                                    project='pcm-wurdinger-testset-OutputStd-3')
    checkpoint_callback = ModelCheckpoint(monitor="ci_val", mode='max', save_top_k=1)

    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback],
                         max_epochs=cfg.training.epochs)
    trainer.fit(model=model, datamodule=dm)
    wandb.finish()


if __name__ == "__main__":

    seeds = np.load('seeds.npy')

    for seed in seeds:
        seed_everything(seed)
        run_testset()

