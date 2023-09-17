import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import wandb

from fully_connected_wurdinger import Fully_Connected_Wurdinger
from dataloader import Kiba_Data_module_hyper


@hydra.main(config_path="configs", config_name="cfg_wurdinger_sanity")
def sanity_check(cfg: OmegaConf):

    dm = Kiba_Data_module_hyper(cfg)

    model = Fully_Connected_Wurdinger(cfg)

    logger = pl_loggers.WandbLogger(save_dir='/system/user/publicwork/pcm_wurdinger/results/wandb_sanity/',
                                    name=cfg.model.name,
                                    project='pcm-wurdinger-sanityCheck-fnn')
    #checkpoint_callback = ModelCheckpoint(monitor="ci_val", mode='max', save_top_k=1)

    # trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback],
    #                     max_epochs=cfg.training.epochs)
    trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=cfg.training.epochs)

    trainer.fit(model=model, datamodule=dm)
    wandb.finish()


if __name__ == "__main__":
    sanity_check()