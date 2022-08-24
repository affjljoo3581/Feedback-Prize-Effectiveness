from __future__ import annotations

import argparse
import os
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import MyLightningDataModule, MyLightningModule

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(config: DictConfig):
    name = [
        config.train.name,
        f"do{config.data.random_seed}",
        f"wi{config.model.random_seed}",
        f"fold{config.data.fold_index}",
    ]
    name = "-".join(name)

    module, datamodule = MyLightningModule(config), MyLightningDataModule(config)
    checkpoint = ModelCheckpoint(monitor="val/loss", mode="min", save_weights_only=True)

    Trainer(
        gpus=1,
        precision=16,
        amp_backend=amp_backend,
        log_every_n_steps=config.train.log_every_n_steps,
        max_steps=config.optim.scheduler.num_training_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        limit_val_batches=0.0 if config.train.evaluate_start_step > 0 else 1.0,
        callbacks=[checkpoint],
        logger=WandbLogger(name, project="feedback-prize-effectiveness"),
    ).fit(module, datamodule)

    # Save the best-scored trained model weights and its tokenizer.
    module = MyLightningModule.load_from_checkpoint(
        checkpoint.best_model_path, config=config
    )
    module.model.save_pretrained(name)
    datamodule.tokenizer.save_pretrained(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config)
