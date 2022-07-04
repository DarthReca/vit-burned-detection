import argparse
import os
from pathlib import Path

import comet_ml
import pytorch_lightning.loggers as loggers
import pytorch_lightning as pl
import torch
import pathlib

import loss
from lightning_modules import LitModel, LitDataModule
import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7, help="Custom seed.")
    parser.add_argument("--tag", default="test")
    parser.add_argument(
        "--steps",
        nargs="*",
        default=["train"],
        choices=["scale_batch", "lr_find", "train", "test"],
    )
    parser.add_argument("--config_file", default="configs/reduced_config.yaml")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--test_set", default="purple")

    return parser.parse_args()


def train():
    # Compatibility for Windows
    temp_posix = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    args = get_args()
    # Read configuration
    parser = utils.ConfigurationParser(args.config_file)
    model_config = parser.get_configuration("model")
    trainer_config = utils.trainer_converter(parser.get_configuration("trainer"))
    dataset_config = utils.create_satellite_groups(parser.get_configuration("dataset"))
    comet_api = parser.get_configuration("comet_api")

    # Add test set
    dataset_config["key"] = args.test_set
    # Get informations from API
    api = comet_ml.API(comet_api["api_key"])
    experiments = api.get(comet_api["workspace"], comet_api["project_name"])

    # Select suffix for experiment
    suffixes = []
    for e in experiments:
        suffix = e.name.split("_")[-1]
        if suffix.isnumeric():
            suffixes.append(int(suffix))
        else:
            suffixes.append(0)
    n = max(suffixes) + 1 if len(suffixes) != 0 else 0
    comet_api[
        "experiment_name"
    ] = f"test_{model_config['model']['name']}_{dataset_config['seed']}_{n}"

    outdir = Path(f"logger_out/{comet_api['experiment_name']}")
    outdir.mkdir(parents=True, exist_ok=True)
    # Set common seed
    pl.seed_everything(dataset_config["seed"], True)

    print(f'Best checkpoints saved in "{outdir}"')
    datamodule = LitDataModule(**dataset_config)
    # Create model
    if args.ckpt_path is None:
        pl_model = LitModel(**model_config)
    else:
        pl_model = LitModel.load_from_checkpoint(args.ckpt_path)
    pathlib.PosixPath = temp_posix
    # Setup logger
    logger = loggers.CometLogger(**comet_api, save_dir=outdir)
    logger.experiment.add_tag(args.tag)

    trainer = pl.Trainer(**trainer_config, logger=logger)
    trainer.checkpoint_callback.dirpath = outdir
    trainer.checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    if "scale_batch" in args.steps:
        trainer.tuner.scale_batch_size(pl_model, datamodule=datamodule)
    if "lr_find" in args.steps:
        lr_finder = trainer.tuner.lr_find(pl_model, datamodule=datamodule)
        logger.experiment.log_figure(
            figure_name="Learning Rate Finder", figure=lr_finder.plot(suggest=True)
        )
        pl_model.hparams["lr"] = lr_finder.suggestion()
    if "train" in args.steps:
        trainer.fit(pl_model, datamodule=datamodule)
    if "test" in args.steps:
        trainer.test(pl_model, datamodule=datamodule)
    if "train" in args.steps:
        logger.experiment.log_model(
            "Best model", trainer.checkpoint_callback.best_model_path
        )
        if len(trainer.checkpoint_callback.best_k_models) != 0:
            last_topk_model = max(trainer.checkpoint_callback.best_k_models)
            logger.experiment.log_model("Last model", last_topk_model)
        timer = [c for c in trainer.callbacks if isinstance(c, pl.callbacks.Timer)][0]
        if timer.time_remaining() <= 1:
            trainer.save_checkpoint("last.ckpt")
            logger.experiment.log_model("Resume Checkpoint", "last.ckpt")
            os.remove("last.ckpt")
    logger.experiment.end()


if __name__ == "__main__":
    train()
