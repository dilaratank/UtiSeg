"""
eval.py

A script to load and evaluate a trained U-Net model and write its results to wandb.
"""

import wandb
import argparse
from train import Segmenter
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from dataloaders import get_dataloaders
from train import get_config

def load_checkpoint(path_to_ckpt):
    """
    A function to return the loaded checkpoint path to a trained wandb model 
    
    Parameters 
    ----------
    path_to_ckpt: The path to the wandb checkpoints retrieved from the artifacts in a wandb run

    Returns 
    -------
    The loaded model checkpoint path from wandb
    """
    api = wandb.Api()
    artifact = api.artifact(path_to_ckpt)
    artifact_dir = artifact.download()
    return artifact_dir + '/model.ckpt'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--imaging_type", type=str, help="STILL, VIDEO or, 3D")
    parser.add_argument("--run_name", type=str, help="Name of the wandb run")
    parser.add_argument("--model_path", type=str, help="Location of the model")
    parser.add_argument("--img_size", type=int, default=128, help="Size of images that the model was trained on")
    parser.add_argument("--clahe", action="store_true", default=False, help="Whether to apply a CLAHE transformation to the images. Default False")
    parser.add_argument("--gaussian_blur", action="store_true", default=False, help="Whether to apply a CLAHE transformation to the images. Default False")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--padding", action="store_true", default=False, help="Whether to apply a padding to the images. Default False")
    parser.add_argument("--f", type=int, default=None, help="Fold for the cross validation")

    args = parser.parse_args()

    config_segm = get_config(args)

    # Load model from wandb checkpoint
    model_checkpoint = load_checkpoint(args.model_path)
    model = Segmenter.load_from_checkpoint(model_checkpoint)
    model.eval()

    wandb_logger = WandbLogger(project='MScUtiSeg', name=args.run_name, tags=["evaluate"])
    wandb_logger.experiment.config.update(vars(args))

    trainer = pl.Trainer(logger = wandb_logger)

    _, val_dataloader, test_dataloader = get_dataloaders(args.imaging_type, 1, args.img_size, clahe=args.clahe, gaussian_blur=args.gaussian_blur, padding=args.padding, f=args.f)

    # Run model on validation and test set, store results in a table and write to wandb
    valid_metrics = trainer.validate(model, dataloaders=val_dataloader,  ckpt_path=model_checkpoint, verbose=True)
    test_metrics = trainer.test(model, dataloaders=test_dataloader, ckpt_path=model_checkpoint, verbose=True)

    val_table_data = [[key, value] for key, value in valid_metrics[0].items()]
    test_table_data = [[key, value] for key, value in test_metrics[0].items()]

    wandb_logger.log_table(key="val_samples", columns=["Validation Metric", "Value"], data=val_table_data)
    wandb_logger.log_table(key="test_samples", columns=["Test Metric", "Value"], data=test_table_data)

    
