import wandb
import argparse
from train import UNETModel
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from dataloaders import get_dataloaders

def load_checkpoint(path_to_ckpt):
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
    args = parser.parse_args()

    model_checkpoint = load_checkpoint(args.model_path)
    model = UNETModel.load_from_checkpoint(model_checkpoint)

    model.eval()

    wandb_logger = WandbLogger(project='MScUtiSeg', name=args.run_name, tags=["evaluate"])
    wandb_logger.experiment.config.update(vars(args))

    trainer = pl.Trainer(logger = wandb_logger)

    _, val_dataloader, test_dataloader = get_dataloaders(args.imaging_type, 1, args.img_size)

    valid_metrics = trainer.validate(model, dataloaders=val_dataloader,  ckpt_path=model_checkpoint, verbose=True)
    test_metrics = trainer.test(model, dataloaders=test_dataloader, ckpt_path=model_checkpoint, verbose=True)

    val_table_data = [[key, value] for key, value in valid_metrics[0].items()]
    test_table_data = [[key, value] for key, value in test_metrics[0].items()]

    # val_table = wandb.Table(data=val_table_data, columns=["Validation Metric", "Value"])
    # test_table = wandb.Table(data=test_table_data, columns=["Test Metric", "Value"])

    # wandb.log({"Eval Validation Metrics": val_table_data})
    # wandb.log({"Eval Test Metrics": test_table_data})

    wandb_logger.log_table(key="val_samples", columns=["Validation Metric", "Value"], data=val_table_data)
    wandb_logger.log_table(key="test_samples", columns=["Test Metric", "Value"], data=test_table_data)

    
