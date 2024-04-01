'''
Code copied from Notebook provided by pracitcal.
We replaces the tensorboard logger with the wandb logger.
'''
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import argparse
import os

import torch
import torchmetrics
import torch.nn.functional as F

import pytorch_lightning as pl

from dataloaders import get_dataloaders

import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

'''
Code copied from Notebook provided by pracitcal.
'''
import torch 
import torch.nn.functional as F

def conv3x3_bn(ci, co):
  return torch.nn.Sequential(torch.nn.Conv2d(ci, co, 3, padding=1), torch.nn.BatchNorm2d(co), torch.nn.ReLU(inplace=True))

def encoder_conv(ci, co):
  return torch.nn.Sequential(torch.nn.MaxPool2d(2), conv3x3_bn(ci, co), conv3x3_bn(co, co))

class deconv(torch.nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
    self.conv1 = conv3x3_bn(ci, co)
    self.conv2 = conv3x3_bn(co, co)

  def forward(self, x1, x2):
    x1 = self.upsample(x1)
    diffX = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (diffX, 0, diffY, 0))
    # concatenating tensors
    x = torch.cat([x2, x1], dim=1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class UNet(torch.nn.Module):
  def __init__(self, n_classes=1, in_ch=1):
    super().__init__()

    # number of filter's list for each expanding and respecting contracting layer
    c = [16, 32, 64, 128]

    # first convolution layer receiving the image
    self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
                                     conv3x3_bn(c[0], c[0]))

    # encoder layers
    self.conv2 = encoder_conv(c[0], c[1])
    self.conv3 = encoder_conv(c[1], c[2])
    self.conv4 = encoder_conv(c[2], c[3])

    # decoder layers
    self.deconv1 = deconv(c[3],c[2])
    self.deconv2 = deconv(c[2],c[1])
    self.deconv3 = deconv(c[1],c[0])

    # last layer returning the output
    self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

  def forward(self, x):
    # encoder
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x = self.conv4(x3)
    # decoder
    x = self.deconv1(x, x3)
    x = self.deconv2(x, x2)
    x = self.deconv3(x, x1)
    x = self.out(x)
    return x

def get_config(args):
    config_segm = {
        'batch_size'     : args.batch_size,
        'optimizer_lr'   : args.optimizer_lr,
        'max_epochs'     : args.epochs,
        'model_name'     : 'unet',
        'optimizer_name' : 'adam',
        'bin'            : 'segm_models/',
        'experiment_name': 'unet'
    }
    return config_segm

models     = {'unet'      : UNet}

optimizers = {'adam'      : torch.optim.Adam,
              'sgd'       : torch.optim.SGD }

metrics    = {'acc'       : torchmetrics.Accuracy(task='binary').to('cuda'),
              'f1'        : torchmetrics.F1Score(task='binary').to('cuda'),
              'precision' : torchmetrics.Precision(task='binary').to('cuda'),
              'recall'    : torchmetrics.Recall(task='binary').to('cuda')}

class Segmenter(pl.LightningModule):
  def __init__(self, *args):
    super().__init__()

    if not args:
      config_segm = {}
      config_segm['model_name'] = 'unet'
      config_segm['optimizer_name'] = 'adam'
      config_segm['optimizer_lr'] = 0.1
    else:
      config_segm = args[0]

    # defining model
    self.model_name = config_segm['model_name']
    assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
    self.model      = models[self.model_name]().to('cuda')

    # assigning optimizer values
    self.optimizer_name = config_segm['optimizer_name']
    self.lr             = config_segm['optimizer_lr']

    # definfing loss function
    self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    self.val_table_data = []
    self.epoch = 0

  def step(self, batch, nn_set):
    X, y = batch
    X, y   = X.float().to('cuda'), y.to('cuda').float()
    y_hat  = self.model(X)
    y_prob = torch.sigmoid(y_hat)

    # pos_weight = torch.tensor([config_segm['loss_pos_weight']]).float().to('cuda')
    #loss = F.binary_cross_entropy_with_logits(y, y_prob, pos_weight=pos_weight)
    # loss = F.binary_cross_entropy_with_logits(y_hat, y.float())#, pos_weight=pos_weight)
    loss = self.loss_fn(y_hat, y)
    
    self.log(f"{nn_set}/loss", loss, on_step=False, on_epoch=True)

    for i, (metric_name, metric_fn) in enumerate(metrics.items()):
      score = metric_fn(y_prob, y.int())
      self.log(f'{nn_set}/{metric_name}', score, on_step=False, on_epoch=True)

    if nn_set == "val" and self.epoch % 10 == 0:
      print(f'saving a row of images on epoch {self.epoch}')
      img = batch[0][0].cpu()
      gt = batch[1][0].cpu()
      pred = y_prob[0].cpu()
      f1score = metrics['f1'](pred, gt.int())
      self.val_table_data.append([wandb.Image(img), wandb.Image(gt), wandb.Image(pred), f1score])

    del X, y_hat, batch
    return loss

  def training_step(self, batch, batch_idx):
    return {"loss": self.step(batch, "train")}

  def validation_step(self, batch, batch_idx):
    return {"val_loss": self.step(batch, "val")}
  
  def on_validation_epoch_end(self):
    self.epoch += 1

  def test_step(self, batch, batch_idx):
    return {"test_loss": self.step(batch, "test")}
  
  def on_train_end(self):
    val_table = wandb.Table(columns=['Image', 'Ground truth', 'Prediction', 'F1 score'], data=self.val_table_data)
    wandb.log({f"{args.run_name} Validation Table": val_table})

  def forward(self, X):
    return self.model(X)

  def configure_optimizers(self):
    assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
    return optimizers[self.optimizer_name](self.parameters(), lr = self.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training. Default: 32")
    parser.add_argument("--optimizer_lr", type=float, default=0.1, help="Learning rate for training. Default: 0.01")
    parser.add_argument("--imaging_type", type=str, help="STILL, VIDEO or, 3D")
    parser.add_argument("--img_size", type=int, default=128, help="Size of the image, must be divisible by 32")
    parser.add_argument("--epochs", type=int, default=10, help="Amount of training epochs. Default: 10")
    parser.add_argument("--run_name", type=str, help="Name of the wandb run")
    parser.add_argument("--clahe", action="store_true", default=False, help="Whether to apply a CLAHE transformation to the images. Default False")
    parser.add_argument("--gaussian_blur", action="store_true", default=False, help="Whether to apply a CLAHE transformation to the images. Default False")
    parser.add_argument("--random_rotation", action="store_true", default=False, help="Whether to apply general augmentations to the images. Default False")
    parser.add_argument("--padding", action="store_true", default=False, help="Whether to apply a padding to the images. Default False")
    # parser.add_argument("--loss_pos_weight", type=int, default=2, help=". Default 2")

    args = parser.parse_args()

    config_segm = get_config(args)

    wandb_logger = WandbLogger(log_model=True, project='MScUtiSeg', name=args.run_name)
    wandb_logger.experiment.config.update(config_segm)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args.imaging_type, args.batch_size, args.img_size, args.clahe, args.padding, args.random_rotation, args.gaussian_blur)

    segmenter           = Segmenter(config_segm)
    logger              = wandb_logger
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/f1', mode='max')
    early_stop          = EarlyStopping(monitor= "val/f1", min_delta=0.00, patience = 20, verbose=True, mode="max")
    SWA                 = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=5, annealing_strategy='cos')
    trainer             = pl.Trainer(devices=1, accelerator='gpu', max_epochs=config_segm['max_epochs'],
                                    logger=logger, callbacks=[checkpoint_callback, early_stop, SWA],
                                    default_root_dir=config_segm['bin'], deterministic=True,
                                    log_every_n_steps=1)
    trainer.fit(segmenter, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader)
    