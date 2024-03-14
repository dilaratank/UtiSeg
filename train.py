# Training pipeline copied and adapted from https://www.kaggle.com/code/heiswicked/pytorch-lightning-unet-segmentation-tumour?rvi=1

# from dataloaders import STILL_train_dataloader, STILL_val_dataloader
from dataloaders import VIDEO_train_dataloader, VIDEO_val_dataloader
# from dataloaders import STILL_train_dataloader, STILL_val_dataloader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

import segmentation_models_pytorch as smp
import wandb
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

class Block(nn.Module):
    def __init__(self, inputs = 1, middles = 32, outs = 32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inputs, middles, 3, 1, 1)
        self.conv2 = nn.Conv2d(middles, outs, 3, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outs)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.bn(self.conv2(x)))
        return self.pool(x), x

class UNet(nn.Module):
    def __init__(self,):
        super().__init__()

        self.en1 = Block(1, 32, 32)
        self.en2 = Block(32, 64, 64)
        self.en3 = Block(64, 128, 128)
        self.en4 = Block(128, 256, 256)
        self.en5 = Block(256, 512, 256)
        
        self.upsample4 = nn.ConvTranspose2d(256, 256, 2, stride = 2)
        self.de4 = Block(512, 256, 128)
        
        self.upsample3 = nn.ConvTranspose2d(128, 128, 2, stride = 2)
        self.de3 = Block(256, 128, 64)
        
        self.upsample2 = nn.ConvTranspose2d(64, 64, 2, stride = 2)
        self.de2 = Block(128, 64, 32)
        
        self.upsample1 = nn.ConvTranspose2d(32, 32, 2, stride = 2)
        self.de1 = Block(64, 32, 32)
        
        self.conv_last = nn.Conv2d(32, 1, kernel_size=1, stride = 1, padding = 0)
        
    def forward(self, x):

        x, e1 = self.en1(x)
        x, e2 = self.en2(x)
        x, e3 = self.en3(x)
        x, e4 = self.en4(x)
        _, x = self.en5(x)
        
        x = self.upsample4(x)
        x = torch.cat([x, e4], dim=1)
        _,  x = self.de4(x)
        
        x = self.upsample3(x)
        x = torch.cat([x, e3], dim=1)
        _, x = self.de3(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, e2], dim=1)
        _, x = self.de2(x)
        
        x = self.upsample1(x)
        x = torch.cat([x, e1], dim=1)
        _, x = self.de1(x)

        x = self.conv_last(x)     
        return x
    
class UNETModel(pl.LightningModule):

    def __init__(self,):
        super().__init__()
        self.model = UNet()

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.val_table_data = []
        self.epoch = 0

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "img": image,
            "gt": mask,
            "pred": pred_mask,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        total_loss = 0
        iter_count = len(outputs)
    
        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        
        metrics = {
            f"{stage}/loss": total_loss/iter_count,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/accuracy": accuracy,
            f"{stage}/f1_score": f1_score,
            f"{stage}/per_image_iou": per_image_iou,
            f"{stage}/dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)


    def training_step(self, batch, batch_idx):
        self.train_step_outputs.append(self.shared_step(batch, "train"))
        return self.shared_step(batch, "train")            

    def on_train_epoch_end(self):
        return self.shared_epoch_end(self.train_step_outputs, "train")

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self.shared_step(batch, "valid"))
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        img = self.validation_step_outputs[self.epoch]['img']
        gt = self.validation_step_outputs[self.epoch]['gt']
        pred = self.validation_step_outputs[self.epoch]['pred']

        tp = self.validation_step_outputs[self.epoch]['tp']
        fp = self.validation_step_outputs[self.epoch]['fp']
        fn = self.validation_step_outputs[self.epoch]['fn']
        tn = self.validation_step_outputs[self.epoch]['tn']

        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        self.val_table_data.append([self.epoch, wandb.Image(img), wandb.Image(gt), wandb.Image(pred), f1_score])
        self.epoch += 1
        return self.shared_epoch_end(self.validation_step_outputs, "valid")

    def test_step(self, batch, batch_idx):
        self.test_step_outputs(self.shared_step(batch, "test")  )
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.test_step_outputs, "test")
    
    def on_train_end(self):
        print('val table data!:', self.val_table_data)
        val_table = wandb.Table(columns=['Epoch', 'Image', 'Ground truth', 'Prediction', 'F1 score'], data=self.val_table_data)
        wandb.log({"Video Validation Table": val_table})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

pl.seed_everything(2022)

# Model Checkpoint Setting
checkpoint_callback = ModelCheckpoint(monitor = "valid/f1_score", mode= 'max',
                                      filename= "model_best",
                                      dirpath ='./',
                                      save_top_k = 1, 
                                      save_weights_only=True
                                    )

# Early Stopping 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stop = EarlyStopping(monitor= "valid/f1_score", min_delta=0.00, patience = 20, verbose=True, mode="max")

## Scheduler
SWA = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=5, annealing_strategy='cos')

model = UNETModel()

wandb_logger = WandbLogger(project='MScUtiSeg')

trainer = pl.Trainer(
    logger = wandb_logger,
    max_epochs= 30,
    callbacks=[checkpoint_callback, early_stop, SWA],
    accelerator="gpu" if torch.cuda.is_available() else "auto",
    devices="auto")

trainer.fit(
    model, 
    train_dataloaders = VIDEO_train_dataloader,
    val_dataloaders = VIDEO_val_dataloader)

# check_path = "/home/sandbox/dtank/my-scratch/MScUtiSeg/model_best.ckpt"

# model.load_from_checkpoint(check_path)

# valid_metrics = trainer.validate(model, dataloaders=STILL_val_dataloader,  ckpt_path=check_path, verbose=True)
# pprint(valid_metrics)

# test_metrics = trainer.test(model, dataloaders=STILL_test_dataloader, ckpt_path=check_path, verbose=True)
# pprint(test_metrics)


        



