# unet model weights

u_net_model = 'dilaratank/MScUtiSeg/model-nw81559f:vbest' #Modality: VIDEO SCREENSHOTS
u_net_model = 'dilaratank/MScUtiSeg/model-or6fn3ep:vbest' #Modality: ALL
u_net_model = 'dilaratank/MScUtiSeg/model-b7dchl80:vbest' #Modality: 3D screenshots
u_net_model = 'dilaratank/MScUtiSeg/model-9y1afl1k:vbest' #Modality: STILL

import wandb
run = wandb.init()
artifact = run.use_artifact(u_net_model), type='model')
artifact_dir = artifact.download()
