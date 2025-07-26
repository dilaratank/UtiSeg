# unet model weights

u_net_model = 'dilaratank/MScUtiSeg/model-nw81559f:vbest' #Modality: VIDEO SCREENSHOTS
u_net_model = '' #Modality: ALL
u_net_model = '' #Modality: 3D screenshots
u_net_model = '' #Modality: STILL

import wandb
run = wandb.init()
artifact = run.use_artifact(u_net_model), type='model')
artifact_dir = artifact.download()
