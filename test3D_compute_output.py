import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import training
import sys
from datasets import Cov19d_3Ddataset
import numpy as np
from pathlib import Path
import torchvision
from utils import modules
from torch.optim import lr_scheduler
import pandas as pd


# Define options
opt_defs = {}
# Dataset options
opt_defs["n_classes"] = dict(flags = ('-nc', '--nclasses'), info=dict(default=1, type=int, help="num of classes"))
opt_defs["mean"] = dict(flags = ('-mean', '--mean'), info=dict(default=[0.43216, 0.394666, 0.37645], type=float, help="mean for dataset normalization, default KINETICS normalization"))
opt_defs["std"] = dict(flags = ('-std', '--std'), info=dict(default=[0.22803, 0.22145, 0.216989], type=float, help="std for dataset normalization, default KINETICS normalization"))
opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../../dataset/Cov19d_subset/", type=str, help="path to dataset PC"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../../../dataset/Covid19/Cov19d/", type=str, help="path to dataset IPLAB"))
opt_defs["base_output_path"] = dict(flags = ('-bop', '--base-output-path'), info=dict(default="./test/test23-36/", type=str, help="where to save output"))
opt_defs["val_test"] = dict(flags = ('-vd','--val-test',), info=dict(default='test1', type=str, help="val or test"))
opt_defs["patient_name"] = dict(flags = ('-pn', '--patient-name'), info=dict(default='', type=str, help="patient name"))
opt_defs["patients"] = dict(flags = ('-patients', '--patients'), info=dict(default=[''], nargs ='+', type=str, help="patients to test (VAL)"))

#Model options
opt_defs["input_dim"] = dict(flags = ('-dim', '--input-dim'), info=dict(default=[112, 112], type=int, help="input dim")) #[300, 350]
opt_defs["model_name"] = dict(flags = ('-model-name', '--model-name'), info=dict(default='resnet3d', type=str, help="model e.g. resnet50, resnext50, resnest50")) #224
opt_defs["enabledmultiheadattention"] = dict(flags = ('-enabledmultiheadattention', '--enabledmultiheadattention'), info=dict(default=False, type=bool, help="enabled multi-head attention")) #224
opt_defs["numHeadMultiHeadAttention"] = dict(flags = ('-numHeadMultiHeadAttention', '--numHeadMultiHeadAttention'), info=dict(default=4, type=int, help="num head multi-head attention")) #224

# Training options
opt_defs["batch_size"] = dict(flags = ('-b', '--batch-size'), info=dict(default=1, type=int, help="batch size"))
opt_defs["optim"] = dict(flags = ('-o', '--optim'), info=dict(default="Adam", help="optimizer i.e. Adam, AdamW, SGD or RMSprop"))
opt_defs["learning_rate"] = dict(flags = ('-lr', '--learning-rate'), info=dict(default=1e-4, type=float, help="learning rate"))
opt_defs["lr_scheduler"] = dict(flags = ('-lrs', '--lr-scheduler'), info=dict(default='step', type=str, help="learning rate decay"))
opt_defs["lr_decay_rate"] = dict(flags = ('-lrdr', '--lr-decay-rate'), info=dict(default=1e-1, type=float, help="learning rate decay rate"))
opt_defs["lr_step_size"] = dict(flags = ('-lrss', '--lr-step-size'), info=dict(default=15, type=int, help="learning rate step size only when LR_SCHEDULER is step"))
opt_defs["lr_step_milestones"] = dict(flags = ('-lrsm', '--lr-step-milestones'), info=dict(default=[10, 15], type=int, help="learning rate decay rate only when LR_SCHEDULER is multistep"))
opt_defs["weight_decay"] = dict(flags = ('-wd', '--weight-decay',), info=dict(default=1e-4, type=float, help="weight decay"))

# Checkpoint options
opt_defs["results_path"] = dict(flags = ('-rp', '--results-path'), info=dict(default="./output3D/", type=str, help="path to results"))
opt_defs["weights_path"] = dict(flags = ('-wp', '--weights-path'), info=dict(default="./weights3D/", type=str, help="path to weights"))
opt_defs["weights_fname"] = dict(flags = ('-wf', '--weights-fname'), info=dict(default='weights-23-36.pth', type=str, help="weights name")) #weights-23-36.pth (resnet_pretrined) weights-24-9.pth or None

# Read options
import argparse
parser = argparse.ArgumentParser()
for k,arg in opt_defs.items():
    print(arg["flags"])
    parser.add_argument(*arg["flags"], **arg["info"])
opt = parser.parse_args(None)
print(opt)

#Dataset options
num_classi = opt.nclasses
mean = opt.mean 
std = opt.std 
DATASET_PATH = opt.dataset_path
base_output_path = opt.base_output_path
Path(base_output_path).mkdir(exist_ok=True)
val_test = opt.val_test
patient_name = opt.patient_name
patients = opt.patients

#Model options
input_dim = opt.input_dim
model_name = opt.model_name
enabledmultiheadattention = opt.enabledmultiheadattention
numHeadMultiHeadAttention = opt.numHeadMultiHeadAttention

# Training options
batch_size = opt.batch_size
optimizer = opt.optim
LR = opt.learning_rate
lr_sched = opt.lr_scheduler
LR_DECAY_RATE = opt.lr_decay_rate
LR_STEP_SIZE = opt.lr_step_size # only when LR_SCHEDULER is multistep 
LR_STEP_MILESTONES = opt.lr_step_milestones  # only when LR_SCHEDULER is multistep
WEIGHT_DECAY = opt.weight_decay

# Checkpoint options
RESULTS_PATH = opt.results_path 
WEIGHTS_PATH = opt.weights_path 
weights_fname = opt.weights_fname

# Test transform
test_transform = T.Compose([
    T.ToPILImage(),
    T.Resize([128, 171], interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(input_dim),
    T.ToTensor(),
    T.Normalize(torch.Tensor(mean).mean(dim=0).item(), torch.Tensor(std).mean(dim=0).item())
])

#IMPOSTO IL DEVICE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)



#3D MODEL
if model_name == 'resnet3d_pretrained':
    model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
    output_feature_dim = 512
    if enabledmultiheadattention:
        model.avgpool = nn.Sequential( # torch.Size([batch, 512, 8, 16, 16])
            model.avgpool, # torch.Size([batch, 512, 1, 1, 1])
            modules.Squeeze([4,3,2]), # torch.Size([batch, 512])
            modules.Unsqueeze([0]), # torch.Size([1, batch_size, 512])
            modules.MultiheadAttentionMod(output_feature_dim, numHeadMultiHeadAttention), # torch.Size([batch, 1, 512])
            modules.Squeeze([0]), # torch.Size([batch_size, 512])
        )
    model.fc = nn.Linear(output_feature_dim, num_classi)

weight = model.stem[0].weight.clone()
model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
with torch.no_grad():
    model.stem[0].weight = nn.Parameter(weight.mean(dim=1).unsqueeze(1))
    

# Move model to device
model=model.to(dev)


# Create optimizer
if optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
elif optimizer == "AdamW":
    optimizer = optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
elif optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
elif optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError("Optimizer chosen not implemented!")

# Scheduler Learning rate decay
if lr_sched == 'step':
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY_RATE)
elif lr_sched == 'multistep':
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_DECAY_RATE)
elif lr_sched == 'reduce_on_plateau':
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='max', 
                                                        factor=0.1, #8e-2
                                                        patience=10, #5
                                                        threshold=0.0001) #1e-2
else:
    raise ValueError('Learning rate scheduler not supported: {}'.format(lr_scheduler))

#LOAD WEIGHTS
training.load_weights(model, optimizer, os.path.join(WEIGHTS_PATH, weights_fname))

model.eval()

seq_size = 50
print("seq_size: ", seq_size)

patients = Cov19d_3Ddataset.check_name_patients(os.path.join(DATASET_PATH, val_test), seq_size)
#print(patients)
#sys.exit(0)


test_dataset = Cov19d_3Ddataset.Cov19d_3DScan_eval(DATASET_PATH, val_test, transform=test_transform, seq_size = seq_size, input_dim = input_dim)#, patient_name = str(p))
print("LUNGHEZZA TEST: ", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#SHOW SINGLE BATCH
#training.show_batch(test_loader)
#training.show_batch(test_loader)
#sys.exit(0)


#TEST
tot_predicted_labels, all_path = training.compute_output(model, test_loader, dev)
print("TOTAL PREDICTED LABELS: ", tot_predicted_labels)
print("TOTAL PATH: ", all_path)

dict_all=dict(zip(all_path, tot_predicted_labels))

cnn_one_pred_df=pd.DataFrame(list(dict_all.items()), columns=['path', 'pred'])
cnn_one_pred_df.to_csv(base_output_path + "/path_pred.csv",index=False)

df_results=pd.read_csv(base_output_path + "/path_pred.csv")

folder_path=base_output_path +"/Result"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
df_results["name"]=df_results["path"].apply(lambda x : x.split("/")[-1])
covid_ct=df_results[df_results["pred"]>0.5].name.values
non_covid_ct=df_results[df_results["pred"]<0.5].name.values

covid_df=pd.DataFrame(covid_ct,columns=["ct_name"])
non_covid_df=pd.DataFrame(non_covid_ct,columns=["ct_name"])
covid_df.sort_values(by=['ct_name'],inplace=True)
non_covid_df.sort_values(by=['ct_name'],inplace=True)

covid_df.to_csv(f"{folder_path}/covid.csv",header=False,index=False)
non_covid_df.to_csv(f"{folder_path}/non-covid.csv",header=False,index=False)
print("covid:",len(covid_df) , "  non_covid:",len(non_covid_df))

