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
import torchvision
from utils import modules
from torch.optim import lr_scheduler


# Define options
opt_defs = {}
# Dataset options
opt_defs["n_classes"] = dict(flags = ('-nc', '--nclasses'), info=dict(default=2, type=int, help="num of classes"))
opt_defs["mean"] = dict(flags = ('-mean', '--mean'), info=dict(default=[0.43216, 0.394666, 0.37645], type=float, help="mean for dataset normalization, default KINETICS normalization"))
opt_defs["std"] = dict(flags = ('-std', '--std'), info=dict(default=[0.22803, 0.22145, 0.216989], type=float, help="std for dataset normalization, default KINETICS normalization"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../../dataset/Cov19d_subset/", type=str, help="path to dataset PC"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="D:/Alessia/4_CovidChallenge/dataset/Cov19d", type=str, help="path to dataset HD"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../../../../storage/data_4T/alessia_medical_datasets/Covid19/Cov19d/", type=str, help="path to dataset SYLVANAS"))
opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../../../dataset/Covid19/Cov19d/", type=str, help="path to dataset IPLAB"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../../../dataset/Covid19/Cov19d_1/Cov19d/", type=str, help="path to dataset MERGE IPLAB"))
opt_defs["val_test"] = dict(flags = ('-vd','--val-test',), info=dict(default='validation', type=str, help="val or test"))

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
opt_defs["loss_type"] = dict(flags = ('-lt', '--loss-type'), info=dict(default='cross_entropy', type = str, help="the type of loss, i.e. cross_entropy"))
opt_defs["num_epochs"] = dict(flags = ('-ne', '--num-epochs',), info=dict(default=2, type=int, help="training epochs"))
opt_defs["regularization"] = dict(flags = ('-re', '--regularization',), info=dict(default='', type=str, help="regularization option"))
opt_defs["num_workers"] = dict(flags = ('-nw', '--num-workers'), info=dict(default=1, type=int, help="num workers in dataloader"))

# Checkpoint options
opt_defs["results_path"] = dict(flags = ('-rp', '--results-path'), info=dict(default="./output3D/", type=str, help="path to results"))
opt_defs["weights_path"] = dict(flags = ('-wp', '--weights-path'), info=dict(default="./weights3D/", type=str, help="path to weights"))
opt_defs["weights_fname"] = dict(flags = ('-wf', '--weights-fname'), info=dict(default=None, type=str, help="weights name")) #weights-34.pth or None
opt_defs["last_tag"] = dict(flags = ('-t', '--last-tag'), info=dict(default=0, type=int, help="last checkpoint tag"))


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
val_test = opt.val_test

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
loss_type = opt.loss_type
N_EPOCHS = opt.num_epochs
regularization = opt.regularization
num_workers = opt.num_workers

# Checkpoint options
RESULTS_PATH = opt.results_path
WEIGHTS_PATH = opt.weights_path 
weights_fname = opt.weights_fname
last_tag = opt.last_tag

tag = last_tag + 1
last_tag = tag

# Train transform
train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize([128, 171], interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(input_dim),
    T.ToTensor(),
    T.Normalize(torch.Tensor(mean).mean(dim=0).item(), torch.Tensor(std).mean(dim=0).item())
])


#IMPOSTO IL DEVICE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

if __name__ == '__main__':
    seq_size = 50
    print("seq_size: ", seq_size)
    #num_patient = len(all_patient)

    num_all_slice, num_all_patient = Cov19d_3Ddataset.check_number_slice(DATASET_PATH)
    print(num_all_patient, num_all_slice)

    #train_dataset, val_dataset = torch.utils.data.random_split(all_train_dataset, [0.8,0.2]) #[2211,552] IPLAB Seq_size=64
    train_dataset = Cov19d_3Ddataset.Cov19d_3DScan(DATASET_PATH, 'train', transform=train_transform, seq_size = seq_size, input_dim = input_dim)
    val_dataset = Cov19d_3Ddataset.Cov19d_3DScan(DATASET_PATH, val_test, transform=train_transform, seq_size = seq_size, input_dim = input_dim)

    print("LUNGHEZZA TRAIN: ", len(train_dataset))
    print("LUNGHEZZA VAL: ", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers = num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, persistent_workers=True)

    #SHOW SINGLE BATCH
    #training.show_batch(train_loader)
    #training.show_batch(val_loader)
    #sys.exit(0)


    dataloaders = {"train": train_loader,
            "val": val_loader}

    #3D MODEL
    if model_name == 'resnet3d_pretrained':
        model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        output_feature_dim = 512
        if enabledmultiheadattention:
            model.avgpool = nn.Sequential( # torch.Size([batch_size, 512, 8, 16, 16])
                model.avgpool, # torch.Size([batch_size, 512, 1, 1, 1])
                modules.Squeeze([4,3,2]), # torch.Size([batch_size, 512])
                modules.Unsqueeze([0]), # torch.Size([1, batch_size, 512])
                modules.MultiheadAttentionMod(output_feature_dim, numHeadMultiHeadAttention), # torch.Size([batch_size, 1, 512])
                modules.Squeeze([0]), # torch.Size([batch_size, 512])
            )
        model.fc = nn.Linear(output_feature_dim, num_classi)


    #REDUCE INPUT CHANNEL
    weight = model.stem[0].weight.clone()
    model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    with torch.no_grad():
        model.stem[0].weight = nn.Parameter(weight.mean(dim=1).unsqueeze(1))
        
    # Move model to device
    model=model.to(dev)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
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


    # Initialize training history
    history_loss = {"train": [], "val": [], "test": []}
    history_accuracy = {"train": [], "val": [], "test": []}
    # Initialize best validation accuracy and test accuracy at best validation accuracy
    best_val_accuracy = 0
    test_accuracy_at_best_val = 0
    batch_accuracy=0
    epoch_loss={"train": [], "val": [], "test": []}
    epoch_accuracy={"train": [], "val": [], "test": []}

    if weights_fname is None:
            start_epoch = 1
    else:
            start_epoch, history_loss, history_accuracy = training.load_weights(model, optimizer, os.path.join(WEIGHTS_PATH, weights_fname))

    #TRAIN
    model, best_val_accuracy, epoch_best_model, optimizer,history_loss,history_accuracy = training.train(
      model,
      dataloaders,
      optimizer,
      exp_lr_scheduler,
      loss_type,
      dev,
      start_epoch,
      N_EPOCHS,
      best_val_accuracy,
      history_loss,
      history_accuracy,
      WEIGHTS_PATH,
      tag,
      regularization
    )  # train model

    print("Migliore epoca VAL: ", epoch_best_model)
    # Plot loss history
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(tag)
    for split in ["train", "val"]:#, "test"]:
        plt.plot(history_loss[split], label=split)
    plt.legend()
    plt.savefig(RESULTS_PATH + 'Loss_{}_bs{}_lr{}_tag{}.png'.format(model_name, batch_size, LR, tag))
    plt.close()

    # Plot accuracy history
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(tag)
    for split in ["train", "val"]:#, "test"]:
        plt.plot(history_accuracy[split], label=split)
    plt.axvline(x=epoch_best_model-1, color='green', ls=':', lw=2)
    plt.legend()
    plt.savefig(RESULTS_PATH +'Accuracy_{}_bs{}_lr{}_tag{}.png'.format(model_name, batch_size, LR, tag))
    plt.close()

    # Get random sample from test set
    import random
    idx = random.randint(0, len(val_dataset)-1)
    input, label = val_dataset[idx]
    model.eval()
    with torch.no_grad():
        output = model(input.unsqueeze(0).to(dev))
    _,pred = output.max(1)
    pred = pred.item()
    print(f"Random input Predicted: {pred} (correct: {label})")


    correct = []
    errate = []
    tot_true_labels = []
    tot_predicted_labels = []
        
    for i, patient in enumerate(val_dataset):
        input, label = val_dataset[i]
        # Predict class
        tot_true_labels.append(label)
        model.eval()
        with torch.no_grad():
            output = model(input.unsqueeze(0).to(dev))
        _,pred = output.max(1)
        pred = pred.item()
        tot_predicted_labels.append(pred)
        if pred == label:
            correct.append(pred)
        else:
            errate.append(pred)
    print(f"Predette corrette: {len(correct)} (Predette errate: {len(errate)})")
    