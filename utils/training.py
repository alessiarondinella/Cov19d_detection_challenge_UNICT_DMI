import torch
import time
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import shutil
import os
from jacobian import JacobianReg
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
from pathlib import Path
import datetime
from tqdm import tqdm


def show_batch(dl):
    for images, labels in dl:
        print(labels)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        print(images.size())
        images = images[0].permute(1,0,2,3)
        ax.imshow(make_grid(images[:900], nrow=10, normalize=True).permute(1, 2, 0))
        plt.show()
        break

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    np.array(img,np.int32)
    images = img[0].numpy()
    print(np.shape(images))
    plt.imshow(images)
    plt.show()

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.2
    epochs_drop = 3.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    print ("epoch: " + str(epoch) + ", learning rate: " + str(lrate)) # added
    return lrate
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def save_weights(model, optim, tag, epoch, loss, best_val_accuracy, history_loss, history_accuracy, weights_path):
    weights_fname = 'weights-%d-%d.pth' % (tag, epoch)
    weights_fpath = os.path.join(weights_path, weights_fname)
    torch.save({
        'startEpoch': epoch + 1,
        'loss': loss,
        'best_val_accuracy': best_val_accuracy,
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'history_loss': history_loss,
        'history_accuracy': history_accuracy
    }, weights_fpath)
    shutil.copyfile(weights_fpath, os.path.join(weights_path, 'latest.th'))


def load_weights(model, optimizer, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    history_loss = weights['history_loss']
    history_accuracy = weights['history_accuracy']
    model.load_state_dict(weights['model_state'])
    optimizer.load_state_dict(weights['optim_state'])
    print("loaded weights (lastEpoch {}, loss {}, accuracy_val {})"
          .format(startEpoch - 1, weights['loss'], weights['best_val_accuracy']))
    return startEpoch, history_loss, history_accuracy 

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def normalize(tensor):
    m1 = tensor.min();
    m2 = tensor.max();
    tensor = (tensor-m1)/(m2-m1)
    return tensor

def train(model, dataloaders, optimizer, scheduler, loss_type, device, start_epoch, num_epochs, best_val_accuracy, history_loss, history_accuracy ,path_weight, tag, regularization):#, class_weight):
  #inizializzazione parametri
  batch_accuracy=0
  epoch_loss={"train": [], "val": []}#, "test": []}
  epoch_acc={"train": [], "val": []}#, "test": []}
  
  try:
    for epoch in  range(start_epoch, num_epochs + start_epoch):
      since = time.time()
      #FILE DI LOG
      log_file = 'checkpoint_{}.txt'.format(tag)
      log = open(log_file, 'a')
      log.write("Checkpoint 0: \n ")
      log.write("  -----  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
      log.close()

      sum_loss = {"train": 0, "val": 0}#, "test": 0}
      sum_accuracy = {"train": 0, "val": 0}#, "test": 0}
      print("-" * 100)
      print("Epoch {}/{}".format(epoch, num_epochs))

      # processo per ogni fase, quindi train e val
      for phase in ["train", "val"]:#, "test"]:
      #for phase in ["train"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        

        if loss_type == "cross_entropy": 
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "bce_logit": 
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "bce": 
            criterion = nn.BCELoss()

        if regularization == 'jacobian':
            reg = JacobianReg() # Jacobian regularization
            lambda_JR = 0.01 # hyperparameter
        
        # Iterate over data.
        loop = tqdm(dataloaders[phase])
        for i,(inputs, labels) in enumerate(loop): #-------------------loader
          
          # Move to CUDA 
          inputs = inputs.to(device)
          labels = labels.to(device)#.long() long per cross entropy
          
          if regularization == 'jacobian':
            inputs.requires_grad = True

          # zero the parameter gradients
          optimizer.zero_grad()

          # Compute loss
          pred = model(inputs)

          if loss_type == "cross_entropy":
            loss_super = criterion(pred, labels) # supervised loss
          elif loss_type == "bce_logit":
            loss_super = criterion(pred, labels.unsqueeze(1)) # supervised loss
          elif loss_type == "bce":
            loss_super = criterion(pred, labels.unsqueeze(1)) # supervised loss

          if regularization == 'jacobian':
              R = reg(inputs, pred)   # Jacobian regularization
              loss = loss_super + lambda_JR*R # full loss
          else:
              loss = loss_super

          # Update variables for average epoch loss
          sum_loss[phase] += loss.item()   
          
          # Backward and optimize
          if phase=="train": 
            loss.backward()
            optimizer.step()

          # Compute accuracy
          if loss_type == "cross_entropy":
                _, preds = pred.max(1)
                batch_accuracy = (preds == labels).sum().item()/inputs.size(0)
          elif loss_type == "bce_logit":         
                preds = torch.round(torch.sigmoid(pred))      
                batch_accuracy = (preds == labels.unsqueeze(1)).sum().item()/inputs.size(0)
          elif loss_type == "bce":
                preds = np.round(pred.detach() ) 
                labels = np.round(labels.detach())
                batch_accuracy = (pred == labels.unsqueeze(1)).sum().item()/inputs.size(0)
          # Update variables for average epoch accuracy 
          sum_accuracy[phase] += batch_accuracy


          loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
          loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item()) 
        
        if phase == 'train':
            scheduler.step()    
        
        epoch_loss = {phase: sum_loss[phase]/len(dataloaders[phase]) for phase in ["train", "val"]}#, "test"]}
        epoch_acc = {phase: sum_accuracy[phase]/len(dataloaders[phase]) for phase in ["train", "val"]}#, "test"]}
  
        print("Fase :",phase,"\t | Loss :","{:.4f}".format(epoch_loss[phase]), "\t | Acc :" ,"{:.4f}".format(epoch_acc[phase]))
      
      for phase in ["train", "val"]:#, "test"]:
        history_loss[phase].append(epoch_loss[phase])
        history_accuracy[phase].append(epoch_acc[phase])  
      # Ccontrollo se ho ottenuto la migliore accuratessa 
      if epoch_acc["val"]>best_val_accuracy:
        # aggiorno se è migliore 
        best_val_accuracy=epoch_acc["val"]
        epoch_best_model = epoch
        save_weights(model, optimizer, tag, epoch, epoch_loss["val"], best_val_accuracy, history_loss, history_accuracy, path_weight)
      
      time_elapsed = time.time() - since
      print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
      
      #FILE DI LOG
      log_file = 'checkpoint_{}.txt'.format(tag)
      log = open(log_file, 'a')
      #log.write("VAL DICE: \n " + str(val_dice))
      log.write("Checkpoint 1: \n ")
      log.write("  -----  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
      log.close()

      # Plot loss history
      plt.title("Loss Epoche: "+ str(len(history_loss["train"])))
      for split in ["train", "val"]:#, "test"]:
          plt.plot(history_loss[split], label=split)
      plt.legend()
      plt.savefig('output2D/train/Loss_tag_{}_epoch_{}.png'.format(tag, epoch))
      #plt.show()
      plt.close()

      # Plot accuracy history
      plt.title("Accuracy Epoche: "+ str(len(history_loss["train"])))
      for split in ["train", "val"]:#, "test"]:
          plt.plot(history_accuracy[split], label=split)
      plt.legend()
      plt.savefig('output2D/train/Accuracy_tag_{}_epoch_{}.png'.format(tag, epoch))
      #plt.show()
      plt.close()
  except KeyboardInterrupt:
    print("Interrupted")    
  print("Best val Acc: {:4f}".format(best_val_accuracy))
  
  return model, best_val_accuracy, epoch_best_model, optimizer, history_loss, history_accuracy

def test(model, test_loader, device):
    #inizializzazione parametri
    correct = []
    errate = []
    tot_true_labels = []
    tot_predicted_labels = []
    image_idx =0


    model.eval()
    
    for inputs, targets in test_loader:
            tot_true_labels.append(targets.cpu().detach())
            inputs = inputs.to(device)
            targets = targets.to(device)


            pred = model(inputs)
            #pred=pred.argmax(1) #CROSS_ENTROPY
            pred = torch.round(torch.sigmoid(pred)) #BCE_LOGIT
            #pred = np.round(pred.detach()) #BCE
            
            tot_predicted_labels.append(pred.cpu().detach())
            pred= pred.item() 

            if pred == targets:
                correct.append(pred)
                image_idx += 1
            else:
                errate.append(pred)
                image_idx += 1
            

    return correct, errate, tot_true_labels, tot_predicted_labels
  
def compute_output(model, test_loader, device):
    #inizializzazione parametri
    tot_predicted_labels = []
    all_path = []

    model.eval()
    
    for inputs, data in test_loader:

            inputs = inputs.to(device) 
            idx = data

            pred = model(inputs)
            #pred=pred.argmax(1) #CROSS_ENTROPY
            pred = torch.round(torch.sigmoid(pred)) #BCE_LOGIT
            #pred = np.round(pred.detach()) #BCE
            print(idx, pred.cpu().detach().numpy()) #STAMPA LA SCAN E LA SUA PREDIZIONE
            tot_predicted_labels.append(pred.cpu().detach().numpy())
            pred= pred.item() 

            all_path.append(idx)
    tot_predicted_labels=np.concatenate(tot_predicted_labels)
    all_path = np.concatenate(all_path)

    tot_predicted_labels=tot_predicted_labels.mean(axis=1)
    return tot_predicted_labels, all_path

