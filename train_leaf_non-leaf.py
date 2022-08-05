import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import torchmetrics
from sklearn.metrics import classification_report
import wandb

from dataset import PlantDataset, LeafDataset
from utils import get_transform, AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--weights", help="Path to the weights file for loading the model.", default=None, required=False)
    arg("--train_csv", help="Path to train csv.", required=True)
    arg("--val_csv", help="Path validation csv.", required=True)
    arg("--batch_size", type=int, help="batch size.", default=16, required=False)
    arg("--num_workers", type=int, help="number of threads workers.", default=16, required=False)
    arg("--num_epochs", type=int, help="number of epochs.", default=30, required=False)
    arg("--save_dir", help="save dir path.", required=True)
    arg("--model_name", help="classification model name.", required=True)
    arg("--lr", type=float, help="learning rate.", default=0.0001, required=False)
    arg("--optimizer", help="optimizer of your choice.", required=True)
    arg("--weight_decay", type=float, help="weight decay for optimizer.", default=0.01, required=False)
    arg("--momentum", type=float, help="momentum for optimizer.", default=0.9, required=False)
    return parser.parse_args()


def validate_epoch(dataloader, model, criterion, optimizer, epoch):
    m = nn.Sigmoid()
    acc = AverageMeter()
    losses = AverageMeter()
    all_pred = []
    all_target = []
    model.eval()
    with torch.no_grad():
      for i, (img, target) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Val"):
          
          img = img.float()
          img = img.to(device)
          target = target.to(device)
          
          optimizer.zero_grad() 
          
          output = model(img) 
          
          loss = criterion(output,target)
          losses.update(loss.item(), img.size(0))
                  
          pred = m(output)
                  
          accuracy = torchmetrics.functional.accuracy(pred,  target.int())
          acc.update(accuracy, img.size(0))          

          target = np.array(target.tolist()).astype(int).tolist()
          pred = (np.array(pred.tolist()) > 0.5).astype(int).tolist()
          
          all_target += target
          all_pred += pred

          torch.cuda.empty_cache()

    cr = classification_report(all_target, all_pred, output_dict=True)
    wandb.log({"val loss": losses.avg, "val acc": acc.avg*100,
               "val precison": cr['macro avg']['precision']*100, "val recall": cr['macro avg']['recall']*100,
               "val f1": cr['macro avg']['f1-score']*100}, step = epoch)
    print(classification_report(all_target, all_pred))
    return cr['macro avg']['f1-score']*100


def train_epoch(dataloader, model, criterion, optimizer, epoch):
    m = nn.Sigmoid()
    accuracy = AverageMeter()
    losses = AverageMeter()
    model.train()
    all_pred = []
    all_target = []
    for i, (img, target) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train"):
        img = img.float()  

        img = img.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = m(output)
        
        acc = torchmetrics.functional.accuracy(pred, target.int())

        losses.update(loss.item(), img.size(0))
        accuracy.update(acc, img.size(0))

        target = np.array(target.tolist()).astype(int).tolist()
        pred = (np.array(pred.tolist()) > 0.5).astype(int).tolist()

        all_target += target
        all_pred += pred
        
        torch.cuda.empty_cache()

    cr = classification_report(all_target, all_pred, output_dict=True)

    wandb.log({"train loss": losses.avg, "train acc": accuracy.avg*100,
               "train precison": cr['macro avg']['precision']*100, "train recall": cr['macro avg']['recall']*100,
               "train f1": cr['macro avg']['f1-score']*100}, step = epoch)


def train(loader_train, loader_val, EPOCHS, model, opt, criterion, model_path, model_name):
    best_acc = 0
    for e in range(EPOCHS):
        print("------------------------epoch " + str(e) + "-----------------------------")
        train_epoch(loader_train, model, criterion, opt, e)
        current_acc = validate_epoch(loader_val, model, criterion, opt, e)
        if e%30 == 0:
            torch.save(model, model_path + '/' + model_name + "_epoch" + str(e) + "_f1=" + str(best_acc))


def create_model(model_name, weights):
  # if(weights):
  model = models.resnet101()
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, 3)
  model.load_state_dict(torch.load("/gdrive/MyDrive/LeafGANDataSet/LeafGAN/LFLSeg/trained_models/trained_resnet101_LFLSeg_v1_90.pth"), strict=True)
  in_features = model.fc.in_features
  model.fc = torch.nn.Linear(in_features, 1)
  # else:
  #   if model_name == 'vit_b_16':
  #     model = models.vit_b_16(pretrained=True)
  #     in_features = model.heads.head.in_features
  #     model.heads.head = torch.nn.Linear(in_features, 7)
  #   elif model_name == 'resnet152':
  #     model = models.resnet152(pretrained = True)
  #     in_features = model.fc.in_features
  #     model.fc = torch.nn.Linear(in_features, 7)
  return model


def main(weights, train_csv_path, val_csv_path, batch_size, num_workers, save_dir, num_epochs, model_name, lr, optimizer, weight_decay, momentum):
    model = create_model(model_name, weights)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    if optimizer.lower() == 'adamw':
      opt = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer.lower() == 'sgd':
      opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    run_name = model_name + "_batch=" + str(batch_size) + "_lr=" + str(lr)
    project_name = "Final_Project"
    entity_name = "seminar"
    wab = wandb.init(project=project_name, entity=entity_name, name=run_name)

    df_train = pd.read_csv(train_csv_path)
    df_valid = pd.read_csv(val_csv_path)

    train_dataset = LeafDataset(df_train, get_transform('train'))
    valid_dataset = LeafDataset(df_valid, get_transform('valid'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    

    train(train_loader, valid_loader, num_epochs, model, opt, criterion, save_dir, model_name)


if __name__ == "__main__":
    args = get_args()
    weights = args.weights
    train_csv_path = args.train_csv
    val_csv_path = args.val_csv
    batch_size = args.batch_size
    num_workers = args.num_workers
    save_dir = args.save_dir
    num_epochs = args.num_epochs
    model_name = args.model_name
    lr = args.lr
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    main(weights, train_csv_path, val_csv_path, batch_size, num_workers, save_dir, num_epochs, model_name, lr, optimizer, weight_decay, momentum)
