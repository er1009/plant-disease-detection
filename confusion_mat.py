import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import ttach as tta

from utils import get_transform
from dataset import PlantDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--tta", help="use tta.", action="store_true")
    return parser.parse_args()

def main():
  args = get_args()
  tta_ = args.tta
  tta_transforms = tta.Compose(
                  [
                      tta.HorizontalFlip(),
                      tta.VerticalFlip(),
                      tta.Rotate90(angles=[0, 90,180]),
                  ]
    ) 
  m = nn.Sigmoid()
  y_pred = []
  y_true = []

  df_valid = pd.read_csv("/gdrive/MyDrive/Final_Project/repo/val.csv")
  valid_dataset = PlantDataset(df_valid, get_transform('valid'))
  valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

  model = torch.load("/gdrive/MyDrive/Final_Project/repo/checkpoints/vit_b_16_epoch76_f1=77.41000066794881")
  if tta_:
    model_tta = tta.ClassificationTTAWrapper(model, tta_transforms)
  model.eval()
  # iterate over test data
  for inputs, labels, target_num in tqdm(valid_loader, total=len(valid_loader), desc="Conf Mat"):
          if tta_:
            output = model_tta(inputs.to(device))
          else:
            output = model(inputs.to(device)) # Feed Network
          output = m(output)
          output = (torch.max(output, 1)[1]).data.cpu().numpy()
          y_pred.extend(output) # Save Prediction
          
          labels = (torch.max(labels, 1)[1]).data.cpu().numpy()
          y_true.extend(labels) # Save Truth

  # constant for classes
  classes =['Septoria',
          'Powdery Mildew',
          'Healthy',
          'Tobacco Mosiac Virus',
          'Spider Mites',
          'Calcium Deficiency' ,
          'Magnesium Deficiency' ]

  # Build confusion matrix
  cf_matrix = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                      columns = [i for i in classes])
  plt.figure(figsize = (20,10))
  sn.heatmap(df_cm, annot=True)
  plt.savefig('/gdrive/MyDrive/Final_Project/output2.png')


if __name__ == "__main__":
    main()