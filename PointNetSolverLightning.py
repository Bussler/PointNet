import torch
from torch.utils.data import DataLoader
from path import Path
from Network.PointNetworkLightning import PointNet
from Network.PointCloudDataset import PointCloudData
from Network.DataTransforms import train_transforms
import pytorch_lightning as pl


path = Path("ModelNet10/ModelNet10") # put in Path to dataset root dir here

train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

pointnet = PointNet()

trainer = pl.Trainer(max_epochs=1, gpus = 1 if torch.cuda.is_available() else None)
trainer.fit(pointnet,train_dataloader=train_loader, val_dataloaders=valid_loader)