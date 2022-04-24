import os
import torch
from torch.utils.data import DataLoader
from path import Path
from Network.PointNetworkLightning import PointNet
from Network.PointCloudDataset import PointCloudData
from Network.DataTransforms import train_transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Visualization.ConfusionMatrixVisualization import plot_confusion_matrix



path = Path("ModelNet10/ModelNet10") # put in Path to dataset root dir here

# load data
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

# load checkpoint
checkpoint = "./lightning_logs/version_6/checkpoints/epoch=0.ckpt"
pointnet = PointNet.load_from_checkpoint(checkpoint)
pointnet.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for i, data in enumerate(valid_loader):
        print('Batch [%4d / %4d]' % (i+1, len(valid_loader)))
                   
        inputs, labels = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.numpy())
        all_labels += list(labels.numpy())


cm = confusion_matrix(all_labels, all_preds)
print(cm)

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)}

plt.figure(figsize=(8,8))
plot_confusion_matrix(cm, list(classes.keys()), normalize=True)