import plotly.graph_objects as go
from path import Path
import numpy as np
from Visualization.PlotlyVisualization import read_off, visualize_rotate, pcshow
from Network.DataTransforms import PointSampler



path = Path("ModelNet10/ModelNet10")

def TestVisualizeMesh():
    with open(path/"bed/train/bed_0001.off", 'r') as f:
        verts, faces = read_off(f)

    i,j,k = np.array(faces).T
    x,y,z = np.array(verts).T

    visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i,j=j,k=k)]).show()

def TestVisualizePointCloud():
    with open(path/"bed/train/bed_0001.off", 'r') as f:
        verts, faces = read_off(f)

    i,j,k = np.array(faces).T
    x,y,z = np.array(verts).T

    pcshow(x,y,z)

def TestVisualizePointCloudTransforms():
    with open(path/"bed/train/bed_0001.off", 'r') as f:
        verts, faces = read_off(f)

    pointcloud = PointSampler(3000)((verts, faces))
    pcshow(*pointcloud.T)


TestVisualizeMesh()
TestVisualizePointCloud()
TestVisualizePointCloudTransforms()