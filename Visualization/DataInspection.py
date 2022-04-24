import os
from path import Path

path = Path("ModelNet10/ModelNet10")

def inspectDataSize(path):
    print("Data at: ",path)
    
    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    print("Classes: \n", classes)

# inspectDataSize(path)