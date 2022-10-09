import os, shutil, glob
import random
import sys

sys.path.append('home/aistore17/Datasets/1.competition_dataset')
num = 0
for root, subdirs, files in os.walk('coco_dataset/train_both'):
    for d in files:
        if 'jpg' in d:
            dir_to_copy = os.path.join(root, d)
            print(dir_to_copy)
            shutil.copy(dir_to_copy, 'coco_dataset/train')
            print("\rCopying: ", num, end="")
            num = num + 1

for root, subdirs, files in os.walk('coco_dataset/test_both'):
    for d in files:
        if 'jpg' in d:
            dir_to_copy = os.path.join(root, d)
            print(dir_to_copy)
            shutil.copy(dir_to_copy, 'coco_dataset/test')
            print("\rCopying: ", num, end="")
            num = num + 1

for root, subdirs, files in os.walk('coco_dataset/val_both'):
    for d in files:
        if 'jpg' in d:
            dir_to_copy = os.path.join(root, d)
            print(dir_to_copy)
            shutil.copy(dir_to_copy, 'coco_dataset/val')
            print("\rCopying: ", num, end="")
            num = num + 1
