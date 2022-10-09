import os, shutil, glob
import random
import sys

sys.path.append('home/aistore17/Datasets/1.competition_dataset')
dirs = ['1_dataset/','2_dataset/','3_dataset/','4_dataset/','5_dataset/','6_dataset/','7_dataset/']
num = 0
for cur_dir in dirs:

    for pathAndFilename in glob.glob(os.path.join(cur_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        pic = cur_dir + title + '.jpg'
        txt = cur_dir + title + '.txt'

        # dir_to_copy = os.path.join(root, d)
        # print(dir_to_copy)

        x = random.random()
        if (x < 0.8):
            shutil.copy(pic, 'coco_dataset/train_both')
            shutil.copy(txt, 'coco_dataset/train_both')
        elif (x < 0.9):
            shutil.copy(pic, 'coco_dataset/test_both')
            shutil.copy(txt, 'coco_dataset/test_both')
        else:
            shutil.copy(pic, 'coco_dataset/val_both')
            shutil.copy(txt, 'coco_dataset/val_both')

        num = num+1
        print("\rCopying: ", num, end="")

    # for root, subdirs, files in os.walk('1_dataset'):
    #     for d in files:
    #         if 'jpg' in d:


