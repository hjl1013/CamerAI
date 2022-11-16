import glob, os
from os import getcwd
import random
# import sys
# sys.path.append()

# Create and/or truncate train.txt and test.txt
file_train = open('/home/aistore17/Datasets/cocoformat_Dataset/train.txt', 'a')
file_val = open('/home/aistore17/Datasets/cocoformat_Dataset/val.txt', 'a')
file_test = open('/home/aistore17/Datasets/cocoformat_Dataset/test.txt', 'a')

ratio = [0.9, 0.1, 0]

list = []


def make_list_cur(class_dir):
    cur_dir = class_dir + '/'
    for pathAndFilename in glob.glob(os.path.join(cur_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        s = cur_dir + title + '.jpg' + "\n"
        list.append(s)


def main():
    data_dir_paths = [
        '/home/aistore17/Datasets/1.competition_dataset/1_dataset',
        '/home/aistore17/Datasets/1.competition_dataset/2_dataset',
        '/home/aistore17/Datasets/1.competition_dataset/3_dataset',
        '/home/aistore17/Datasets/1.competition_dataset/4_dataset',
        '/home/aistore17/Datasets/1.competition_dataset/5_dataset',
        '/home/aistore17/Datasets/1.competition_dataset/6_dataset',
        '/home/aistore17/Datasets/1.competition_dataset/7_dataset',
        '/home/aistore17/Datasets/NewDataset',
        # '/home/aistore17/Datasets/RandomBackgroundDataset'
    ]

    for data_dir_path in data_dir_paths:
        make_list_cur(data_dir_path)

    total_data_num = len(list)
    cnt = 0
    while list:
        name = random.choice(list)
        print(name)
        if cnt < total_data_num * ratio[0]:
            file_train.write(name)
        elif cnt < total_data_num * (ratio[0] + ratio[1]):
            file_val.write(name)
        else:
            file_test.write(name)
        list.remove(name)
        cnt += 1


if __name__ == '__main__':
    main()
