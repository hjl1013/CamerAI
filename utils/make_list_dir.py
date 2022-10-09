import glob, os
from os import getcwd
import random

path = os.getcwd() 


# Create and/or truncate train.txt and test.txt
file_train = open('val.txt', 'a')

list = []
def make_list_cur(root, class_dir):
    cur_dir = root + '/' + class_dir + '/'
    for pathAndFilename in glob.glob(os.path.join(cur_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        s = cur_dir + title + '.jpg' + "\n"
        list.append(s)

def main():

    dir_list = os.listdir(path)
    for root, dirnames, filenames in os.walk(path):
        break  
    for class_dir in dirnames:
        if class_dir == "7_dataset":
            make_list_cur(root, class_dir)

    while list:
        name = random.choice(list)
        print(name)
        file_train.write(name)
        list.remove(name)



if __name__ == '__main__':
    main()
