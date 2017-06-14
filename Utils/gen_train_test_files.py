import cv2
import argparse
import os
import glob
import random

# python gen_train_test_files.py --dataFolder data 

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the videos are saved")
args = parser.parse_args()

train_file = 'train.txt'
test_file = 'test.txt'
try:
    os.remove(train_file)
    os.remove(test_file)
except OSError:
    pass

f_train = open(train_file, 'w')
f_test = open(test_file, 'w')

# Read image names and write them in the file with its label
folder_label = 0
for folder in glob.glob(args.dataFolder + "/train/*"):
	for image in glob.glob(folder + "/*"):
		f_train.write(image + ' ' + str(folder_label) + '\n')  # python will convert \n to os.linesep
	folder_label +=1


folder_label = 0
for folder in glob.glob(args.dataFolder + "/test/*"):
	for image in glob.glob(folder + "/*"):
		f_test.write(image + ' ' + str(folder_label) + '\n')  # python will convert \n to os.linesep
	folder_label +=1

f_train.close()
f_test.close()