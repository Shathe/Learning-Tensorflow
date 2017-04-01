import os
import glob
import argparse
import PIL
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--width", help="width for the images to resize")
parser.add_argument("--height", help="height for the images to resize")
args = parser.parse_args()
DATA_FILE_NAME = args.dataFolder
WIDTH = args.width
DATA_FILE_NAME_TEST = DATA_FILE_NAME + "/test"
DATA_FILE_NAME_TRAIN = DATA_FILE_NAME + "/train"
HEIGHT = args.height
# Read each label
with open("labels.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
labels = [x.strip() for x in content]

for label in labels:
    # Creates a directory for each label
    all_names = label.split(',')
    name = all_names[0].replace(' ', '_')
    print(name)

    LABEL_PATH = os.path.join(DATA_FILE_NAME_TRAIN, name)
    for image in glob.glob(LABEL_PATH + "/*"):
        print(image)
        try:
            img = Image.open(image)
            img = img.resize((int(WIDTH), int(HEIGHT)), PIL.Image.ANTIALIAS)
            img.save(image)
        except:
            os.remove(image)

    LABEL_PATH = os.path.join(DATA_FILE_NAME_TEST, name)
    for image in glob.glob(LABEL_PATH + "/*"):
        print(image)
        try:
            img = Image.open(image)
            img = img.resize((int(WIDTH), int(HEIGHT)), PIL.Image.ANTIALIAS)
            img.save(image)
        except:
            os.remove(image)
