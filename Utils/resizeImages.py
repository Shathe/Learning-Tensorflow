import PIL
import argparse
import glob
import os
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
labels = [x.strip() for x in content]


def resize(image):
    print(image)
    try:
        img = Image.open(image)
        img = img.resize((int(WIDTH), int(HEIGHT)), PIL.Image.ANTIALIAS)
        img.save(image)
    except:
        os.remove(image)


for label in labels:
    all_names = label.split(',')
    name = all_names[0].replace(' ', '_')
    print(name)
    # Each label/folder

    label_path = os.path.join(DATA_FILE_NAME_TRAIN, name)
    for image in glob.glob(label_path + "/*"):
        # Each train image
        resize(image)

        label_path = os.path.join(DATA_FILE_NAME_TEST, name)
    for image in glob.glob(label_path + "/*"):
        # Each test image
        resize(image)
