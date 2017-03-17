from bs4 import BeautifulSoup
import urllib2
import os
import glob
import argparse
import PIL
from PIL import Image


def get_soup(url, header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')


parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--width", help="width for the images to resize")
parser.add_argument("--height", help="height for the images to resize")
args = parser.parse_args()
DATA_FILE_NAME = args.dataFolder
WIDTH = args.width
HEIGHT = args.height
# Read each label
with open("labels.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
labels = [x.strip() for x in content]

for label in labels:
    # Creates a directory for each label
    all_names = label.split(',')
    name = all_names[0]

    LABEL_PATH = os.path.join(DATA_FILE_NAME, name)
    for image in glob.glob(LABEL_PATH + "/*"):
        print image
        img = Image.open(image)
        img = img.resize((int(WIDTH), int(HEIGHT)), PIL.Image.ANTIALIAS)
        img.save(image)
