import argparse
import json
import os
import random
import sys
import urllib2
from bs4 import BeautifulSoup


def get_soup(url, header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')


parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
args = parser.parse_args()
DATA_FILE_NAME = args.dataFolder
TRAIN_PATH = DATA_FILE_NAME + "/train"
TEST_PATH = DATA_FILE_NAME + "/test"
percentage_train = 0.75


# Delete (if exists) and create the test.txt and train.txt files

train_file = 'train.txt'
test_file = 'test.txt'
try:
    os.remove(train_file)
    os.remove(test_file)
except OSError:
    pass

f_train = open(train_file, 'w')
f_test = open(test_file, 'w')


# Read each label
with open("labels.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
labels = [x.strip() for x in content]

# Creates the directories where the data is going to be saved
if not os.path.exists(DATA_FILE_NAME):
    os.makedirs(DATA_FILE_NAME)

if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)


label_int = 0
for label in labels:
    # Creates a directory for each label
    print(label)
    all_names = label.split(',')
    name = all_names[0].replace(' ', '_')  # Gests the name of the label

    LABEL_PATH_TRAIN = os.path.join(TRAIN_PATH, name)
    LABEL_PATH_TEST = os.path.join(TEST_PATH, name)
    if not os.path.exists(LABEL_PATH_TRAIN):
        os.makedirs(LABEL_PATH_TRAIN)
    if not os.path.exists(LABEL_PATH_TEST):
        os.makedirs(LABEL_PATH_TEST)

    cntr = 0
    err = 0
    # For each label, several queries
    for name_i in all_names:

        # Prepare query
        query = name_i.split()
        query = '+'.join(query)
        url = "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"
        print(url)
        # add the directory for your image here
        header = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
        }
        # Query
        soup = get_soup(url, header)

        # Get the data
        ActualImages = []  # contains the link for Large original images, type of  image
        for a in soup.find_all("div", {"class": "rg_meta"}):
            link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            ActualImages.append((link, Type))

        print("there are total", len(ActualImages), "images")

        # Save each image
        for i, (img, Type) in enumerate(ActualImages):
            try:
                req = urllib2.Request(img, headers={'User-Agent': header})
                raw_img = urllib2.urlopen(req).read()
                # percentage_train out of 1 will be save in the training folder
                PATH_TO_SAVE = LABEL_PATH_TRAIN
                f_writer = f_train
                if random.uniform(0, 1) > percentage_train:
                    PATH_TO_SAVE = LABEL_PATH_TEST
                    f_writer = f_test

                if cntr % 20 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                download = False
                # Only save jpg or png or jpeg
                nameFile = os.path.join(PATH_TO_SAVE, name_i.replace(' ', '_') + "_" + str(cntr) + ".jpg")

                if len(Type) == 0:
                    if "jpg" in raw_img and "body" not in raw_img:
                        f = open(nameFile, 'wb')
                        download = True
                elif "body" not in raw_img and "jpg" in Type:
                    f = open(nameFile, 'wb')
                    download = True
                if download:
                    cntr += 1
                    f.write(raw_img)
                    f_writer.write(nameFile + ' ' + str(label_int) + '\n')  # python will convert \n to os.linesep
                    f.close()
            except Exception as e:
                err += 1
    label_int += 1
    print("for label " + name + " images well downloaded: " + str(cntr) + ", images not downloaded: " + str(err))

f_train.close()
f_test.close()
