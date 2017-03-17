from bs4 import BeautifulSoup
import urllib2
import os
import json
import argparse
import random


def get_soup(url, header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')


parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
args = parser.parse_args()
DATA_FILE_NAME = args.dataFolder
TRAIN_PATH = DATA_FILE_NAME + "/train"
TEST_PATH = DATA_FILE_NAME + "/test"
percentage_train = 0.8

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

for label in labels:
    # Creates a directory for each label
    print label
    all_names = label.split(',')
    name = all_names[0]  # Gests the name of the label

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
        print url
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

        print "there are total", len(ActualImages), "images"

        # Save each image
        for i, (img, Type) in enumerate(ActualImages):
            try:
                req = urllib2.Request(img, headers={'User-Agent': header})
                raw_img = urllib2.urlopen(req).read()
                # percentage_train out of 1 will be save in the training folder
                PATH_TO_SAVE = LABEL_PATH_TRAIN
                if random.uniform(0, 1) > percentage_train:
                    PATH_TO_SAVE = LABEL_PATH_TEST

                if cntr % 20 == 0:
                    print cntr

                download = False
                # Only save jpg or png or jpeg
                if len(Type) == 0:
                    if "jpeg" in raw_img and "body" not in raw_img:
                        f = open(os.path.join(PATH_TO_SAVE, name_i + "_" + str(cntr) + ".jpeg" + Type), 'wb')
                        download = True
                    elif "png" in raw_img and "body" not in raw_img:
                        f = open(os.path.join(PATH_TO_SAVE, name_i + "_" + str(cntr) + ".png" + Type), 'wb')
                        download = True
                    elif "jpg" in raw_img and "body" not in raw_img:
                        f = open(os.path.join(PATH_TO_SAVE, name_i + "_" + str(cntr) + ".jpg" + Type), 'wb')
                        download = True
                elif "body" not in raw_img and ("jpg" in Type or "png" in Type or "jpeg" in Type):
                    f = open(os.path.join(PATH_TO_SAVE, name_i + "_" + str(cntr) + "." + Type), 'wb')
                    download = True
                if download:
                    cntr += 1
                    f.write(raw_img)
                    f.close()
            except Exception as e:
                err += 1

    print ("for label" + name + " images well downloaded: " + cntr + ", images not downloaded: " + err)
