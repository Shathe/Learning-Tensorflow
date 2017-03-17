from bs4 import BeautifulSoup
import urllib2
import os
import json
import argparse


def get_soup(url, header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url, headers=header)), 'html.parser')
'''
parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
args = parser.parse_args()
DATA_FILE_NAME = args.dataFolder
'''
DATA_FILE_NAME = "data"
# Read each label
with open("labels.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
labels = [x.strip() for x in content]

# Creates the directory where the data is going to be saved
if not os.path.exists(DATA_FILE_NAME):
    os.makedirs(DATA_FILE_NAME)

print labels

for label in labels:
    # Creates a directory for each label
    all_names = label.split(',')
    name = all_names[0]

    LABEL_PATH = os.path.join(DATA_FILE_NAME, name)
    if not os.path.exists(LABEL_PATH):
        os.makedirs(LABEL_PATH)

    for name_i in all_names:
        query = name_i.split()
        query = '+'.join(query)
        url = "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"
        print url
        # add the directory for your image here
        header = {
            'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
            }
        soup = get_soup(url, header)

        ActualImages = []  # contains the link for Large original images, type of  image
        for a in soup.find_all("div", {"class": "rg_meta"}):
            link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
            ActualImages.append((link, Type))

        print  "there are total", len(ActualImages), "images"

        for i, (img, Type) in enumerate(ActualImages):
            try:
                req = urllib2.Request(img, headers={'User-Agent': header})
                raw_img = urllib2.urlopen(req).read()

                cntr = len([i for i in os.listdir(LABEL_PATH) if name_i in i]) + 1
                if cntr % 20 == 0:
                    print cntr
                if len(Type) == 0:
                    if "jpeg" in raw_img and "body" not in raw_img:
                        f = open(os.path.join(LABEL_PATH, name_i + "_" + str(cntr) + ".jpeg" + Type), 'wb')
                    elif "png" in raw_img and "body" not in raw_img:
                        f = open(os.path.join(LABEL_PATH, name_i + "_" + str(cntr) + ".png" + Type), 'wb')
                elif "body" not in raw_img and ("jpg" in Type or "png" in Type):
                    f = open(os.path.join(LABEL_PATH, name_i + "_" + str(cntr) + "." + Type), 'wb')

                f.write(raw_img)
                f.close()
            except Exception as e:
                print "could not load : " + img
