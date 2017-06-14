import argparse
import glob
import os
import random
import uuid

import cv2

# python getFrames.py --dataFolder data --videoFolder videoinigo --className inigo  --framerRate 5

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the videos are saved")
parser.add_argument("--videoFolder", help="folder where the videos are saved")
parser.add_argument("--className", help="folder where the train and test folders are going to be created")
parser.add_argument("--framerRate", help="frame rate to pick the frame or to drop")
args = parser.parse_args()

percentage_train = 0.75

# Check if folders exist
if not os.path.exists(args.dataFolder):
    os.makedirs(args.dataFolder)

if not os.path.exists(args.dataFolder + "/test"):
    os.makedirs(args.dataFolder + "/test")

if not os.path.exists(args.dataFolder + "/train"):
    os.makedirs(args.dataFolder + "/train")

if not os.path.exists(args.dataFolder + "/test/" + args.className):
    os.makedirs(args.dataFolder + "/test/" + args.className)

if not os.path.exists(args.dataFolder + "/train/" + args.className):
    os.makedirs(args.dataFolder + "/train/" + args.className)

count = 0
# For each video
for video in glob.glob(args.videoFolder + "/*.mp4"):
    print(video)

    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    success = True
    while success:
        success, image = vidcap.read()
        # If it should be saved
        if count % int(args.framerRate) == 0:
            # train or test
            if random.uniform(0, 1) <= percentage_train:
                cv2.imwrite(args.dataFolder + "/train/" + args.className + "/" + str(uuid.uuid4()) + ".jpg",
                            image)  # save frame as JPEG file

            else:
                cv2.imwrite(args.dataFolder + "/test/" + args.className + "/" + str(uuid.uuid4()) + ".jpg",
                            image)  # save frame as JPEG file

        count += 1
