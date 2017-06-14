import PIL
import argparse
import glob
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder where the images are going to be saved")
args = parser.parse_args()


for image in glob.glob(args.folder + "/*/*/*"):
	try:
		print(image)
		im = Image.open(image)
		im = im.rotate(270)
		im.save(image)
	except:
		try:
		    os.remove(image)
		except OSError:
		    pass

