# import the necessary packages
from __future__ import print_function
from utils.conf import Conf
from extractor.extractor import Extractor
import numpy as np
import argparse
from utils import dataset

import cPickle

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to predict")
args = vars(ap.parse_args())

# load the configuration, label encoder, and classifier
print("[INFO] loading model...")
conf = Conf(args["conf"])
labelEncoderPath = conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
le = cPickle.loads(open(labelEncoderPath).read())
model = cPickle.loads(open(conf["classifier_path"]+ conf["modelClassifier"] + ".cpickle").read())

imagePath = args["image"]



oe = Extractor(conf["model"])
(labels, images) = dataset.build_batch([imagePath], conf["model"])
features = oe.describe(images)
for (label, vector) in zip(labels, features):
    prediction = model.predict(np.atleast_2d(vector))[0]
    print(prediction)
    prediction = le.inverse_transform(prediction)
    print("[INFO] predicted: {}".format(prediction))