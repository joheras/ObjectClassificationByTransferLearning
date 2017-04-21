# import the necessary packages

from sklearn_theano.feature_extraction.caffe.googlenet import GoogLeNetTransformer
from sklearn_theano.feature_extraction.overfeat import SMALL_NETWORK_FILTER_SHAPES
from sklearn_theano.feature_extraction import OverfeatTransformer
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
import numpy as np

MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

class Extractor:
	def __init__(self,modelText):
		# store the layer number and initialize the Overfeat transformer
		#self.layerNum = layerNum
		self.modelText=modelText
		print("[INFO] loading {}...".format(modelText))
		if modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
			Network = MODELS[modelText]
			self.model = Network(include_top=False)
		if modelText == "googlenet":
			self.model=GoogLeNetTransformer()
		if modelText == "overfeat":
			self.model = OverfeatTransformer(output_layers=[-3])

	def reshape(self,res):
		if(self.modelText=="resnet"):
			return np.reshape(res,2048)
		else:
			return res.flatten()


	def describe(self, images):
		if self.modelText in ("inception", "xception", "vgg16", "vgg19", "resnet"):
			return [self.reshape(self.model.predict(image)) for image in images]
		if self.modelText in ("googlenet","overfeat"):
			return self.model.transform(images)

