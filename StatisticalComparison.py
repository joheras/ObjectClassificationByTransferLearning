#=======================================================================================================================
# Multilayer perceptron is given in a separate file since it is not available in the python version employed in
# the other files.
#=======================================================================================================================


from sklearn.cross_validation import train_test_split
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from Comparing import compare_methods, compare_methods_h5py
import argparse
from utils.conf import Conf
from StatisticalAnalysis.statisticalAnalysis import statisticalAnalysis
import h5py
import numpy as np
import cPickle

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())
conf = Conf(args["conf"])
featuresPath = conf["features_path"][0:conf["features_path"].rfind(".")] + "-"+ conf["model"] +".hdf5"
#db = h5py.File(featuresPath)
#labels = db["image_ids"]
labelEncoderPath = conf["label_encoder_path"][0:conf["label_encoder_path"].rfind(".")] + "-"+ conf["model"] +".cpickle"
#le = cPickle.loads(open(labelEncoderPath).read())
#labels = [le.transform([l.split(":")[0]])[0] for l in labels]
#df1 = pd.DataFrame([np.append(x,y) for (x,y) in zip(db["features"],labels)])
featuresCSVPath = conf["features_csv_path"][0:conf["features_csv_path"].rfind(".")] + "-"+ conf["model"] +".csv"

#df1.to_csv(featuresCSVPath)
# Loading dataset
dataset = featuresCSVPath
#df = pd.read_csv(featuresCSVPath)
#data = df.ix[:, :-1].values


#================================================================================================================
print("RandomForest")
#================================================================================================================
clfRF = RandomForestClassifier(random_state=84,n_estimators=20)

# specify parameters and distributions to sample from
param_distRF = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


#================================================================================================================
print("SVM")
#================================================================================================================

clfSVC = SVC(random_state=84)
# specify parameters and distributions to sample from
param_distSVC = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
  'kernel': ['rbf'], 'class_weight':['balanced', None]}


#================================================================================================================
print("KNN")
#================================================================================================================

param_distKNN = {'n_neighbors':sp_randint(3, 30)}
clfKNN = KNeighborsClassifier()


#================================================================================================================
print("Logistic Regression")
#================================================================================================================

clfLR = LogisticRegression(random_state=84)
param_distLR = {'C': [0.1,0.5,1, 10, 100, 1000]}


#================================================================================================================
print("MultiLayer Perceptron")
#================================================================================================================


clfMLP = MLPClassifier(random_state=84)

param_distMLP = {'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver':['lbfgs','sgd','adam'],
              'alpha': sp_randint(0.0001, 1),'learning_rate':['constant','invscaling','adaptive'],'momentum':[0.9,0.95,0.99]}


#================================================================================================================
print("Gradient Boost")
#================================================================================================================
clfGB = GradientBoostingClassifier(random_state=84,n_estimators=20)

# specify parameters and distributions to sample from
param_distGB = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "criterion": ["friedman_mse", "mse", "mae"]}



listAlgorithms = [clfRF,clfSVC,clfKNN,clfLR,clfMLP]
listParams = [param_distRF,param_distSVC,param_distKNN,param_distLR,param_distMLP]
listNames = ["RF", "SVM", "KNN", "LR", "MLP"]

#results = compare_methods(dataset,listAlgorithms,listParams,listNames,[20,10,10,5,10],normalization=False)
results = compare_methods_h5py(featuresPath,labelEncoderPath,listAlgorithms,listParams,listNames,[20,10,10,5,10],normalization=False)

df = pd.DataFrame.from_dict(results,orient='index')
KFoldComparisionPath = conf["kfold_comparison"][0:conf["kfold_comparison"].rfind(".")] + "-"+ conf["model"] +".csv"
df.to_csv(KFoldComparisionPath)

statisticalAnalysis(KFoldComparisionPath)












