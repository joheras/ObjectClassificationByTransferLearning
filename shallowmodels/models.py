import cv2
import mahotas
from skimage import exposure
from skimage import feature
from imutils import auto_canny
import numpy as np
from scipy.spatial import distance

class LABModel:
    def __init__(self,bins=[8,8,8],channels=[0,1,2],histValues=[0,256,0,256,0,256]):
        self.bins =bins
        self.channels=channels
        self.histValues = histValues

    def describe(self,image):
        checkLAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        histLAB = cv2.calcHist([checkLAB], self.channels, None, self.bins, self.histValues)
        histLAB = cv2.normalize(histLAB).flatten()
        return histLAB


class HSVModel:
    def __init__(self,bins=[8,8,8],channels=[0,1,2],histValues=[0,180,0,256,0,256]):
        self.bins =bins
        self.channels=channels
        self.histValues = histValues

    def describe(self,image):
        checkHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        histHSV = cv2.calcHist([checkHSV], self.channels, None, self.bins, self.histValues)
        histHSV = cv2.normalize(histHSV).flatten()
        return histHSV


class Haralick:
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = mahotas.features.haralick(gray).mean(axis=0)
        return features

class LBP:
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = mahotas.features.lbp(gray, 3, 24)
        return features

class HOG:
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True)
        return features


class HaarHOG:
    def __init__(self):
        pass

    def describe(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        featuresHOG = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), transform_sqrt=True)
        featuresHaar = mahotas.features.haralick(gray).mean(axis=0)
        return np.append(featuresHOG,featuresHaar)

class HistogramsSeveralMasksAnnulusLabSegments:

    def __init__(self,plainImagePath,bags=[8,8,8],channels=[0,1,2],histValues=[0,256,0,256,0,256]):
        self.plainImagePath = plainImagePath
        self.bags = bags
        self.channels = channels
        self.histValues=histValues

    def describe(self,image):
        (h,w) = image.shape[:2]
        control = image[0:h,0:w/2]
        control = cv2.resize(control, (100, 100))
        plain = cv2.imread(self.plainImagePath)
        plain = cv2.resize(plain, (100, 100))
        check = image[0:h,w/2:w]
        check = cv2.resize(check, (100, 100))
        combinations = [(control * float(n) / 100 + plain * float(100 - n) / 100).astype("uint8") for n in
                        range(1, 101, 1)]
        combinationPercentage = [((100 - n)) for n in range(1, 101, 1)]


        # Mask to only keep the centre
        mask = np.zeros(control.shape[:2], dtype="uint8")

        (h, w) = control.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        masks = [mask.copy() for i in range(0, 32)]
        # Generating the different annulus masks
        for i in range(0,32):
            cv2.circle(masks[i], (cX, cY), min(90 - 10 * (i % 8), control.shape[1]) / 2, 255, -1)
            cv2.circle(masks[i], (cX, cY), min(80 - 10 * (i % 8), control.shape[1]) / 2, 0, -1)
        # Keeping only segments of the annulus
        for i in range(0, 8):
            cv2.rectangle(masks[i], (cX, 0), (cX * 2, cY * 2), 0, -1)
            cv2.rectangle(masks[i], (0, 0), (cX * 2, cY), 0, -1)
        for i in range(8, 16):
            cv2.rectangle(masks[i], (cX, 0), (cX * 2, cY * 2), 0, -1)
            cv2.rectangle(masks[i], (0, cY), (cX * 2, cY * 2), 0, -1)
        for i in range(16, 24):
            cv2.rectangle(masks[i], (0, 0), (cX, cY * 2), 0, -1)
            cv2.rectangle(masks[i], (0, 0), (cX * 2, cY), 0, -1)
        for i in range(24, 32):
            cv2.rectangle(masks[i], (0, 0), (cX, cY * 2), 0, -1)
            cv2.rectangle(masks[i], (0, cY), (cX * 2, cY * 2), 0, -1)
        results = []


        for mask in masks:

            checkLAB = cv2.cvtColor(check, cv2.COLOR_RGB2LAB)

            histLAB = cv2.calcHist([checkLAB], self.channels, mask, self.bags, self.histValues)
            histLAB = cv2.normalize(histLAB).flatten()
            histsLAB = [cv2.normalize(
                cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_RGB2LAB)],
                             self.channels, mask, self.bags, self.histValues)).flatten() for im in combinations]
            # Compare histograms
            comparisonLABeuclidean = [distance.euclidean(histLAB, histLAB2) for histLAB2 in histsLAB]
            mins = np.where(np.asarray(comparisonLABeuclidean) == np.asarray(comparisonLABeuclidean).min())
            results.append([[combinationPercentage[n], comparisonLABeuclidean[n]] for n in mins[0].tolist()])

        percentageNew = []
        for p in results:
            if p[0][0] > 60:
                percentageNew.append(p[np.argmax(p, axis=0)[0]])
            else:
                percentageNew.append(p[np.argmin(p, axis=0)[0]])

        percentage = [p[0] for p in percentageNew]
        return (percentage)