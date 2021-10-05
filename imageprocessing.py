import cv2 as cv
import numpy
from sklearn.cluster import MiniBatchKMeans as MBK

def quantize(image,k):
    img = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    # reshaping to a feature vector (2d matrix) into a single vector
    imgVector = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = MBK(n_clusters=k)
    labels = clt.fit_predict(imgVector)
    quants = clt.cluster_centers_.astype("uint8")[labels]

    # reshape it back to the original 2d matrix (picture)
    quantsReshaped = quants.reshape((img.shape[0], img.shape[1], 3))

    # convert it back from l*a*b* to RGB
    quantsRecolor = cv.cvtColor(quantsReshaped, cv.COLOR_LAB2BGR)

    return quantsRecolor


