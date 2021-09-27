import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans as MBK

def nothing(x):
    pass

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


cpdef threshold(img1, img2, T, red):

    # raise an exception if the two images aren't from the same video (assumption that the resolutions are the same)
    if(img1.shape[0]!=img2.shape[0] or img1.shape[1] !=img2.shape[1]):
        raise Exception("The two images provided are not of the same length")

    # height and width
    cdef h = img1.shape[0]
    cdef w = img1.shape[1]

    # I will default the new image to a matrix of 0's (black) and populate with 1's (white) when necessary
    diffpre = np.zeros((h, w, 3), dtype='uint8')
    diff = cv.cvtColor(diffpre,cv.COLOR_BGR2GRAY)

    fin = cv.cvtColor(np.zeros((h, w, 3), dtype='uint8'), cv.COLOR_BGR2GRAY)
    for y in range(0,h):
        for x in range(0,w):
            diff[y,x] = np.uint8(abs(img1[y,x]-img2[y,x]))

    norm = cv.fastNlMeansDenoising(diff,None, red, 21, 7)

    for y in range(0,h):
        for x in range(0,w):
            if (norm[y,x] > T):
                fin[y,x] = np.uint8(255)
            else:
                fin[y,x] = np.uint8(0)

    cv.imshow("output",np.hstack([diff,norm,fin]))
    cv.waitKey(1)

def main():
    cv.namedWindow("output")
    cv.createTrackbar("threshold", "output", 0, 255, nothing)
    cv.createTrackbar("noise", "output", 0, 1000, nothing)

    image1 = cv.imread("../Lab2/images01/park466.bmp")
    image2 = cv.imread("../Lab2/images01/park468.bmp")
    img1Q = quantize(image1, 4)
    img2Q = quantize(image2, 4)
    img1QG = cv.cvtColor(img1Q, cv.COLOR_BGR2GRAY)
    img2QG = cv.cvtColor(img2Q, cv.COLOR_BGR2GRAY)

    while(1):
        t = cv.getTrackbarPos("threshold","output")
        red = cv.getTrackbarPos("noise","output")

        print("Threshold value: "+str(t))
        print("Noise Reduction value: " + str(red))

        threshold(img1QG,img2QG,T=t,red=red)
        cv.waitKey(1)

    cv.destroyAllWindows()

