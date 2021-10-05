import cv2 as cv
import numpy as np
from imageprocessing import quantize

def threshold(img1, img2, T, red):

    # raise an exception if the two images aren't from the same video (assumption that the resolutions are the same)
    if(img1.shape[0]!=img2.shape[0] or img1.shape[1] !=img2.shape[1]):
        raise Exception("The two images provided are not of the same length")

    # height and width
    h = img1.shape[0]
    w = img1.shape[1]

    # I will default the new image to a matrix of 0's (black) and populate with 1's (white) when necessary
    diffpre = np.zeros((h, w, 3), dtype='uint8')
    diff = cv.cvtColor(diffpre,cv.COLOR_BGR2GRAY)

    fin = cv.cvtColor(np.zeros((h, w, 3), dtype='uint8'), cv.COLOR_BGR2GRAY)
    for y in range(0,h):
        for x in range(0,w):
            diff[y,x] = np.uint8(abs(int(img1[y,x])-int(img2[y,x])))

    # Use this to try and remove noise from the image
    norm = cv.fastNlMeansDenoising(diff,None, red, 21, 7)

    for y in range(0,h):
        for x in range(0,w):
            if (norm[y,x] > T):
                fin[y,x] = np.uint8(255)
            else:
                fin[y,x] = np.uint8(0)

    #show images side by side
    return fin


def diff(image1,image2,t=30,red=400,k=4):
    img1Q = quantize(image1, k)
    img2Q = quantize(image2, k)
    img1QG = cv.cvtColor(img1Q, cv.COLOR_BGR2GRAY)
    img2QG = cv.cvtColor(img2Q, cv.COLOR_BGR2GRAY)
    return threshold(img1QG,img2QG,T=t,red=red)

# if __name__ == "__main__":
#     cv.namedWindow("output")
#     cv.createTrackbar("threshold", "output", 0, 255, nothing)
#     cv.createTrackbar("noise", "output", 0, 1000, nothing)
#
#     # I just switched back and forth between the park and the car here
#     image1 = cv.imread("../Lab2/images01/park466.bmp")
#     image2 = cv.imread("../Lab2/images01/park468.bmp")
#     img1Q = quantize(image1, 4)
#     img2Q = quantize(image2, 4)
#     img1QG = cv.cvtColor(img1Q, cv.COLOR_BGR2GRAY)
#     img2QG = cv.cvtColor(img2Q, cv.COLOR_BGR2GRAY)
#
#     while(1):
#         t = cv.getTrackbarPos("threshold","output")
#         red = cv.getTrackbarPos("noise","output")
#
#         print("Threshold value: "+str(t))
#         print("Noise Reduction value: " + str(red))
#
#         threshold(img1QG,img2QG,T=t,red=red)
#         cv.waitKey(1)
#
#     cv.destroyAllWindows()

