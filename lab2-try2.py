import cv2 as cv
import numpy as np

def nothing(x):
    pass

def threshold(img1, img2, T, red):

    # raise an exception if the two images aren't from the same video (assumption that the resolutions are the same)
    if(img1.shape[0]!=img2.shape[0] or img1.shape[1] !=img2.shape[1]):
        raise Exception("The two images provided are not of the same length")

    # height and width
    h = img1.shape[0]
    w = img1.shape[1]

    # I will default the new image to a matrix of 0's (black) and populate with 1's (white) when necessary
    diffpre = np.zeros((h, w, 3), dtype='uint8')
    diff = cv.cvtColor(diffpre,cv.COLOR_RGB2GRAY)

    fin = cv.cvtColor(np.zeros((h, w, 3), dtype='uint8'), cv.COLOR_RGB2GRAY)
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

if __name__ == "__main__":
    cv.namedWindow("output")
    cv.createTrackbar("threshold", "output", 0, 255, nothing)
    cv.createTrackbar("noise", "output", 0, 1000, nothing)

    while(1):
        image1 = cv.imread("../Lab2/images01/park466.bmp")
        imgG1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
        image2 = cv.imread("../Lab2/images01/park467.bmp")
        imgG2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        t = cv.getTrackbarPos("threshold","output")
        red = cv.getTrackbarPos("noise","output")
        print("Threshold value"+str(t))
        threshold(imgG1,imgG2,T=t,red=red)
        cv.waitKey(1)

    cv.destroyAllWindows()

