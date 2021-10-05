import cv2 as cv
import numpy as np
from optparse import OptionParser
from diff import diff

def nothing(x):
    pass

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--inputFilePath","-i")
    parser.add_option("--outputFilePath","-o")

    options, _ = parser.parse_args()

    if options.inputFilePath is None:
        raise Exception("Did not input the required argument: --inputFilePath")
    elif options.outputFilePath is None:
        raise Exception("Did not input the required argument: --outputFilePath")

    cv.namedWindow("output")
    # load image
    video = cv.VideoCapture(options.inputFilePath)
    width, height = int(video.get(3)), int(video.get(4))
    writer = cv.VideoWriter(options.outputFilePath, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (width, height))

    cv.createTrackbar("threshold", "output", 0, 255, nothing)
    cv.createTrackbar("noise", "output", 0, 1000, nothing)
    cv.createTrackbar("k", "output", 1, 10, nothing)

    cv.waitKey(0)

    #while loop
    while (video.isOpened()):
        # retrieval and frame for both 2 consecutive video frames
        ret1, frame1 = video.read()
        ret2, frame2 = video.read()

        # In the scenario where there is no video or the end of the video is reached...
        # This will break the loop and return the result
        if ret1 and ret2:

            # output differenced frame
            diff_frame=diff(frame1,frame2)
            #write to file
            writer.write(diff_frame)

            # while parsing can see a side-by-side of the video being captured.
            cv.imshow("output",np.hstack([cv.cvtColor(frame2,cv.COLOR_BGR2GRAY),diff_frame]))
            cv.waitKey(10)
        else:
            break
    video.release()
    cv.destroyAllWindows()