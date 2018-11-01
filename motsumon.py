import os

import cv2
import numpy as np
import math
import time
import csv

aruco = cv2.aruco
dir(aruco)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

MARKER_LENGTH = 83



def generateMarker():
    marker = aruco.drawMarker(dictionary, 0, 64)
    #cv2.imshow('0.64', marker)
    cv2.imwrite('0.64.png', marker)


def prepareExperiment():
    print('出力ファイル名を指定してね．')
    name = input('>>')
    filename = name + '.csv'
    outputpath = os.path.join('./output/', filename)
    return outputpath


def calibrate_image(input_image):
    height, width = input_image.shape[:2]
    CAMERA_MATRIX = np.array([[3800, 0, int(width/2)], 
                              [0, 6000, int(height/2)], 
                              [0, 0, 1]])
    DISTORTION_COEFFICIENTS = np.array([0.735355, -18.7537, 0.008532, -0.01289, 33.69365])
    #undistort
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    calibrated_image = cv2.undistort(input_image, CAMERA_MATRIX,
                                     DISTORTION_COEFFICIENTS, None, CAMERA_MATRIX)
    return calibrated_image


def detectMarker(outputpath):
    cap = cv2.VideoCapture(0)
    time_start = time.time()

    with open(outputpath, 'a') as f:

        while (cap.isOpened()):
            outputlist = []
            elapsed_time = time.time() - time_start
            outputlist.append(elapsed_time)

            ret, frame = cap.read()
            frame = calibrate_image(frame)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)

            for i, corner in enumerate(corners):
                points = corner[0].astype(np.int32)
                
                mklength = distance(points[0][0], points[0][1], points[1][0], points[1][1])
                length_per_pix = MARKER_LENGTH / mklength
                center = [int(0.5 * (points[0][0] + points[2][0])), int(0.5 * (points[0][1] + points[2][1]))]
                outputlist.append(center[0])
                outputlist.append(center[1])
                outputlist.append(points[0][0])
                outputlist.append(points[0][1])
                cv2.polylines(frame, [points], True, (0,255,0))
                cv2.putText(frame, str(ids[i][0]),
                            tuple(points[0]),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,(0,0,0), 1)
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
                cv2.line(frame, tuple(center), tuple(points[0]), (0, 0, 255), 2)

            cv2.imshow('Frame', frame)
            print(outputlist)
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            writer.writerow(outputlist)     # list（1次元配列）の場合

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()


def distance(x1, y1, x2, y2):
        xd = x2 - x1
        yd = y2 - y1 
        distance = math.sqrt(pow(xd, 2)+pow(yd, 2))
        return distance


def main():
    outputpath = prepareExperiment()

    #generateMarker()
    detectMarker(outputpath)


if __name__ == '__main__':
    main()