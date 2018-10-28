import cv2
import numpy as np
import math

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


def calibrate_image(input_image):
    CAMERA_MATRIX = np.array([[3800, 0, 940], 
                              [0, 6000, 540], 
                              [0, 0, 1]])
    DISTORTION_COEFFICIENTS = np.array([0.735355, -18.7537, 0.008532, -0.01289, 33.69365])
    #undistort
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    calibrated_image = cv2.undistort(input_image, CAMERA_MATRIX,
                                     DISTORTION_COEFFICIENTS, None, CAMERA_MATRIX)
    return calibrated_image


def detectMarker():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = calibrate_image(frame)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
        #print(corners)
        for i, corner in enumerate(corners):
            points = corner[0].astype(np.int32)
            cv2.polylines(frame, [points], True, (0,255,0))

            mk_length = distance(points[0][0], points[0][1], points[1][0], points[1][1])
            lengthperpix = MARKER_LENGTH / mk_length
            print(lengthperpix)
            center = [int(0.5*(points[0][0]+points[2][0])), int(0.5*(points[0][1]+points[2][1]))]
            cv2.putText(frame, str(ids[i][0]),
                        tuple(points[0]),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,(0,0,0), 1)
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
            cv2.line(frame, tuple(center), tuple(points[0]), (0, 0, 255), 5)

        cv2.imshow('Frame', frame)

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
    #generateMarker()
    detectMarker()


if __name__ == '__main__':
    main()