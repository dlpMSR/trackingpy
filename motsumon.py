import cv2
import numpy as np

aruco = cv2.aruco
dir(aruco)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

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
    CAMERA_MATRIX = np.array([[1279.334, 0, 331.0683], 
                              [0, 1276.837, 202.1932], 
                              [0, 0, 1]])
    DISTORTION_COEFFICIENTS = np.array([0.735355, -18.7537, 0.008532, -0.01289, 33.69365])
    #undistort
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    calibrated_image = cv2.undistort(input_image, CAMERA_MATRIX,
                                     DISTORTION_COEFFICIENTS, None, CAMERA_MATRIX)
    return calibrated_image


def main():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = calibrate_image(frame)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
        #print(corners)
        for i, corner in enumerate(corners):
            points = corner[0].astype(np.int32)
            cv2.polylines(frame, [points], True, (0,0,255))
            cv2.putText(frame, str(ids[i][0]),
                        tuple(points[0]),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,(0,0,0), 1)
        cv2.imshow('Frame', frame)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()