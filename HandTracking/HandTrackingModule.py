"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

import cv2
import mediapipe as mp #pip install mediapipe
import time
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.ROI = None

    def rectangle_roi(self,img):

        h, w, _ = img.shape
        hand_landmarks = self.results.multi_hand_landmarks
        img_roi = np.copy(img)
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                # #cv2.rectangle(img,(x,y),(w+x,h+y),[255,0,0],thickness=1)
                # cv2.rectangle(img_roi, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # #ROI = image[y:y+h, x:x+w]
                ROI = img_roi[y_min-30:y_max+30, x_min-30:x_max+30]
                if ROI is not None and ROI.shape[0] > 1 and ROI.shape[1] > 1:
                    # cv2.imshow('ROI', ROI)
                    self.ROI = ROI
                    return self.ROI
        return None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print('self.results: ' + str(self.results.multi_hand_landmarks))

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        # cv2.imshow('findHands', img)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                # h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                #_ = self.rectangle_roi(img)

                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    # tipIds = [4, 8, 12, 16, 20]
    # tipIds_point_hand = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # tipIds_point_hand = [5, 9, 10, 11, 5, 6, 15, 16, 13, 17, 18, 19, 1, 2, 6, 7, 0, 17, 3, 4, 9, 10, 0, 5, 2, 3, 14, 15,
    #                      11, 12, 19, 20, 0, 1, 9, 13, 17, 18, 13, 14, 7, 8]

    # dic_point = {
    #     'point_hand_1' : [8, 7, 6, 5],
    #     'point_hand_2' : [12, 11, 10, 9],
    #     'point_hand_3' : [16, 15, 14, 13],
    #     'point_hand_4' : [20, 19, 18, 17],
    #     'point_hand_5' : [4, 3, 2, 1, 0],
    #     'point_hand_6' : [0, 5, 9, 13, 17, 0]
    # }

    dic_point = {
        'line_hand_1': [5, 9],
        'line_hand_2': [10, 11],
        'line_hand_3': [5, 6],
        'line_hand_4': [15, 16],
        'line_hand_5': [13, 17],
        'line_hand_6': [18, 19],
        'line_hand_7': [1, 2],
        'line_hand_8': [6, 7],
        'line_hand_9': [0, 17],
        'line_hand_10': [3, 4],
        'line_hand_11': [9, 10],
        'line_hand_12': [0, 5],
        'line_hand_13': [2, 3],
        'line_hand_14': [14, 15],
        'line_hand_15': [11, 12],
        'line_hand_16': [19, 20],
        'line_hand_17': [0, 1],
        'line_hand_18': [9, 13],
        'line_hand_19': [17, 18],
        'line_hand_20': [13, 14],
        'line_hand_21': [7, 8],
    }


    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img_findHands = np.copy(img)
        img_findHands = detector.findHands(img_findHands, draw=True)
        lmList = detector.findPosition(img,  draw=True)
        if len(lmList) == 21:
            # print(len(lmList))

            for ii, data in enumerate(dic_point):
                # print(data)
                x1, y1 = lmList[dic_point[data][0]][1], lmList[dic_point[data][0]][2]
                x2, y2 = lmList[dic_point[data][1]][1], lmList[dic_point[data][1]][2]

                img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.imshow("img_findHands", img_findHands)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()