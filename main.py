import cv2
import numpy as np
import time

conf_thresh= 0.62
nms_threshold = 0.2
vs = cv2.VideoCapture(0)

CLASSES = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    CLASSES = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn.readNet(weightsPath,configPath)

time.sleep(2.0)

while True:
    _, frame = vs.read()  # reading frame from the camera
    # frame = imutils.resize(frame, width=1355)  # resize the frame to be displayed as window
    (h, w) = frame.shape[:2]  # h w
    # preprocessing
    imResize = cv2.resize(frame, (600, 600))  # resize
    blob = cv2.dnn.blobFromImage(imResize,
                                 0.007843, (300, 300), 127.5)  # blobed image

    net.setInput(blob)  # set the blobbed image as input
    detections = net.forward()  # passing pre processed image into model
    #print(detections)
    detShape = detections.shape[2]
    for i in np.arange(0, detShape):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            idx = int(detections[0, 0, i , 1])
            # print("ClassID:",detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # print("boxCoord:",detections[0, 0, i, 3:7])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx -1 ],
                                         confidence * 100)
            # print(label)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            if startY - 15 > 15:
                y = startY - 15
            else:
                y = startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

vs.release()
cv2.destroyAllWindows()
