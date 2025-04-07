#importing necessary modules and library
import cv2
import numpy as np
import time
from flask import Flask, render_template, request
# creates the Flask instance.
app = Flask(__name__)
# Pass the required route to the decorator.
@app.route("/")
def cvv():
    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNet("yolov3.cfg","yolov3.weights")
    classes = []
    # Load names of classes
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
    # determine the output layer
    layer_names = net.getLayerNames()
    output_layers= [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    #Load Video
    cap = cv2.VideoCapture("demovidlos.mp4")
    font = cv2.FONT_HERSHEY_SIMPLEX
    startingtime = time.time()
    frame_id = 0
    while True:
        """ This Loop will take frame as image and detect the objects in that frame
        
        """
        _, frame = cap.read()
        height, width, channels = frame.shape
        #detecting Objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, #scale
                             (416, 416), #size of img 416x416 square image to yolo
                             (0, 0, 0),
                             True, crop=False)
        net.setInput(blob) 
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " , (x + 50, y + 50), font, 1, color, 1)
    

        cv2.imshow("Rsr Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0")