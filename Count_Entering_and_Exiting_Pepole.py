import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO

path = r"D:\Computer Vision\yolov8\Count-entering-and-exiting-people-from-hotel-using-YOLOv8-main\Count-entering-and-exiting-people-from-hotel-using-YOLOv8-main"
os.chdir(path)

from tracker import Tracker


# Model
model = YOLO(r"yolov8s.pt")


# areas for entering and exiting
area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]


with open(r"coco.txt") as f:
    classes = f.read().split("\n")

    
people_enter = {}
people_exit = {}
enter = set()
exit = set()


def points(events, x, y, flags, param):
    if events == cv2.EVENT_MOUSEMOVE:
        coordxy = [x, y]
        # print(f"Coordinates {coordxy}")

cv2.namedWindow("points")
cv2.setMouseCallback("points", points)


tracker = Tracker()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame not Read")
            break

        frame = cv2.resize(frame, (1020, 500))
        result = model(frame)
        bb = result[0].boxes.data
        df = pd.DataFrame(bb).astype("float")
        
        lis = []
        
        for box in bb:
            x1, y1, x2, y2, c = map(int, box[:5])
            if classes[c] == "person":
                lis.append([x1, y1, x2, y2])
        
        bbox_id = tracker.update(lis)
        
        update_tracking_info(frame, bbox_id)
        display_info(frame)
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(10) == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    
    
def update_tracking_info(frame, bbox_id):
    for bb in bbox_id:
        x3, y3, x4, y4, id = bb
        update_area_info(frame, id, x3, y3, x4, y4)

        
        
def update_area_info(frame, id, x3, y3, x4, y4):
        results = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results >= 0:
            people_enter[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)

        if id in people_enter:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.putText(frame, "Person", (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, str(id), (x3 + 65, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
                enter.add(id)
                

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results2 >= 0:
            people_exit[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)

        if id in people_exit:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.putText(frame, "Person", (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, str(id), (x3 + 55, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
                exit.add(id)

                
def display_info(frame):
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, "1", (505, 470), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, "2", (466, 485), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, 'Number of Entering People = ' + str(len(enter)), (20, 44), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.putText(frame, 'Number of Exiting People = ' + str(len(exit)), (20, 82), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    
if __name__ == "__main__":
    process_video(r"peoplecount1.mp4")
