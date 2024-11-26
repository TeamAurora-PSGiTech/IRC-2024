#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import Image,CompressedImage # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from collections import Counter


class Autonomousmode:
    def __init__(self):
        rospy.init_node('simple_node', anonymous=True)

        # Publisher
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.img_publisher = rospy.Publisher("cvcam/image_raw/compressed", CompressedImage, queue_size=1)
        self.model_path = "/home/aurora/Downloads/smallYolov8Best.pt"
        # Subscriber
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.init()

        # Load the YOLOv7 model
        self.model = YOLO(self.model_path).to(self.device)

        # Open a handle to the video source (default webcam)
        #self.cap = cv2.VideoCapture(video_source)
        #if not self.cap.isOpened():
        #    print("Error: Could not open video stream from the default webcam.")
        #    exit()
        self.Twist_node = Twist()
        self.threshold = 0.3
        self.close = 0.7
        #self.close_contour = 175019
        self.arrow_wait_search_sec = 5
        self.bar = 0.7
        self.wait_start = 15
        self.wait_time_path = 5      
        self.frame = None
        self.left90deg = 4            #Turn 90 degree sec
        self.right90deg = 4   
        self.var=0                              #Function to count left anf right dir 
        self.bbox_detect_start=0.01
        self.bbox_detect_stop=0.03
        self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.callback)
        #this part needed to be optimised for ros
        #self.f_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #self.f_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def callback(self,data):
        #global frame
        br = np.frombuffer(data.data, np.uint8)
        self.frame = cv2.imdecode(br, cv2.IMREAD_COLOR)
        self.f_width = self.frame.shape[1]
        self.f_height = self.frame.shape[0]
        #self.start_rover()
    def what_time(self):
        current_time = rospy.get_time()
        return current_time
    
    def compressed_imgpub(self,frame):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
        self.img_publisher.publish(msg)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        bounding_boxes1 = []
        bounding_boxes =0
        if results.boxes.data.tolist():
            classNames = dict(results.names)
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                label = classNames[int(class_id)]

                if score > self.threshold:
                    # Draw the bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    #self.compressed_imgpub(frame)
                    #dir=int(-2) #Classifcation code -2=>Left , 2=>right
                    id=0 #Classifcation code -2=>Left , 2=>right
                    if "dir" in label.lower():
                        try:
                            arrowBB = frame[int(y1):int(y2), int(x1):int(x2)]
                            gray = cv2.cvtColor(arrowBB, cv2.COLOR_BGR2GRAY)
                            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            contours = max(contours, key=cv2.contourArea)
                            image_with_contour = arrowBB
                            cv2.drawContours(image_with_contour, [contours], -1, (0, 255, 0), 2)
                            bx_x, bx_y, bx_w, bx_h = cv2.boundingRect(contours)
                            croppedImg = arrowBB[bx_y:bx_y + bx_h, bx_x:bx_x + bx_w]
                            gray = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
                            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            largest_contour = max(contours, key=cv2.contourArea)
                            fillImg = np.zeros(shape=croppedImg.shape)
                            cv2.fillPoly(fillImg, [largest_contour], color=255)
                            img_cntX = fillImg.shape[1] // 2
                            img_cntY = fillImg.shape[0] // 2
                            lHalf = fillImg[:img_cntY*2, :img_cntX]
                            rHalf = fillImg[:, img_cntX:]
                            lHalfVal = dict(Counter(lHalf.flatten()))
                            rHalfVal = dict(Counter(rHalf.flatten()))
                            try:
                                if lHalfVal[255.0]>rHalfVal[255.0]:
                                    dir,id = "leftDir",-2
                                else:
                                    dir,id = "rightDir",2
                            except:
                                if lHalfVal[max(list(lHalfVal.keys()))]>rHalfVal[max(list(rHalfVal.keys()))]:
                                    dir,id = "leftDir",-2
                                else:
                                    dir,id = "rightDir",2
                            label = dir
                            print("---> ", label)
                        except:
                            pass

                    # Add bounding box coordinates to the list
                    bounding_boxes1.append((int(x1), int(y1), int(x2), int(y2), int(score), int(class_id) , int(id)))
                bounding_boxes=sorted(bounding_boxes1,key=lambda box: box[4] ,reverse=True)
        self.compressed_imgpub(frame)
        return bounding_boxes

    def run_detection(self):
        while True:
            # frame = self.cap.read() 
            #if not ret:
            #    print("Failed to grab frame")
            #    break

            processed_frame = self.process_frame(self.frame)
            if processed_frame:
                
                x1, y1, x2, y2, score, class_id,dir = processed_frame[0]
                cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                self.compressed_imgpub(self.frame)
            #publisher needs to be written
            self.compressed_imgpub(self.frame)
#            cv2.imshow('Frame', self.frame)

#            if cv2.waitKey(1) & 0xFF == ord('q'):
 #               break

        #self.cap.release()
        cv2.destroyAllWindows()


    
    def start_rover(self):
        print('Start')
        start_time = self.what_time()

        while (self.what_time() - start_time) < self.wait_time_path:
            #global frame
            #do we need to declare this as a global veriable
            #ret, frame = self.cap.read() #needed to change for ros
        #    if not ret:
        #        print("Failed to grab frame")
        #        return
 
            bbox1 = self.process_frame(self.frame)
            if bbox1:
                x1,y1,x2,y2,a,b,c=bbox1[0]
                ratio1 = self.ratio_bounding_box(x1,y1,x2,y2)
                if ratio1 is not None:  # Check if contour1 is not None
                    start_time2 = self.what_time()

                    while (self.what_time() - start_time2) < 3:

                        #ret, frame = self.cap.read()
                        #if not ret:
                        #   print("Failed to grab frame")
                        #  return
                        bbox2 = self.process_frame(self.frame)
                        if bbox2: 
                            print('oo')   
                            x1,y1,x2,y2,a,b,c=bbox2[0]
                            ratio2 = self.ratio_bounding_box(x1,y1,x2,y2)
                            if ratio2 is not None:
                                if ((ratio1 / ratio2) > 0.95 and (ratio1 / ratio2) < 1.05):
                                    self.move_rover()
                                    print('move')
                                    return
        print("Force Forward...")                            
        self.force_forward()
        print("Turn around and check")


    
    def ratio_bounding_box(self, x1, y1, x2, y2):
        ratio = ((x2-x1)*(y2-y1))/(self.f_height*self.f_width)
        return ratio
        
    def move_forward(self):
        self.Twist_node.linear.x = 0.4
        print("Moving Front...")
        self.twist_publisher.publish(self.Twist_node)
    
    def stop(self):
        self.Twist_node.linear.x = 0
        self.Twist_node.angular.z = 0
        self.twist_publisher.publish(self.Twist_node)
    
    def move_rover(self):
        start_time = self.what_time()
        #var=0
        while (self.what_time() - start_time) < self.wait_time_path:
            #ret, frame = self.cap.read()
            #if not ret:
             #   print("Failed to grab frame")
              #  return
            
            bounding_boxes = self.process_frame(self.frame)
            if bounding_boxes:
                x1, y1, x2, y2, score, class_id , dir = bounding_boxes[0]
                #var+=int(dir)

                if self.align_rover(x1,y1,x2,y2):
                    if self.bbox_detect_stop <=self.ratio_bounding_box(x1,y1,x2,y2)<=self.bbox_detect_stop:
                        self.var+=int(dir)
                        
                    if self.ratio_bounding_box(x1,y1,x2,y2) <= self.bbox_detect_stop:
                        print('Go Forward')
                        self.move_forward()
                    else:
                        self.stop()
                        self.arrow_reached()
                        print("Arrow is reached")
                        return          
        else:
            self.force_forward()
            print("Again Force Forward...")

    
    
    def arrow_reached(self):
        print("arrow reached")
        start_time = self.what_time()
        while (self.what_time() - start_time) < 3:    #2sec wait and check arrow
            bounding_boxes = self.process_frame(self.frame)                      
            if bounding_boxes:
                x1, y1, x2, y2, score, class_id , dir = bounding_boxes[0]
                if self.ratio_bounding_box(x1,y1,x2,y2)<=self.bbox_detect_stop:
                    if self.var>0:
                        print("Right arrow detected")
                        self.var=0 
                        self.right_search()                          #Reset Classifier after using it once
                            #mark the gps co - ordinates here -----------------------------------------------------------

                    elif self.var<0:
                        print("Left arrow detected")
                        self.var=0
                        self.left_search()
                            #mark the gps co - ordinates here -----------------------------------------------------------
                    else:
                        self.move_rover()
                        print("arrow reach not verified")
                    return
        else:
            self.move_rover()
            print('arrow reach not verified')
        
            
        
    

    def turn_left(self):
        self.Twist_node.linear.x = 0
        self.Twist_node.angular.z = -0.7
        self.twist_publisher.publish(self.Twist_node)
    
    def turn_right(self):
        self.Twist_node.linear.x = 0
        self.Twist_node.angular.z = 0.7
        self.twist_publisher.publish(self.Twist_node)
    
    def turn_left_slow(self):
        self.Twist_node.linear.x = 0
        self.Twist_node.angular.z = -0.5
        self.twist_publisher.publish(self.Twist_node)
    
    def turn_right_slow(self):
        self.Twist_node.linear.x = 0
        self.Twist_node.angular.z = 0.5
        self.twist_publisher.publish(self.Twist_node)

    def align_rover(self,x1,y1,x2,y2):
        x1, y1, x2, y2 =int(x1), int(y1), int(x2), int(y2)        
        
        if (x1+x2)/2<= (self.f_width/2) - (self.f_width*(self.bar)/2):
            print("Adjust Left")
            self.turn_left_slow()
            rospy.sleep(0.5)            
            return False

        elif (x1+x2)/2< (self.f_width/2) - 2*(self.f_width*(self.bar)/2):
            print("Adjust Left")
            self.turn_left()
            rospy.sleep(0.5)
            return False
        
        elif (x1+x2)/2>= (self.f_width/2) + (self.f_width*(self.bar)/2):
            print("Adjust Right")
            self.turn_right_slow()
            rospy.sleep(0.5)
            return False
    
        elif (x1+x2)/2> (self.f_width/2) + 2*(self.f_width*(self.bar)/2):
            print("Adjust Right")
            self.turn_right()
            rospy.sleep(0.5)
            return False
        else:
            print("Proper Alignment")
            return True
    
    
    def left_search(self):
        #Turn 90 left
        print("Turn Left")
        self.turn_left()
        rospy.sleep(self.left90deg)
        self.stop()
        leftmost = 0
        co_ord=None
        start_time = self.what_time()
        while (self.what_time() - start_time) < self.arrow_wait_search_sec:                 #Search in wide angle 
            bbox3 = self.process_frame(self.frame)
            print('some left object detected')
            if bbox3:
                for detect in bbox3:
                    x1, y1, x2, y2, score, class_id , dir = detect
                    if (x1+x2)/2 < leftmost:
                        leftmost = (x1+x2)/2
                        co_ord=[x1,y1,x2,y2]
        if co_ord is not None:
            x1,y1,x2,y2=co_ord    
            self.align(x1,y1,x2,y2)
            self.move_forward()  #The move forward - is programmed to only operate with the first detected box and wont work with multiple
            
        if co_ord is None:
            self.blind_move()
            

    
    
    def right_search(self):
        #Turn 90 right
        print('Turn Right')
        self.turn_right()
        rospy.sleep(self.right90deg)
        self.stop()
        rightmost = 0
        co_ord=None                                        #Search in wide angle CAM
        start_time = self.what_time()
        while (self.what_time() - start_time) < self.arrow_wait_search_sec:    
            bbox3 = self.process_frame(self.frame)
            print('some right object detected')
            if bbox3:
                for detect in bbox3:
                    x1, y1, x2, y2, score, class_id , dir = detect
                    #DetectedBBOX               
                    if (x1+x2)/2 > rightmost:
                        rightmost = (x1+x2)/2
                        co_ord=[x1,y1,x2,y2]
                        #Lower Detected
        if co_ord is not None:
            x1,y1,x2,y2=co_ord    
            self.align(x1,y1,x2,y2)
            self.move_forward()   #The move forward - is programmed to only operate with the first detected box and wont work with multiple which is most probable here
        
        if co_ord is None:
            self.blind_move()

    def search(self):
        pass


    def prioritze_bbox(self):
        pass

    def force_forward(self):
        print("1112222333")
        self.move_forward()
        while True:
            bbox4=self.process_frame(self.frame)
            if bbox4 is not None:
                #rospy.sleep(2)
                self.move_rover()
                break
            print(bbox4)
        print("At start . rover moves forward by default")
    

    
    def blind_move(self):
        print("Blind move to next arrow") 
        pass
    

     
    

# Create an instance of the Arrow class
#arrow_instance = Arrow(model_path="/home/aurora/Downloads/arrow_300.pt", video_source=0)

   # def publish_data(self, data):
        #self.publisher.publish(data)

    #def callback(self, data):
        #rospy.loginfo('Received: %s', data.data)

if __name__ == '__main__':
    try:
        node_auto = Autonomousmode()

        rate = rospy.Rate(1)  # 1 Hz

        while not rospy.is_shutdown():
            #data = 'Hello World %d' % count
            #node_auto.publish_data(data)
            #count += 1
            if node_auto.frame is not None:
                node_auto.start_rover()
                #node_auto.left_search()
                #node_auto.run_detection()
            #node_auto.start_rover()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
