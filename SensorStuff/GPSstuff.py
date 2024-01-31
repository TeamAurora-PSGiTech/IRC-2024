#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
import serial
import pynmea2

class GPS:
    def __init__(self):
        rospy.init_node('GPS_NODE')
        self.serial_port = serial.Serial(port="/dev/ttyUSB0", baudrate=9600, timeout=10.0)
        self.float_longi_pub = rospy.Publisher("longi", Float32, queue_size=10)
        self.float_lati_pub = rospy.Publisher("lati", Float32, queue_size=10)

    def get_longi_lati(self):
        while not rospy.is_shutdown():
            try:
                gps_data = self.serial_port.readline().decode('ascii', errors='replace')
                #rospy.loginfo("Raw GPS Data: {}".format(gps_data))  # Log raw GPS data

                # Check for GNRMC sentence
                if gps_data.startswith('$GNRMC'):
                    nmea_sentence = pynmea2.parse(gps_data)
                    latitude = nmea_sentence.latitude
                    longitude = nmea_sentence.longitude

                    # Log the formatted information
                    rospy.loginfo("Latitude: {:.6f}, Longitude: {:.6f}".format(latitude, longitude))

                    # Publish to ROS topics if needed
                    self.float_longi_pub.publish(float(longitude))
                    self.float_lati_pub.publish(float(latitude))
            except Exception as e:
                rospy.logerr(f"Error processing GPS data: {e}")

if __name__ == '__main__':
    gps = GPS()
    try:
        gps.get_longi_lati()
    except rospy.ROSInterruptException:
        pass
    finally:
        gps.serial_port.close()