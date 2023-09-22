#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import time
from numpy_ros import to_numpy

import numpy



def callback(msg):

    points = point_cloud2.read_points_list(
            msg, field_names=("x", "y", "z"))

    print(type(points[1]))
    print(points[1].x)


    time.sleep(10)


def listener():
 
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/first_lidar", PointCloud2, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
if __name__ == '__main__':
    listener()