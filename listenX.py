import os
import sys
import time
import rospy
import multiprocessing
from std_msgs.msg import String
def start_ros_core():
    result_roscore = os.system('roscore')
# 使用代码
ros1bag_core_process = multiprocessing.Process(target=start_ros_core)
ros1bag_core_process.start()
rospy.init_node("test_pub", anonymous=True)#初始化节点 名称：test
def sub_mutil():
    num = 0
    while True:
        my_msg = String()
        my_msg.data = f'{num}'
        pub_parking.publish(my_msg)
        time.sleep(1)
        print(f"发送消息:{num}")
        num+=1
#发布话题
pub_parking = rospy.Publisher('mystring', String, queue_size=10)
sub_mutil()