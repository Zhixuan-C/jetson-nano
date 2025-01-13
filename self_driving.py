#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# 无人驾驶
import os
import cv2
import math
import rospy
import signal
import threading
import numpy as np
import lane_detect
import hiwonder_sdk.pid as pid
import hiwonder_sdk.misc as misc
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
import geometry_msgs.msg as geo_msg
import hiwonder_sdk.common as common
from hiwonder_interfaces.msg import ObjectsInfo
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from hiwonder_servo_controllers.bus_servo_control import set_servos

class SelfDrivingNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        self.name = name
        self.image = None
        self.is_running = True
        self.pid = pid.PID(0.01, 0.0, 0.0)

        self.detect_far_lane = False
        self.park_x = -1  # 停车标识的x像素坐标

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # 开始转弯

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # 右转标志
        
        self.stop = False  # 停下标识
        self.start_park = False  # 开始泊车标识

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # 离斑马线距离
        self.crosswalk_length = 0.1 + 0.35  # 斑马线长度 + 车长

        self.start_slow_down = False  # 减速标识
        self.normal_speed = 0.15  # 正常前进速度
        self.slow_down_speed = 0.1  # 减速行驶的速度

        self.traffic_signs_status = None  # 记录红绿灯状态

        self.colors = common.Colors()
        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")
        self.mecanum_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', geo_msg.Twist, queue_size=1)  # 底盘控制
        # self.result_publisher = rospy.Publisher(self.name + '/image_result', Image, queue_size=1)  # 图像处理结果发布
        self.joints_pub = rospy.Publisher('servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)  # 舵机控制
        depth_camera = rospy.get_param('/depth_camera/camera_name', 'depth_cam')  # 获取参数
        rospy.Subscriber('/%s/rgb/image_raw' % depth_camera, Image, self.image_callback)  # 摄像头订阅
        rospy.Subscriber('/yolov5/object_detect', ObjectsInfo, self.get_object_callback)
        if not rospy.get_param('~only_line_follow', False):
            while not rospy.is_shutdown():
                try:
                    if rospy.get_param('/yolov5/init_finish'):
                        break
                except:
                    rospy.sleep(0.1)
            rospy.ServiceProxy('/yolov5/start', Trigger)()
        while not rospy.is_shutdown():
            try:
                if rospy.get_param('/hiwonder_servo_manager/init_finish') and rospy.get_param('/joint_states_publisher/init_finish'):
                    break
            except:
                rospy.sleep(0.1)
        
        set_servos(self.joints_pub, 1000, ((6, 490),))  # 初始姿态
        rospy.sleep(1)
        self.mecanum_pub.publish(geo_msg.Twist())
        #self.park_action() 
        self.image_proc()

    def shutdown(self, signum, frame):  # ctrl+c关闭处理
        self.is_running = False
        rospy.loginfo('shutdown')

    def image_callback(self, ros_image):  # 目标检查回调
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data) # 原始 RGB 画面
        self.image = rgb_image
    
    # 泊车处理
    def park_action(self):
        twist = geo_msg.Twist()
        twist.linear.x = 0.15
        twist.angular.z = twist.linear.x*math.tan(-0.6)/0.213
        self.mecanum_pub.publish(twist)
        rospy.sleep(3)

        twist = geo_msg.Twist()
        twist.linear.x = 0.15
        twist.angular.z = -twist.linear.x*math.tan(-0.6)/0.213
        self.mecanum_pub.publish(twist)
        rospy.sleep(2)

        twist = geo_msg.Twist()
        twist.linear.x = -0.15
        twist.angular.z = twist.linear.x*math.tan(-0.6)/0.213
        self.mecanum_pub.publish(twist)
        rospy.sleep(1.5)

        self.mecanum_pub.publish(geo_msg.Twist())

    def image_proc(self):
        while self.is_running:
            if self.image is not None:
                h, w = self.image.shape[:2]

                # 获取车道线的二值化图
                binary_image = self.lane_detect.get_binary(self.image)
                # 检测到斑马线,开启减速标志
                if 450 < self.crosswalk_distance and not self.start_slow_down:  # 只有足够近时才开始减速
                    self.count_crosswalk += 1
                    if self.count_crosswalk == 3:  # 多次判断，防止误检测
                        self.count_crosswalk = 0
                        self.start_slow_down = True  # 减速标识
                        self.count_slow_down = rospy.get_time()  # 减速固定时间
                else:  # 需要连续检测，否则重置
                    self.count_crosswalk = 0

                twist = geo_msg.Twist()
                # 减速行驶处理
                if self.start_slow_down:
                    if self.traffic_signs_status == 'red':  # 如果遇到红灯就停车
                        self.mecanum_pub.publish(geo_msg.Twist())
                        self.stop = True
                    elif self.traffic_signs_status == 'green':  # 遇到绿灯，速度放缓
                        twist.linear.x = self.slow_down_speed
                        self.stop = False
                    elif not self.stop:  # 其他非停止的情况速度放缓， 同时计时，时间=斑马线的长度/行驶速度
                        twist.linear.x = self.slow_down_speed
                        if rospy.get_time() - self.count_slow_down > self.crosswalk_length/twist.linear.x:
                            self.start_slow_down = False
                else:
                    twist.linear.x = self.normal_speed  # 直走正常速度
                
                # 检测到 停车标识+斑马线 就减速, 让识别稳定
                if 0 < self.park_x and 150 < self.crosswalk_distance:
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park and 255 < self.crosswalk_distance:  # 离斑马线足够近时就开启停车
                        self.mecanum_pub.publish(geo_msg.Twist())
                        self.start_park = True
                        self.stop = True
                        threading.Thread(target=self.park_action).start()                       
                
                # 右转及停车补线策略
                if self.turn_right:
                    y = self.lane_detect.add_horizontal_line(binary_image)
                    if 300 < y < 400:
                        roi = [(0, y), (w, y), (w, 0), (0, 0)]
                        cv2.fillPoly(binary_image, [np.array(roi)], [0, 0, 0])  # 将上面填充为黑色，防干扰
                        min_x = cv2.minMaxLoc(binary_image)[-1][0]
                        cv2.line(binary_image, (min_x, y), (w, y), (255, 255, 255), 30)  # 画虚拟线来驱使转弯
                elif 0 < self.park_x and not self.start_turn:  # 检测到停车标识需要填补线，使其保持直走
                    if not self.detect_far_lane:
                        up, down, center = self.lane_detect.add_vertical_line_near(binary_image)
                        binary_image[:, :] = 0  # 全置黑，防止干扰
                        if center > 130:  # 当将要看不到车道线时切换到识别较远车道线
                            self.detect_far_lane = True
                    else:
                        up, down = self.lane_detect.add_vertical_line_far(binary_image)
                        binary_image[:, :] = 0
                    if up != down:
                        cv2.line(binary_image, up, down, (255, 255, 255), 20)  # 手动画车道线

                result_image, lane_angle, lane_x = self.lane_detect(binary_image, self.image.copy())  # 在处理后的图上提取车道线中心
                # 巡线处理
                if lane_x >= 0 and not self.stop:
                    if lane_x > 140:  # 转弯
                        if self.turn_right:  # 如果是检测到右转标识的转弯
                            self.count_right_miss += 1
                            if self.count_right_miss >= 100:
                                self.count_right_miss = 0
                                self.turn_right = False
                        self.count_turn += 1
                        if self.count_turn > 5 and not self.start_turn:  # 稳定转弯
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = rospy.get_time()
                        twist.angular.z = twist.linear.x*math.tan(-0.6)/0.213  # 转弯速度
                    else:  # 直道由pid计算转弯修正
                        self.count_turn = 0
                        if rospy.get_time() - self.start_turn_time_stamp > 2 and self.start_turn:
                            self.start_turn = False
                        if not self.start_turn:
                            if abs(lane_x - 90) < 15:
                                lane_x = 90
                            self.pid.SetPoint = 90  # 在车道中间时线的坐标
                            self.pid.update(lane_x)
                            twist.angular.z = twist.linear.x*math.tan(misc.set_range(self.pid.output, -0.1, 0.1))/0.213
                        else:
                            twist.angular.z = 0.15*math.tan(-0.6)/0.213  # 转弯速度
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()
                bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('result', bgr_image)
                key = cv2.waitKey(1)
                if key != -1:
                    self.is_running = False
                #ros_image.data = result_image.tostring()
                #self.result_publisher.publish(ros_image)
            else:
                rospy.sleep(0.01)
        self.mecanum_pub.publish(geo_msg.Twist())

    # 获取目标检测结果
    def get_object_callback(self, msg):
        objects_info = msg.objects
        if objects_info == []:  # 没有识别到时重置变量
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                
                if class_name == 'crosswalk':  
                    if center[1] > min_distance:  # 获取最近的人行道y轴像素坐标
                        min_distance = center[1]
                elif class_name == 'right':  # 获取右转标识
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 10:  # 检测到多次就将右转标志至真
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'park':  # 获取停车标识中心坐标
                    self.park_x = center[0]
                elif class_name == 'red' or class_name == 'green':  # 获取红绿灯状态
                    self.traffic_signs_status = class_name
        
            self.crosswalk_distance = min_distance

if __name__ == "__main__":
    SelfDrivingNode('self_driving')
