#!/usr/bin/python3
# coding=utf8
"""
    MyAthleticsPerform.py 我的田径表演
    2022.10.07 17点50分
"""
# ---------------------------------------------- 库 START ----------------------------------------------
import sys
import cv2
import time
import math
import threading
import numpy as np
import subprocess  # 仅用于缝合KickBall.py (领导要求)

# 幻尔官方给的库, 注意库更新需要重新 install
import hiwonder.Misc as Misc
import hiwonder.Board as Board
import hiwonder.PID as PID
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle
from hiwonder.MP3 import MP3
import hiwonder.ASR as ASR

import hiwonder.Mpu6050 as Mpu6050

# ---------------------------------------------- 库 END ----------------------------------------------


# ---------------------------------------------- 全局变量 START ----------------------------------------------
# 田径表演所用到的动作组
hk_stand = 'hk_stand'
hk_go_forward = '20221007_hk_go'
hk_go_forward_1_step = '20221011_hk_go_1_step'
hk_back_1_step = '20221012_hk_back_1_step_v1'
hk_turn_right = '20221008_hk_turn_right'
hk_turn_left = '20221011_hk_turn_left'
hk_left_move10 = '20221011_hk_left_move10'
hk_right_move10 = '20221009_hk_right_move_10'
hk_left_move20 = '20221009_hk_left_move20'
hk_right_move20 = '20221009_hk_right_move20'

# 从yaml文件中读取的字典数据
lab_data = {}
servo_data = {}

# 视觉相关
obj_left_x, obj_right_x, obj_center_y, obj_angle = -1, -1, -1, 0  # obj 为 台阶, 栏杆
switch = False  # 是否已经执行stand_slow, 开关

# 循迹相关变量
line_centroid_x = -1  # 黑线质心x坐标在图像中的位置, -1未找到, Value >= 0 识别到黑线
SIZE = (640, 480)  # resize后的图像大小 (宽, 高)

# 场景判断所用到的变量 sports::体育
skip = 1  # 跨栏项目, skip == 1跨栏, skip == 2 上下台阶,skip == 3 踢球(绿色)
skip_st = True  # 是否检测到 skip 项目?, 若未检测到 skip+1
items = None  # 字符串 "Hurdles.py", "stairway.py"
line_st = True  # True: 巡线, False: 跳过巡线项目, 是否执行巡线项目
stair_up = True  # True 执行上台阶, False 执行下台阶
perform_end = False  # 田径表演结束标志位, 在第一次登上台阶, 下台阶时置位, 第二次准备登台时后退5步, 启动踢球程序
x_center = 320  # 线的正中心, LINE_CENTER_X

ready_2_pass = False  # 位姿调整完毕, 小步前进准备通过
last_broadcast_time = 0  # 上次语音播报时间

ret2 = False  # 摄像头图像读取返回值

mpu = Mpu6050.mpu6050(0x68)  # 启动Mpu6050
mpu.set_gyro_range(mpu.GYRO_RANGE_2000DEG)  # 设置Mpu6050的陀螺仪的工作范围
mpu.set_accel_range(mpu.ACCEL_RANGE_2G)  # 设置Mpu6050的加速度计的工作范围

count1 = 0
count2 = 0


# ---------------------------------------------- 全局变量 END ----------------------------------------------


# ---------------------------------------------- 函数 START ----------------------------------------------
def standup():
    global count1, count2

    try:
        accel_date = mpu.get_accel_data(g=True)  # 获取传感器值
        angle_y = int(math.degrees(math.atan2(accel_date['y'], accel_date['z'])))  # 将获得的数据转化为角度值

        if abs(angle_y) > 160:  # y轴角度大于160，count1加1，否则清零
            count1 += 1
        else:
            count1 = 0

        if abs(angle_y) < 10:  # y轴角度小于10，count2加1，否则清零
            count2 += 1
        else:
            count2 = 0

        time.sleep(0.1)

        if count1 >= 2:  # 往前倒了一定时间后起来
            count1 = 0
            print("stand up back！")  # 打印执行的动作名
            AGC.runAction('stand_up_back')  # 执行动作

        elif count2 >= 2:  # 往后倒了一定时间后起来
            count2 = 0
            print("stand up front！")  # 打印执行的动作名
            AGC.runAction('stand_up_front')  # 执行动作

    except BaseException as e:
        print(e)


# 变量重置 reset
def rst():
    global obj_left_x, obj_right_x, obj_center_y, obj_angle
    global switch

    switch = False
    obj_left_x, obj_right_x, obj_center_y, obj_angle = -1, -1, -1, 0


# 初始头部舵机位置
def init_head():
    Board.setPWMServoPulse(1, 850, 500)
    Board.setPWMServoPulse(2, 1300, 500)


# 从文件中加载 lab_data、servo_data
def load_config():
    global lab_data, servo_data

    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    # servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)   # 没有用到, 手动写入了


# 初始化函数?
def init():
    load_config()  # 从文件中加载配置
    init_head()  # 头部舵机初始化
    rst()  # 全局变量复位


# buzzer 线程
def t_buzzer():
    for _x in range(6):  # 3*2=6 3个周期,一个周期两个状态
        Board.setBuzzer(True if _x % 2 == 0 else False)  # 偶数响, 奇数哑
        time.sleep(0.1)


def cmp_area(c0):
    return math.fabs(cv2.contourArea(c0))


# 找出面积最大的轮廓, 参数为要比较的轮廓的列表
def get_area_max(contours, area_min=10):
    _area_max = 0  # 最大轮廓的面积
    c_area_max = None  # 面积最大的轮廓对象

    for c in contours:  # 历遍所有轮廓
        _area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if _area_temp > _area_max:
            _area_max = _area_temp

            if _area_temp >= area_min:  # 只有在面积大于设定值时，最大面积的轮廓才是有效的，以过滤干扰
                c_area_max = c

    return c_area_max, _area_max  # 返回最大的轮廓


def line_patrol(_img):
    """
        巡线
        参数1 : 原始图像
        (弃用)参数2 : 文本型   欲寻线的颜色 'red'、'blue' 与 lab_tool 中的颜色一一对应

        返回值: 整数型, 黑线通过加权平均计算出的中心点
    """
    # 全局变量
    global line_centroid_x

    # ====================================== 局部变量 Start ======================================
    n = 0  # 用于遍历ROI, 计数
    centroid_x_sum = 0  # 质心 x 权重和
    weight_sum = 0  # sum(ROI[:, 0:4])
    # line_centroid_x =  centroid_x_sum / weight_sum

    # region of interest 感兴区域
    # y0, y1, x0, x1, 权重
    roi = [(300, 360, 0, 640, 0.1),  # (y1:y1), (x1:x2), 权重
           (360, 420, 0, 640, 0.3),  # 每层高度 60
           (420, 480, 0, 640, 0.6)]  # 将图像分割成上中下三个部分, 加速运算

    # ====================================== 局部变量 End ========================================

    # 复制一份源图像，Debug 显示用; _img 用作数据处理
    img_debug = _img.copy()

    # 获取图像大小
    img_h, img_w = _img.shape[:2]

    # 图像本身就是 (640, 480) 的, 所以没必要再 resize 了
    # img_resize = cv2.resize(_img, SIZE, interpolation=cv2.INTER_NEAREST)  # 重置图像大小

    # 遍历自定义的 ROI 区域
    for r in roi:
        # 画出 ROI 区域, 蓝色线, # (r[2], r[0]), (r[3], r[1]) 分别为 roi 左上角和右下角
        cv2.rectangle(img_debug, (r[2], r[0]), (r[3], r[1]), (255, 0, 0), 1)

        # 将图像切割成块 (高1, 高2)、(宽0， 宽1), 例如 (300, 340), (0, 640)
        img_block = _img[r[0]:r[1], r[2]:r[3]]

        # 将图像块转换成灰度图
        img_gray = cv2.cvtColor(img_block, cv2.COLOR_BGR2GRAY)

        # 局部二值化处理, 消除光晕及影子对黑线的影响; 参考 Jupyter
        ada_dst = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 77)

        # 找出所有外轮廓
        contours, _ = cv2.findContours(ada_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 该块未发现轮廓
        if len(contours) == 0:
            continue

        # 对轮廓对象按面积进行排序, 降序, 面积最大的为序号为0
        c_sorted = sorted(contours, key=cmp_area, reverse=True)

        # 如果面积小于 20, 认为是噪声, 下一个块处理
        if math.fabs(cv2.contourArea(c_sorted[0])) <= 20:
            continue

        # 否则识别到黑线, 这个程序目前只能处理黑线, 其他颜色的线不能用灰度色彩空间处理;
        # 在原图像上绘制轮廓, offset 为 roi 左上角坐标, 用于偏移
        cv2.drawContours(img_debug, c_sorted, 0, (0, 0, 255), 2, offset=(r[2], r[0]))

        # 画框处理
        rect = cv2.minAreaRect(c_sorted[0])  # 最小外接矩形
        box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点

        # 获取矩形的对角点, 根据对角线求出中心点, 1,3 2,4 对角
        pt1_x, pt1_y = box[0, 0], box[0, 1]
        pt3_x, pt3_y = box[2, 0], box[2, 1]

        center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2  # 中心点
        cv2.circle(img_debug, (int(center_x), int(center_y) + r[0]), 3, (0, 0, 255), -1)  # 画出中心点, -1填充
        # +r[0]是block ROI y起始位置;

        # 按权重不同对上中下三个中心点进行求和
        centroid_x_sum += center_x * r[4]  # 执行了三次
        weight_sum += r[4]  # 0.1 + 0.3 + 0.6   # 三个区域, 的权重之和, 越靠后, 权重越大

        # 显示 ROI 块, 识别到的黑线块
        n = n + 1
        cv2.imshow('img_mask' + str(n), ada_dst)

    # 权重值为0未找到黑线, 不为0, 某个块识别到了黑线; 或者说面积大于 20(根据实际情况修改)
    if weight_sum != 0:
        # 求得加权平均中心点
        line_centroid_x = int(centroid_x_sum / weight_sum)  # 根据加权平均, 计算出的中心点
        cv2.circle(img_debug, (line_centroid_x, 360), 7, (0, 255, 255), -1)  # 画出中心, 这里y坐标是固定的; 巡线不关心y坐标;
    else:
        line_centroid_x = -1  # 错误值, 未寻找到黑线

    # 显示 debug 图像, 用户 UI
    cv2.imshow('line_patrol', img_debug)

    # 返回线的质心
    return line_centroid_x


def color_identify(_img, target_color='blue'):
    hsv_data = {
        'red': {'lower': (0, 43, 46), 'upper': (10, 255, 255)},
        'red1': {'lower': (156, 43, 46), 'upper': (180, 255, 255)},  # 红色有两个区间
        'blue': {'lower': (100, 43, 46), 'upper': (124, 255, 255)}
    }

    # 图像分割
    _img = _img[250:480, 0:640]

    # 复制一份源图像，Debug 显示用; _img 用作数据处理
    img_debug = _img.copy()

    # 获取图像宽高
    img_h, img_w = _img.shape[:2]

    # 图像已经是 640*480 了
    # img_resize = cv2.resize(_img, SIZE, interpolation=cv2.INTER_NEAREST)

    # 高斯模糊, 抑制摄像头噪声(一些白点会被过滤掉，但影响不大, 因为他们的面积不会超过被识别物体?)
    imt_g_blur = cv2.GaussianBlur(_img, (7, 7), 0)  # 可以改成均值模糊

    # 将图像转换到 HSV 色彩空间
    frame_hsv = cv2.cvtColor(imt_g_blur, cv2.COLOR_BGR2HSV)

    # 对图像进行掩膜运算
    frame_mask = cv2.inRange(frame_hsv,
                             hsv_data[target_color]['lower'],
                             hsv_data[target_color]['upper'])

    # 红色有两个区间
    if target_color == 'red':
        frame_mask1 = cv2.inRange(frame_hsv,
                                  hsv_data['red1']['lower'],
                                  hsv_data['red1']['upper'])
        frame_mask = cv2.bitwise_or(frame_mask, frame_mask1)

    # 将图像转换到LAB空间
    # frame_lab = cv2.cvtColor(_img, cv2.COLOR_BGR2LAB)

    # 对原图像进行掩模运算, lower 和 upper 从lab_tool保存的文件中读出, 在 init()函数中被初始化;
    # frame_mask = cv2.inRange(frame_lab,
    #                          (lab_data[target_color]['min'][0],
    #                           lab_data[target_color]['min'][1],
    #                           lab_data[target_color]['min'][2]),
    #                          (lab_data[target_color]['max'][0],
    #                           lab_data[target_color]['max'][1],
    #                           lab_data[target_color]['max'][2]))

    # 开运算, 消除白点
    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 开运算, 消除小白点
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算, 闭合被识别的障碍物

    # 找出所有外轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 如果未发现任何颜色, 或者说 没有发现轮廓
    if len(contours) == 0:
        return -1, -1, -1, 0

    # 对轮廓对象按面积进行排序, 降序, 面积最大的为序号为0
    c_sorted = sorted(contours, key=cmp_area, reverse=True)

    c_0 = c_sorted[0]
    # TODO(2022.10.12 11点46分): 这里可以写一个识别轮廓形状

    # 如果面积小于 50, 认为是噪声, 进行下一次处理
    if math.fabs(cv2.contourArea(c_0)) <= 50:
        return -1, -1, -1, 0

    # 绘制识别到的颜色轮廓
    # cv2.drawContours(img_debug, c_sorted, 0, (0, 0, 255), 2)

    """
    轮廓的数据是这样的
    [
    [[369 287]]],
    [[370 287]],
    ...
    ]
    """
    # 法1: 取轮廓顶点, 这里使用最小外接矩形也是可以的
    # 一维指是那个点, 二维是冗余的? 所以[0], 三维是点坐标(x, y)
    up_x = (c_0[c_0[:, :, 1].argmin()][0])[0]  # 图像最高点
    up_y = (c_0[c_0[:, :, 1].argmin()][0])[1]

    down_x = (c_0[c_0[:, :, 1].argmax()][0])[0]
    down_y = (c_0[c_0[:, :, 1].argmax()][0])[1]  # 最低点; 相反, y argmin 是最高点;

    left_x = (c_0[c_0[:, :, 0].argmin()][0])[0]  # 最左边的点
    left_y = (c_0[c_0[:, :, 0].argmin()][0])[1]

    right_x = (c_0[c_0[:, :, 0].argmax()][0])[0]  # 最右边的点
    right_y = (c_0[c_0[:, :, 0].argmax()][0])[1]

    # 方法2: 最小外接矩形
    rect = cv2.minAreaRect(c_0)
    box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点

    # 获取矩形的对角点, 根据对角线求出中心点, 1,3 2,4 对角
    pt1_x, pt1_y = box[0, 0], box[0, 1]
    pt3_x, pt3_y = box[2, 0], box[2, 1]
    center_x, center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2  # 中心点

    theta = math.atan2(right_y - left_y, right_x - left_x)
    angle = int(math.degrees(theta))  # 带符号

    cv2.imshow('color_identify', closed)

    # 因为机器人视野被切割，所以y坐标加上250偏移
    return left_x, right_x, int(center_y + 250), angle


# 视觉主程序
def run(_img):
    global skip_st

    global obj_left_x, obj_right_x, skip, items
    global obj_center_y, obj_angle, line_centroid_x

    skip_st = True  # 默认这个项目可以跳过

    # 障碍物识别
    if skip == 1:
        obj_left_x, obj_right_x, obj_center_y, obj_angle = \
            color_identify(_img.copy(), target_color='blue')  # Hurdlers

        # 打印位置角度参数
        print('hurdles', obj_left_x, obj_right_x, obj_center_y, obj_angle)

        # 判断是否为 Hurdles 项目, y坐标大于350在(图像下方),目标长度大于100, 目标角度 < 20
        if obj_center_y >= 350 and abs(obj_left_x - obj_right_x) > 100 and abs(obj_angle) < 20:  # 准备跨栏，关闭错帧检测
            print("lock hurdles")
            items = 'hurdles'  # 当前锁定项目
            skip_st = False  # 锁定 hurdles 后不允许跳过该项目
    # 台阶
    elif skip == 2:
        obj_left_x, obj_right_x, obj_center_y, obj_angle = \
            color_identify(_img.copy(), target_color='red')

        # 打印位置角度参数
        print('stairway', obj_left_x, obj_right_x, obj_center_y, obj_angle)

        # 判断是否为 Stairway 项目
        if obj_center_y >= 350 and abs(obj_left_x - obj_right_x) > 100 and abs(obj_angle) < 20:  # 准备上台阶，关闭错帧检测
            print("lock stairway!")
            items = 'stairway'
            skip_st = False  # 同理, 锁定该项目后不允许跳过

    if skip_st:  # 设置跨栏和台阶错帧检测
        skip += 1  # 未检测到skip项目(跨栏、上下台阶),skip +1

        if skip > 2:  # 复位
            skip = 1

    # 巡线, 这里二次赋值了，不影响
    line_centroid_x = line_patrol(_img.copy())  # 原图像复制一份给，巡线函数


# 机器人动作线程
def move():
    global ret2
    global obj_center_y, line_centroid_x, items  # 障碍物y坐标; 线中心坐标; items 项目(字符串类型)
    global stair_up, skip_st  # 是否上台阶,如果为True执行上台阶动作, False执行下台阶动作; 是否错误帧
    global ready_2_pass  # 准备通过, 小碎步前进
    global perform_end  # 结束标志位
    global last_broadcast_time  # 最后语音播报时间

    # 目标y坐标 >= 350 且 目标角度 ∈ [-20, 20] 且 目标长度 > 100
    # obj_center_y >= 350 且 abs(obj_angle) < 30 且 abs(obj_left_x - obj_right_x) > 100 锁定目标, 开始调整位姿
    # [350, 470] 开始调整位姿, 先小幅度偏转, 再左右平移; 若先平移后旋转, 有角度的平移会使机器人更接近障碍物, 出现无法通过的bug;
    # 位姿与障碍物平行时, 小碎步前进, 准备通过, 此时 ready_2_pass = True;

    while True:
        # 如果图像读取失败, 等待0.5后继续, 直到图像读取成功后再移动;
        if not ret2:
            time.sleep(0.5)
            continue

        standup()  # 摔倒检测

        # switch == False 运动开关关闭, 不执行动作
        if not switch:
            continue

        # 因摄像头晃动出现的障碍物在脚下丢失的情况处理;
        # 正在小碎步准备通过, 若目标丢失, 则后退一步;
        if ready_2_pass and obj_center_y < 350:
            AGC.runAction(hk_stand)
            AGC.runActionGroup(hk_back_1_step)
            time.sleep(0.2)
            ready_2_pass = False  # 后退一步后, 默认目标丢失; 但是后退执行完成目标重现在脚下 ready_2_pass 会重新被设置为True;

        # obj_center_y 来自 主线线程, 相当于眼睛线程
        if obj_center_y >= 350 and abs(obj_angle) < 30 and abs(obj_left_x - obj_right_x) > 100:  # 检测到台阶或者跨栏,进行位置微调
            # 播报完成, 3秒后才可重新播报;
            if time.time() - last_broadcast_time >= 3:
                last_broadcast_time = time.time()  # 播报计时重置
                mp3.playNum(11)  # 发现障碍物
                time.sleep(0.1)
                mp3.loopOff()

            print("i find " + str(obj_center_y))  # debug

            if obj_center_y >= 450:  # 位置靠近，可以跨栏或者上下台阶
                mp3.playNum(12)  # 正在尝试通过障碍物
                last_broadcast_time = time.time()
                time.sleep(0.1)
                mp3.loopOff()
                t_buzzer()  # 自带 sleep

                """ 障碍物判断 """
                if items == 'hurdles':  # 跨栏
                    AGC.runAction(hk_stand)
                    AGC.runActionGroup(hk_go_forward_1_step)  # 前进一步, 微调
                    time.sleep(0.2)
                    AGC.runActionGroup('new_h')
                    time.sleep(0.2)
                    AGC.runAction(hk_stand)

                elif items == 'stairway':
                    # 第二次上台阶时 结束线程
                    if perform_end:
                        perform_end = False
                        for _ in range(5):
                            AGC.runAction(hk_stand)
                            AGC.runActionGroup(hk_back_1_step)
                        sys.exit(0)

                    if stair_up:  # 上台阶
                        AGC.runAction(hk_stand)
                        AGC.runActionGroup('climb_stairs')
                        time.sleep(0.2)
                        AGC.runAction(hk_stand)
                        stair_up = False
                    else:  # 下台阶
                        AGC.runAction(hk_stand)
                        AGC.runActionGroup(hk_go_forward_1_step)
                        time.sleep(0.2)
                        AGC.runActionGroup(hk_go_forward_1_step)
                        time.sleep(0.2)

                        AGC.runActionGroup('down_floor')
                        AGC.runAction(hk_stand)
                        time.sleep(0.2)
                        stair_up = True
                        perform_end = True

                ready_2_pass = False  # 已通过障碍物
                mp3.playNum(13)
                last_broadcast_time = time.time()
                time.sleep(0.1)
                mp3.loopOff()

            # 小幅度偏转; 能调整身位就调，调不了直接迈过去
            elif 10 < obj_angle < 20:
                AGC.runAction(hk_stand)
                AGC.runAction(hk_turn_right)
                time.sleep(0.2)
            elif -10 > obj_angle > -20:
                AGC.runAction(hk_stand)
                AGC.runAction(hk_turn_left)
                time.sleep(0.2)
            # 是否在正中间
            elif line_centroid_x - x_center >= 30:
                AGC.runAction(hk_stand)
                AGC.runAction(hk_right_move10)
                time.sleep(0.2)

            elif line_centroid_x - x_center <= -30:
                AGC.runAction(hk_stand)
                AGC.runAction(hk_left_move10)
                time.sleep(0.2)

            # 已在中心, 小幅度前进
            elif 350 <= obj_center_y < 470:  # 在中心
                ready_2_pass = True
                AGC.runAction(hk_stand)
                AGC.runAction(hk_go_forward_1_step)
                time.sleep(0.2)  # 防止摄像头晃动

        # 无障碍物巡线, 有障碍物处理障碍物
        elif line_st and line_centroid_x != -1:  # line_st巡线开关 False 不巡线, 执行巡线
            # [-20, 20] 以内, 前进
            if abs(line_centroid_x - x_center) <= 20:  # 目标x line_center_x, 设定x中心 x_center
                AGC.runAction(hk_go_forward)
                time.sleep(0.1)
            # (20, +∞) 向右转向
            elif line_centroid_x - x_center > 20:
                AGC.runAction(hk_stand)
                AGC.runAction(hk_turn_right)
                time.sleep(0.2)
            # (-∞, -20) 向左转向
            elif line_centroid_x - x_center < -20:
                AGC.runAction(hk_stand)
                AGC.runAction(hk_turn_left)
                time.sleep(0.2)


# ---------------------------------------------- 函数 END ----------------------------------------------


if __name__ == '__main__':
    AGC.runAction(hk_stand)  # 站好

    # ---------------------------------------------- ASR 词条初始化 START ----------------------------------------------
    try:
        asr = ASR.ASR()

        if True:  # debug 使用, 初始化一次后, 改为 False
            asr.eraseWords()
            asr.setMode(1)  # 1: 一直检测, 2: 关键词检测(序号1), 3: 按键检测
            asr.addWords(1, 'xiao yue')
            asr.addWords(2, 'ren lian shi bie')
            asr.addWords(3, 'tian jing biao yan')
        data = asr.getResult()
    except:
        print('传感器初始化出错')

    # ---------------------------------------------- ASR 词条初始化 END ----------------------------------------------

    """ MP3 初始化 """
    IIC_MP3_ADDR = 0x7b  # 传感器iic地址
    mp3 = MP3(IIC_MP3_ADDR)
    time.sleep(0.1)  # IIC通信空闲帧, 增加程序的稳定性

    mp3.volume(30)  # 设置音量为30，注意在播放前设置
    time.sleep(0.1)  # IIC通信空闲帧, 增加程序的稳定性

    # mp3.playNum(7)  # 播报 '程序已启动, 田径表演开始'
    # time.sleep(3)  # IIC通信空闲帧, 增加程序的稳定性

    mp3.loopOff()  # loop 状态掉电保存
    time.sleep(0.1)

    """ ASR循环检测 """
    while True:
        # break  # 调试用
        data = asr.getResult()  # 没有指令输入，默认会返回0
        print(data)
        time.sleep(0.1)  # 延时100毫秒, IIC冲突

        if data == 1:  # '唤醒词: 小悦'
            mp3.playNum(16)  # 我在, 你说
            time.sleep(1)
            mp3.loopOff()  # 每次运行完毕都需要 loop off?
        elif data:  # '其他指令'
            print('result:', data)
            mp3.playNum(15)  # 收到
            time.sleep(1)
            mp3.loopOff()
            break

        time.sleep(0.1)  # 延时100毫秒

    """ 田径任务开始 """
    # 收到之后，开始运行
    mp3.playNum(7)  # 程序已启动, 田径表演开始
    time.sleep(3)  # IIC通信空闲帧, 增加程序的稳定性
    mp3.loopOff()  # loop 状态掉电保存

    """ 从文件中读取颜色及舵机位置, 设置舵机位置 """
    init()  # 初始化
    t_buzzer()  # 初始化完成, 蜂鸣器响三下

    """ 视频流处理 """
    cap = cv2.VideoCapture(-1)

    th = threading.Thread(target=move, daemon=True).start()
    switch = True  # 动作开关, False 时 move 线程不动作, debug 用

    while True:
        # Capture frame-by-frame
        ret2, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)  # 这里显示原图像

        # Our operations on the frame come here
        run(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
