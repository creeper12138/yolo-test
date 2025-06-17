import dxcam
import cv2
from ultralytics import YOLO
import os
import ctypes
import pynput
from pynput.mouse import Controller
import numpy as np



try:
    root = os.path.abspath(os.path.dirname(__file__))
    driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
    ok = driver.device_open() == 1  # 该驱动每个进程可打开一个实例
    if not ok:
        print('Error, GHUB or LGS driver not found')
except FileNotFoundError:
    print(f'Error, DLL file not found')

def move(x: int, y: int):
    if (x == 0) & (y == 0):
        return
    driver.moveR(x, y, True)

def calculate_head_center(keypoints, confidence_threshold=0.6):
    """
    根据可用的关键点计算头部中心点
    :param keypoints: 关键点数据，形状 (17, 3)
    :param confidence_threshold: 置信度阈值
    :return: 头部中心点坐标 (x, y)，如果无法计算则返回 None
    """
    # 定义关键点索引
    nose_idx = 0
    left_eye_idx = 1
    right_eye_idx = 2
    left_ear_idx = 3
    right_ear_idx = 4

    # 提取关键点及其置信度
    points = []
    for idx in [nose_idx, left_eye_idx, right_eye_idx, left_ear_idx, right_ear_idx]:
        x, y, conf = keypoints[idx]
        if conf > confidence_threshold:  # 过滤低置信度关键点
            points.append((x, y))

    if len(points) < 2:  # 至少需要两个关键点
        return None

    # 计算头部中心点
    head_center = np.mean(points, axis=0)
    return head_center



# 初始化模型和摄像头
model = YOLO("yolov8s-pose.engine") # 使用姿态检测专用模型
camera = dxcam.create()           # 创建DXGI捕获器

# 设置捕获区域 (left, top, right, bottom)

region = (880, 400, 1680, 1000)  # 根据实际屏幕调整区域
camera.start(region=region, target_fps=60)  # 启动捕获
mouse = Controller()

# 实时处理循环
while True:
    frame = camera.get_latest_frame()  # 获取最新帧
    
    if frame is not None:
        # 转换为BGR格式（YOLOv8需要）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 运行姿态检测
        results = model(
            frame_bgr, 
            conf=0.7,        # 置信度阈值
            iou=0.5,         # NMS阈值
            classes=0,       # 只检测person类（COCO class 0）
            verbose=False,   # 关闭 余输出
            stream=True
				)
        for result in results:     
            if result.keypoints.conf is not None and result.keypoints.conf.shape[0] > 0:
                print(result)
                print(result.keypoints.data.shape)# 获取检测框的中心点 (x, y)
                boxes_xy = result.boxes.xywh[:, :2].cpu().numpy()  # 形状 (N, 2)# 计算每个检测框中心点到 mouse.position 的距离

                current_x, current_y = mouse.position
                fixed_point = np.array([current_x, current_y])

                distances = np.sqrt(np.sum((boxes_xy-fixed_point)**2, axis=1))  # 形状 (N,) # 找到距离最近的检测框索引
                closest_idx = np.argmin(distances) # 提取对应人的关键点数据
                closest_keypoints = result.keypoints.data[closest_idx].cpu().numpy()  # 形状 (K, 3)
                head_center = calculate_head_center(closest_keypoints)
                print(head_center)
                if head_center is not None:
                    x=head_center[0]
                    y=head_center[1]
                    current_x, current_y = mouse.position
                    x = x + 880 - current_x
                    y = y + 400 - current_y
                    if x + y > 100:
                        dx=int(x*1.5)
                        dy=int(y*1.5)
                    elif 30<x+y<100:
                        dx = int(x*0.5)
                        dy = int(y*0.5)
                    elif 30 > x+y:
                        dx = int(x*0.1)
                        dy = int(y*0.1)
                    move(dx,dy)
        # 绘制检测结果data[0]是第一个人
            annotated_frame = result.plot()
        
        # 显示处理后的帧
            cv2.imshow("Real-time Pose Detection", annotated_frame)
    
    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
camera.stop()
cv2.destroyAllWindows()