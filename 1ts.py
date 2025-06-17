import dxcam
import cv2
import winsound
import time
from ultralytics import YOLO
import pynput
import numpy as np
from pynput.mouse import Controller
from pynput.keyboard import Key, Listener
from multiprocessing import Value, Process
import PID
import torch


# ------------------------- 全局控制变量 -------------------------
lock_flag = Value('b', True)  # 自瞄开关
stop_flag = Value('b', False)  # 退出程序

# ------------------------- 键盘监听回调 -------------------------
def on_release(key):
    if key == Key.home:  # Home键切换开关
        lock_flag.value = not lock_flag.value
        status = "启动" if lock_flag.value else "关闭"
        print(f"检测循环已{status}")
        winsound.Beep(800 if lock_flag.value else 400, 200)
    elif key == Key.end:  # End键结束程序
        stop_flag.value = True
        print("程序正在退出...")
        winsound.Beep(400, 200)
        return False  # 停止监听
    return True
# ------------------------- 主检测循环 -------------------------
def detection_loop(lock_flag, stop_flag):
    mouse = Controller()
    model = YOLO("yolo11n-pose.engine")
    camera = dxcam.create(output_idx=0, output_color="BGR")
    region = (1000,650, 1560, 1000)
    camera.start(region=region, target_fps=120)
    fixed_point = torch.tensor([280, 175], 
                          device='cuda', 
                          dtype=torch.float32)  
    #pid_x = PID.PID()
    #pid_y = PID.PID()
    lasmove_x=0
    lasx = 0
    try:
        while True:
            if stop_flag.value:
                break

            if not lock_flag.value:
                continue

            frame = camera.get_latest_frame()
            if frame is None:
                continue

            # YOLOv8姿态检测逻辑
            t1= time.perf_counter()
            results = model(
                frame,
                half=True,
                conf=0.6,
                iou=0.5,
                classes=0,
                verbose=False,
                stream=False,
            )
            t2= time.perf_counter()
            for result in results:
                if result.boxes.data.shape[0] > 0:
                    centers = result.boxes.xywh[:, :2]
                    all_xywh = result.boxes.xywh[:, :]
                    distances_sq = torch.sum((centers - fixed_point) ** 2, dim=1)
                    min_index = torch.argmin(distances_sq)        # 获取距离最小的中心坐标
                    min_center = centers[min_index]
                    min_distance = torch.sqrt(distances_sq[min_index])

                    h = all_xywh[min_index][3]
                    x = min_center[0] + 1000
                    y = min_center[1] + 650 - h*0.15
                    
                    #x = result.boxes.xywh[0][0] + 1000
                    #y = result.boxes.xywh[0][1] + 650 - result.boxes.xywh[0][3] * 0.13   
                    if 1270<x<1290:
                        PID.click(1)
                        PID.move(0,5)
                        #pid_x.errSum = 0
                        #pid_x.lastErr= 0
                    #move_x = pid_x.pidPosition(x, 1280)
                    #move_y = pid_y.pidPosition(y, 800)
                    move_x = x - 1280
                    move_y = y - 800

                    p = lasx - x
                    lasx = x
                    err = lasmove_x - p
                    lasmove_x = move_x
                    if -10 < p < 10:
                        err = err *0.65
                    if move_x >= 0 and err > 0:
                        move_x = move_x - err * 0.8
                    elif move_x < 0 and err > 0:
                        move_x = move_x + err * 0.8
                    elif move_x > 0 and err < 0:
                        move_x = move_x + err * 0.8
                    elif move_x < 0 and err < 0:
                        move_x = move_x - err * 0.8
                    if 0 < move_x< 1:
                        move_x = 1
                    elif 0 > move_x > -1:  # 限制负最小值
                        move_x = -1
                    PID.move(int(move_x),int(move_y*0.2))
                    #time.sleep(0.005)
                    print(t2-t1)
            #annotated_frame = result.plot()
            #cv2.imshow("Real-time Pose Detection", annotated_frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    finally:
        camera.stop()
        cv2.destroyAllWindows()



# ------------------------- 主程序入口 -------------------------
if __name__ == "__main__":
    import time

    # 启动键盘监听线程
    keyboard_listener = Listener(on_release=on_release)
    keyboard_listener.start()

    # 启动检测循环进程（正确传递共享变量）
    detection_process = Process(
        target=detection_loop,
        args=(lock_flag, stop_flag)
    )
    detection_process.start()

    # 主线程监控
    try:
        while not stop_flag.value:
            time.sleep(1)  
    except KeyboardInterrupt:
        stop_flag.value = True

    # 清理资源
    detection_process.join(timeout=2)
    if detection_process.is_alive():
        detection_process.terminate()
    
    keyboard_listener.stop()
    print("程序已安全退出")