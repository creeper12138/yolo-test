from collections import defaultdict
 
import cv2
import numpy as np
 
from ultralytics import YOLO
 
# 加载YOLOv8模型
model = YOLO('best.engine')
 
# 打开视频文件
video_path = "test_track.mp4"
cap = cv2.VideoCapture(video_path)
 
# 存储追踪历史
track_history = defaultdict(lambda: [])
 
# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()
 
    if success:
        # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
        results = model.track(frame, persist=True)
 
        # 获取框和追踪ID
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
 
        # 在帧上展示结果
        annotated_frame = results[0].plot()
 
        # 绘制追踪路径
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y中心点
            if len(track) > 30:  # 在90帧中保留90个追踪点
                track.pop(0)
 
            # 绘制追踪线
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
 
        # 展示带注释的帧
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
 
        # 如果按下'q'则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则退出循环
        break
 
# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()