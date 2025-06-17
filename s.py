import socket
import cv2
import struct

def send_frames(host, port,frame):
    # 创建TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 将帧编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = buffer.tobytes()
            
            # 发送数据长度和数据
            data_length = len(data)
            client_socket.sendall(struct.pack("!I", data_length) + data)
            
            # 可选：接收服务器响应
            # response = client_socket.recv(1024)
    finally:
        cap.release()
        client_socket.close()

if __name__ == "__main__":
    send_frames('服务器IP', 12345)