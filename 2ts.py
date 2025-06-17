import dxcam
import cv2
import winsound
import time
import pynput
from pynput.mouse import Controller
from pynput.keyboard import Key, Listener
from multiprocessing import Value, Process
import socket
import struct
import PID

class FrameSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self._connect()  # 初始化时立即连接

    def _connect(self):
        """建立TCP连接（含自动重试）"""
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用Nagle算法
                self.socket.connect((self.host, self.port))
                print("成功连接服务器")
                return
            except ConnectionRefusedError:
                print("服务器未就绪，5秒后重试...")
                time.sleep(5)

    def send_frame(self, frame):
        """同步发送帧并返回服务器响应"""
        try:
            # 压缩帧数据
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = buffer.tobytes()
            
            # 发送数据头+内容
            header = struct.pack("!I", len(data))
            self.socket.sendall(header + data)
            
            # 接收服务器响应头
            response_header = self._receive_all(4)
            if not response_header:
                raise ConnectionError("连接已断开")
            
            # 解析响应长度并接收数据
            response_len = struct.unpack("!I", response_header)[0]
            response_data = self._receive_all(response_len)
            x, y,c = struct.unpack("!iii", response_data)  # 直接解包两个int32
            return x, y,c  # 返回元组形式的坐标            
        except (ConnectionResetError, BrokenPipeError):
            print("连接异常，尝试重连...")
            self._reconnect()
            return None

    def _receive_all(self, n):
        """可靠接收指定长度的数据"""
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _reconnect(self):
        """异常后重新连接"""
        self.socket.close()
        self._connect()

    def close(self):
        """主动关闭连接"""
        self.socket.close()

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
    camera = dxcam.create(output_idx=0, output_color="BGR")
    region = (1000,650, 1560, 1000)
    camera.start(region=region, target_fps=60)
    sender = FrameSender('10.100.15.134', 12345)  # 初始化发送器
    try:
        while True:
            if stop_flag.value:
                break

            if not lock_flag.value:
                continue
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            t1= time.perf_counter()
            server_response = sender.send_frame(frame)
            t2= time.perf_counter()
            print(t2-t1)
            if server_response is not None:
                move_x,move_y,c = server_response
            if c == 1:
                PID.click(1)
                PID.move(0,5)
            PID.move(int(move_x),int(move_y*0.2))
            cv2.imshow("Real-time Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
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