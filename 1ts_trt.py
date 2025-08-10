
import dxcam
import cv2
import winsound
import time
import numpy as np
from pynput.mouse import Controller
from pynput.keyboard import Key, Listener
from multiprocessing import Value, Process
import PID
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# ------------------------- TRT 推理模块 -------------------------
class TrtPoseEngine:
	def __init__(self, engine_path, input_shape=(1, 3, 640, 640)):
		self.input_shape = input_shape
		self.logger = trt.Logger(trt.Logger.WARNING)
		with open(engine_path, "rb") as f:
			runtime = trt.Runtime(self.logger)
			self.engine = runtime.deserialize_cuda_engine(f.read())
		self.context = self.engine.create_execution_context()

		# Allocate memory
		self.input_size = int(np.prod(input_shape))
		self.output_shape = (1, 56, 8400)
		self.output_size = int(np.prod(self.output_shape))

		self.d_input = cuda.mem_alloc(int(self.input_size * 4))
		self.d_output = cuda.mem_alloc(int(self.output_size * 4))
		self.h_input = cuda.pagelocked_empty(self.input_size, dtype=np.float32)
		self.h_output = cuda.pagelocked_empty(self.output_size, dtype=np.float32)

		self.bindings = [int(self.d_input), int(self.d_output)]
		self.stream = cuda.Stream()

	def preprocess(self, frame):
		img = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.float32) / 255.0
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)
		return img

	def infer(self, frame):
		img_input = self.preprocess(frame)
		np.copyto(self.h_input, img_input.ravel())
		cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
		self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
		cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
		self.stream.synchronize()
		return self.h_output.reshape(56, 8400).T  # (8400, 56)

def draw_pose_points(img, kpts, conf_thres=0.3):
	for x, y, c in kpts:
		if c > conf_thres:
			cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
	return img
# ------------------------- 全局控制变量 -------------------------
lock_flag = Value('b', True)
stop_flag = Value('b', False)

# ------------------------- 键盘监听回调 -------------------------
def on_release(key):
	if key == Key.home:
		lock_flag.value = not lock_flag.value
		status = "启动" if lock_flag.value else "关闭"
		print(f"检测循环已{status}")
		winsound.Beep(800 if lock_flag.value else 400, 200)
	elif key == Key.end:
		stop_flag.value = True
		print("程序正在退出...")
		winsound.Beep(400, 200)
		return False
	return True

# ------------------------- 主检测循环 -------------------------
def detection_loop(lock_flag, stop_flag):
	mouse = Controller()
	model = TrtPoseEngine("pose_fp16.engine")
	camera = dxcam.create(output_idx=0, output_color="BGR")
	region = (960, 480, 1600, 1120)
	fixed_point = np.array([320, 320], dtype=np.float32)


	lasmove_x = 0
	lasx = 0
	x=0
	y=0
	try:
		while True:
			if stop_flag.value:
				break
			if not lock_flag.value:
				continue

			frame = camera.grab(region=region)
			if frame is None:
				continue

			t1 = time.perf_counter()
			output = model.infer(frame)
			t2 = time.perf_counter()

			mask = output[:, 4] > 0.6
			filtered = output[mask]

			if filtered.shape[0] > 0:
				dists = np.sum((filtered[:, :2] - fixed_point) ** 2, axis=1)
				min_index = np.argmin(dists)
				bbox = filtered[min_index]

				
				if filtered.shape[0] > 0:
					dists = np.sum((filtered[:, :2] - fixed_point) ** 2, axis=1)
					min_index = np.argmin(dists)
					bbox = filtered[min_index]

					kpts = bbox[5:].reshape(-1, 3)
					head_x, head_y, head_conf = kpts[0]

					if head_conf > 0.5:
						x = int(head_x + 960)
						y = int(head_y + 480)


				if 1270 < x < 1290:
					PID.click(1)
					PID.move(0, 1)

				move_x = x - 1280
				move_y = y - 800

				p = lasx - x
				lasx = x
				err = lasmove_x - p
				lasmove_x = move_x
				if -10 < p < 10:
					err = err * 0.7
				if move_x >= 0 and err > 0:
					move_x = move_x - err * 0.8
				elif move_x < 0 and err > 0:
					move_x = move_x + err * 0.8
				elif move_x > 0 and err < 0:
					move_x = move_x + err * 0.8
				elif move_x < 0 and err < 0:
					move_x = move_x - err * 0.8
				if 0 < move_x < 1:
					move_x = 1
				elif 0 > move_x > -1:
					move_x = -1
				PID.move(int(move_x), int(move_y * 0.2))
				print(f"Inference time: {t2 - t1:.3f}s")


	finally:
		camera.stop()
		cv2.destroyAllWindows()

# ------------------------- 主程序入口 -------------------------
if __name__ == "__main__":
	keyboard_listener = Listener(on_release=on_release)
	keyboard_listener.start()

	detection_process = Process(target=detection_loop, args=(lock_flag, stop_flag))
	detection_process.start()

	try:
		while not stop_flag.value:
			time.sleep(1)
	except KeyboardInterrupt:
		stop_flag.value = True

	detection_process.join(timeout=2)
	if detection_process.is_alive():
		detection_process.terminate()

	keyboard_listener.stop()
	print("程序已安全退出")
