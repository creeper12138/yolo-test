# 🎯 YOLO 自动瞄准系统（自瞄辅助）

本项目是一个基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 和 `dxcam` 的实时目标检测与自瞄辅助系统，支持通过键盘控制开启/关闭以及退出程序。适用于需要高帧率、低延迟的视觉检测任务，如游戏辅助、自定义姿态识别应用等。

## ✨ 功能概览

* 🔍 **目标检测**：使用 YOLOv8 pose 模型 (`.engine`) 进行人体关键点检测
* 🎯 **自动瞄准**：对距离屏幕中心最近的目标进行追踪并移动鼠标指针
* 🖱 **自动点击**：当目标进入预设区域范围内自动执行点击操作
* 🎛 **PID-like 平滑控制**：缓解高频检测导致的鼠标抖动，实现更平滑的移动
* ⌨️ **快捷键控制**：

  * `Home`：启动/关闭自瞄
  * `End`：退出程序

## 📷 检测区域设置

通过 `dxcam` 采集部分屏幕区域（默认分辨率为 2560x1440）：

```python
region = (1000, 650, 1560, 1000)  # 左上角 (x, y), 右下角 (x, y)
```

## 🔧 依赖安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容参考：

```txt
ultralytics
dxcam
opencv-python
pynput
torch
numpy
```

此外，需要安装 DirectX Runtime 以确保 `dxcam` 正常工作。

## 🚀 使用方式

1. 替换模型文件：
   将你自己的 YOLOv8 TensorRT 引擎模型放入项目目录中，并命名为 `yolo11n-pose.engine`，或修改代码中模型路径。

2. 启动程序：

   ```bash
   python main.py
   ```

3. 操作控制：

   * 按下 `Home` 键启动或暂停自动检测
   * 按下 `End` 键终止程序

## 🧠 工作原理简要说明

* 使用 `dxcam` 捕获设定屏幕区域图像；
* 通过 YOLOv8 模型进行人体姿态检测；
* 选取距离屏幕中心最近的目标；
* 计算其位置偏移量，使用“惯性/差分”方式控制鼠标平滑移动；
* 检测到目标进入中央区域时触发点击操作。

## 🔒 安全与合法性声明

本项目仅用于学习与研究目的，**禁止用于任何违反平台或游戏规则的行为**。作者不对任何滥用造成的后果承担责任。

## 📎 项目结构（关键部分）

```
├── main.py               # 主程序逻辑
├── PID.py                # 鼠标控制与点击逻辑模块
├── yolo11n-pose.engine   # TensorRT 编译后的 YOLOv8 模型文件
└── README.md             # 项目说明
```

## 💬 参考组件

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [dxcam](https://github.com/Prayag2/dxcam)
* [pynput](https://pynput.readthedocs.io/en/latest/)
* [Torch (CUDA)](https://pytorch.org/)

---
哈上面都是AI写的，这个东西是参考https://blog.csdn.net/mrathena/article/details/126860226 做的 1ts.py是主要版本 2ts.py是一个联网版本，就是计算放在其他电脑上 ，3ts.py和1差不多，只不过是直接锁头的，但是需要计算的就更多，1里面锁身体用算法来间接所头
S.PY 是客户端，T.PY是服务器端 你说怎么没有t.py 放服务器上了 反正就是把模型运算的部分拆出来 大概就是这样 我的配置是4060的笔记本 每个循环耗时大概在30-40ms。还有什么问题呢，在dxcam.py 里面
 if frame is not None:
                    with self.__lock:
                        # Reconstruct frame buffer when resolution change
                        if frame.shape[0] != self.height:
                            region = (1000,650, 1560, 1000)
                            frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
                            self.__frame_buffer = np.ndarray(
                                (self.max_buffer_len, *frame_shape), dtype=np.uint8
                            )
                        self.__frame_buffer[self.__head] = frame
这个地方这么改，不然会报错，我记得是帧溢出？改了有时候也会报错，不知道为什么，知道的可以说说
PID默认没有开，整个过程耗时间最多的是具体移动鼠标的部分，
虽然我是菜鸟，但是还算是解决了一个问题, 我参考的博主在最后说会产生震荡现象，但我推迟原因是程序的驱动函数move实际上发给驱动一个信号，然后驱动控制鼠标移动，平常这个移动时间很小可以忽略，但是如果运行的速度够快，就会导致在鼠标上一次命令没有执行完，还在移动的过程中，程序又一次来到这个地方，然后计算目标位置，再次发出指令。这里是异步的，鼠标驱动的逻辑应该是队列指令，最终导致原本是A移动到B，变成A-C-B的指令，第一个指令A-B加上C-B的距离也就超过了A-B的原始路程，性能越好速度越快震荡也就越严重
这里的代码就是解决这个问题的，不是完美解决，这东西会有多少人看呢
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

Here's a full English version of your `README.md`, including both the polished formal documentation and your detailed developer notes, preserving all your technical insight:

---

# 🎯 YOLO Auto-Aim System (Aimbot Assistant)

This project is a real-time auto-aiming assistant built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and `dxcam`, supporting keyboard control to toggle detection or exit the program. It is designed for high-FPS, low-latency scenarios like game assistance or custom pose detection.

## ✨ Features Overview

* 🔍 **Object Detection**: Uses YOLOv8 pose model (`.engine`) for human keypoint detection
* 🎯 **Auto Aim**: Tracks the closest target to the screen center and moves the mouse pointer accordingly
* 🖱 **Auto Click**: Automatically triggers mouse clicks when the target enters a specified region
* 🎛 **PID-like Smooth Control**: Mitigates jitter from high-frequency detections, ensures smoother motion
* ⌨️ **Keyboard Shortcuts**:

  * `Home`: Toggle auto-detection on/off
  * `End`: Exit the program

## 📷 Screen Capture Region

Uses `dxcam` to capture a portion of the screen (default for 2560×1440 resolution):

```python
region = (1000, 650, 1560, 1000)  # Top-left (x, y) to bottom-right (x, y)
```

## 🔧 Dependencies

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
ultralytics
dxcam
opencv-python
pynput
torch
numpy
```

Also make sure [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35) is installed for `dxcam` to work correctly.

## 🚀 How to Use

1. **Add your model**:
   Put your custom YOLOv8 TensorRT engine file in the project root, named `yolo11n-pose.engine`. Modify the path in `main.py` if using a different name.

2. **Run the program**:

   ```bash
   python main.py
   ```

3. **Control**:

   * Press `Home` to toggle detection
   * Press `End` to stop the program

## 🧠 Behind the Scenes

* Uses `dxcam` to continuously grab frames from a defined screen region
* Feeds images into YOLOv8 pose model for human keypoint detection
* Selects the closest target to screen center
* Calculates movement offsets and applies smoothed movement via mouse controller
* Triggers clicks when the target is within a defined "headshot zone"

## 🧪 Developer Notes & Tips

This project is heavily inspired by [this blog post (CSDN)](https://blog.csdn.net/mrathena/article/details/126860226). Here's how my version expands on it:

### 🗃 Versions

* `1ts.py`: main version – detects body and indirectly locks head
* `2ts.py`: offloaded (networked) version – sends frames to another PC for inference
* `3ts.py`: similar to `1ts`, but directly locks onto the head (heavier computation)
* `S.py`: client code (for networked version)
* `T.py`: server code (runs on the remote machine with model, not included here for security)

### ⚙ Hardware

* GPU: Laptop with RTX 4060
* Inference loop time: \~30–40ms per cycle
* Most time-consuming step: moving the mouse (due to system-level driver delay)

### ⚠ `dxcam.py` Patch

To avoid "frame overflow" errors, I modified the frame shape reset in `dxcam.py` like this:

```python
if frame is not None:
	with self.__lock:
		# Reconstruct frame buffer when resolution change
		if frame.shape[0] != self.height:
			region = (1000,650, 1560, 1000)
			frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
			self.__frame_buffer = np.ndarray(
				(self.max_buffer_len, *frame_shape), dtype=np.uint8
		 )
		self.__frame_buffer[self.__head] = frame
```

This prevents crash in most cases, but it still sometimes fails. Possibly related to buffer size or async grabbing bugs – contributions welcome.

### 🧩 Why it Jitters (and How I Solved It)

The original author mentioned “oscillation” at high FPS. I discovered it's not just control error – it's because:

> The mouse movement is **asynchronous**. When you send move A→B, the cursor might still be moving from A when your loop sends another command C→B. The system queues them up (likely in the driver), causing A→C→B pattern, overshooting your target.

This bug is *worse on faster systems*, because the loop updates more quickly than the mouse can physically move.

I applied a lightweight prediction correction to reduce this:

```python
p = lasx - x
lasx = x
err = lasmove_x - p
lasmove_x = move_x
if -10 < p < 10:
	err = err * 0.65
if move_x >= 0 and err > 0:
	move_x -= err * 0.8
elif move_x < 0 and err > 0:
	move_x += err * 0.8
elif move_x > 0 and err < 0:
	move_x += err * 0.8
elif move_x < 0 and err < 0:
	move_x -= err * 0.8
if 0 < move_x < 1:
	move_x = 1
elif 0 > move_x > -1:
	move_x = -1
```

It’s not perfect but helps *a lot*. Feel free to improve it.

### 🤔 Who Will Use This?

Maybe only a few hobbyists – but even if you're a beginner like me, solving this kind of practical problem is rewarding.

## 🔒 Legal Disclaimer

This tool is for **educational and research use only**.
**Do not** use it to violate any game rules, platform terms, or ethical boundaries.
The author is not responsible for misuse.

## 📂 Project Structure

```
├── main.py               # Core detection logic
├── PID.py                # Mouse movement + click logic
├── yolo11n-pose.engine   # Your TensorRT YOLOv8 model
├── S.py / T.py           # Networked inference version (client/server)
├── 1ts.py / 2ts.py / 3ts.py  # Different versions with slight behavioral tweaks
└── README.md             # This file
```

## 💬 References & Components

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [dxcam](https://github.com/Prayag2/dxcam)
* [pynput](https://pynput.readthedocs.io/en/latest/)
* [Torch (CUDA)](https://pytorch.org)

---

Let me know if you'd like this split into multiple language versions (`README.md` + `README.zh.md`) or if you'd like an auto-generated `requirements.txt` and `dxcam` patch template.
