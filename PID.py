from pynput import mouse
import time
import os
import ctypes
import torch


try:
    root = os.path.abspath(os.path.dirname(__file__))
    driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
    ok = driver.device_open() == 1
    if not ok:
        print('Error, GHUB or LGS driver not found')
except FileNotFoundError:
    print(f'Error, DLL file not found')

def move(x: int, y: int):
    if (x == 0) & (y == 0):
        return
    driver.moveR(x, y, False)

def click(code):
            if not ok:
                return
            driver.mouse_down(code)
            driver.mouse_up(code)


def fix(move_x,x,last=0):
    zs = last - x   
    err = lasmove_x - zs  
    if move_x > 0:
        move_x= move_x - err    
    else:
        move_x = move_x + err
    last = x
    lasmove_x = move_x
    return move_x

class PID:
    """PID"""

    def __init__(self):
        """PID"""
        self.kp = torch.tensor(0.8,device='cuda')
        self.ki = torch.tensor(0.01, device='cuda')
        self.kd = torch.tensor(0.01, device='cuda')
        self.lastErr = torch.tensor(0.0, device='cuda')
        self.errSum = torch.tensor(0.0, device='cuda')


    def pidPosition(self,setValue, curValue,DIS = 1):
        """位置式 PID 输出控制量"""
        err = setValue - curValue  # 计算差值, 作为比例项
        dErr = err - self.lastErr  # 计算近两次的差值, 作为微分项
        self.errSum += err  # 累加这一次差值,作为积分项
        outPID = (DIS * err * self.kp) + (self.ki * self.errSum) + (self.kd * dErr)  # PID
        self.lastErr = err  # 保存这一次差值,作为下一次的上一次差值
        return outPID  # 输出



def PIDMoveTo(target_x, target_y, xishu=1,min_x=1, min_y=1):
    pid_x = PID()
    pid_y = PID()
    cnt = 0
    while True:
        if abs(1280 - target_x)<10:
            driver.mouse_down(1)
            driver.mouse_up(1)
            break
        move_x = int(pid_x.pidPosition(target_x, 1280))
        move_y = int(pid_y.pidPosition(target_y, 800))
        if 100 < move_x:
            move_x = 100
        elif 0 > move_x > -min_x:  # 限制负最小值
            move_x = -min_x
        if 0 < move_y < min_y:
            move_y = min_y
        elif 0 > move_y > -min_y:
            move_y = -min_y
        move_x = move_x * xishu
        move_y = move_y * xishu
        move(int(move_x), int(move_y))
        target_x = target_x - move_x
        target_y = target_y - move_y
        cnt += 1
        #print(target_x, target_y, cnt)
        time.sleep(0.3)

def moveTo(target_x, target_y):
    control = mouse.Controller()
    now_x, now_y = control.position
    while True:
        if now_x == target_x and now_y == target_y:
            break
        moveX = target_x - now_x
        moveY = target_y - now_y
        move(moveX, moveY)

if __name__ == '__main__':
    target_y = 1000
    target_x = 2200
    PIDMoveTo(target_x, target_y)
    # moveTo(target_x, target_y)