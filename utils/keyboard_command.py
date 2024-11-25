import threading
import signal
import sys

# class 功能：在后台监听用户的键盘输入，并处理 Ctrl+C 信号以优雅地退出程序
class CommandThread(threading.Thread):
    def __init__(self, keyboard_signal):
        super(CommandThread, self).__init__()
        self.keyboard_signal = keyboard_signal
        self.running = True
        # 注册信号处理器，当接收到 SIGINT（通常是 Ctrl+C）信号时，调用 signal_handler 方法
        signal.signal(signal.SIGINT, self.signal_handler)

    # 线程在 self.running 为 True 时持续运行
    def run(self):
        while self.running:
            if sys.stdin.isatty():
                keyboard_info = sys.stdin.readline().strip()
                if keyboard_info:
                    self.keyboard_signal.append(keyboard_info)

                if len(self.keyboard_signal) > 1:
                    self.keyboard_signal.pop(0)

    # 退出程序
    def signal_handler(self, sig, frame):
        self.running = False     # 使线程的 while 循环终止
        print('Ctrl+C detected, exiting...')
        sys.exit(0)