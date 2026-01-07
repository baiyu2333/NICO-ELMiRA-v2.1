from pyrep import PyRep
import time

print("正在尝试启动仿真器...")
pr = PyRep()
# 启动仿真器 (headless=False 表示我们要看界面)
pr.launch(headless=False)
pr.start()
print("✨ 成功连接！仿真器应该已经弹出来了")

# 让它跑 10 秒
time.sleep(10)

pr.stop()
pr.shutdown()
print("测试结束")
