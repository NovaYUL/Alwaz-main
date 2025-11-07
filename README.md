## 项目概述

这是一个基于 MediaPipe Pose + OpenCV 的“老人/摔倒监测”示例工程，提供带 GUI 和不带 GUI 的演示，用关键点角度与位置阈值判断是否“安全/摔倒/悬挂/行走”等状态。

## 主要文件与职责

- `main.py`
  - 带 Tkinter GUI 的主程序（窗口、按钮、视频显示区）。
  - 支持从本地选择视频或使用摄像头实时输入。
  - 使用 MediaPipe Pose 提取人体关键点，调用 `calculate_angle` 计算角度，并基于一组硬编码阈值判断用户状态（safe / fallen / hanging / trying to walk / barely walking）。
  - 在 OpenCV 窗口与 Tkinter 界面中同时显示帧、骨架与状态文字。

- `python code with GUI.py`
  - 功能上与 `main.py` 非常相似（也是带 GUI 的演示），代码结构略有差别。
  - 同样使用 MediaPipe 提取关键点、基于角度与关键点位置判断状态，并在 GUI/窗口显示。
  - 该文件在本次修改中已被清理并加入注释，仍然仅为演示用途，阈值为经验值。

- `python code without GUI.py`
  - 无 GUI 的处理脚本，直接读取本地视频（默认 `test.mp4`，若没有会尝试 `patient.mp4`），在每帧上绘制骨架与状态并在控制台打印关键点数据（便于调试/离线分析）。
  - 适合在命令行/服务器上运行或用于快速调试判定逻辑。

另外，本工程还包含两个实用脚本（位于 `tools/`）：
- `tools/export_landmarks.py`：把视频中每帧的 MediaPipe 关键点导出为 CSV，方便标注与离线训练。
- `tools/train_frame_classifier.py`：示例性的帧级训练脚本（RandomForest），演示如何用导出的关键点训练简单的帧分类器。

## 核心逻辑简介

- 关键函数：`calculate_angle(a, b, c)` —— 计算以点 `b` 为顶点的 `a-b-c` 的夹角（使用 atan2）。
- 常用关键点（示例）：左右眼、左右臀部、左右脚跟、左右食指（`RIGHT_INDEX` / `LEFT_INDEX`）等。
- 判定方法（演示/经验阈值）：
  - 先根据左右眼/臀/跟的归一化 x、y 值区间判断是否处于“安全”姿态。
  - 若不满足“安全”条件，则根据左右侧角度（例如胯部或躯干角度）与手指位置判断是否属于“悬挂”“摔倒”或“尝试走动”“艰难行走”等状态。
  - 这些阈值为经验值，适合快速演示；若用于生产需用标注数据重新训练/调优。

## 可视化与输出

- 在每帧上绘制：MediaPipe 骨架、若干关键角度数值、状态文字与状态框。
- GUI 版本：OpenCV 窗口 + Tkinter 界面（将帧缩放后放入 Tkinter 的 `Label` 中）。
- 非 GUI 版本：在控制台打印关键点与状态，便于记录/分析。

## 演示方法（在你的工作区快速运行）

在项目根目录下运行 `main.py`（演示会尝试优先使用项目根的 `test.mp4`，否则回退到摄像头）：

```powershell
cd "c:\Users\gslenovo\Desktop\Vscode Files\Alwaz-main"; python -c "import sys; print('PYTHON:', sys.executable); print(sys.version)"; python .\main.py
```

界面说明：
- 点击 `TEST` 按钮：播放项目根的 `test.mp4`（一次性播放，播放结束后停在最后一帧以便观察）。
- 点击 `LIVE` 按钮：切换并使用计算机摄像头实时输入。
- 点击 `VIDEO`（或“选择视频”）：选择其它本地视频文件进行检测。

## 当前运行状态与下一步

- 我已在你的机器上启动过 `main.py`（使用 Python 可执行路径：`D:\Profession software\python.exe`，版本 3.12.3）。
- 终端输出显示 MediaPipe / TFLite 已开始初始化（包含若干警告或信息日志，这些通常为初始化正常现象）。
- 若 `test.mp4` 在项目根且可读取，GUI 窗口会打开并开始播放；播放结束时会停在最后一帧。你可以：
  - 点击 `TEST` 开始播放（若未自动开始）；
  - 或点击 `LIVE` 切换到摄像头；
  - 或点击 `VIDEO` 选择其它文件。

如果你看到 OpenCV 窗口与 Tkinter 界面，说明演示已成功启动；若遇到错误或黑屏，请把终端输出或窗口截图发给我，我会继续协助排查。

## 常见故障与排查建议

- 出现 `ModuleNotFoundError: No module named 'mediapipe'`：请确保在运行程序的 Python 环境中已安装依赖（示例）：
  ```powershell
  pip install -r requirements.txt
  ```
- 如果 GUI 无法弹出或显示黑屏：确认程序是否在本地桌面会话中运行（远程/无头环境会导致无法显示 GUI），或试着运行非 GUI 脚本 `python "python code without GUI.py" test.mp4` 来验证处理逻辑是否正常。
- 需要更高的判定准确率：请使用 `tools/export_landmarks.py` 导出关键点并标注，使用 `tools/train_frame_classifier.py` 训练更好的模型替换阈值规则。


