import cv2
import mediapipe as mp
import numpy as np
from tkinter import filedialog

#from HandTrackingModule import HandDetector

from tkinter import *
from PIL import Image, ImageTk


# --- Tkinter GUI 初始化 ---
root = Tk()
root.geometry("1920x1080+0+0")
root.state("zoomed")  # 窗口最大化
root.config(bg="#3a3b3c")
root.title("Eldering Monitring")

# 状态栏：用于显示持久状态信息（播放、错误、完成等）
status_label = Label(root, text="", bg="#3a3b3c", fg="#ffffff", font=("Calibri", 18))
status_label.place(x=250, y=110)


def path_select():
    """弹出文件选择对话框并打开所选视频文件。将视频句柄保存在全局 cap 中。"""
    global video_path, cap
    video_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(video_path)
    status_label.config(text="Recorded Video")
    # 启用/禁用按钮状态恢复以确保可交互
    live_btn.config(state=NORMAL)
    browse_btn.config(state=NORMAL)
    test_btn.config(state=NORMAL)


def video_live():
    """切换到摄像头直播（video_path=0）。"""
    global video_path, cap
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    status_label.config(text="Live Video Feed")
    live_btn.config(state=NORMAL)
    browse_btn.config(state=NORMAL)
    test_btn.config(state=NORMAL)


def play_test_video():
    """在 GUI 中播放项目根目录下的 test.mp4（如果存在），并切换全局 cap。"""
    global video_path, cap, project_root
    test_path = os.path.join(project_root, 'test.mp4')
    if not os.path.exists(test_path):
        # 在 GUI 中提示找不到文件
        status_label.config(text="test.mp4 not found", fg="#ff5555")
        return
    try:
        # 释放之前的句柄（如果有）
        if cap is not None:
            cap.release()
    except Exception:
        pass
    video_path = test_path
    cap = cv2.VideoCapture(video_path)
    # 标记为测试播放模式：播放一次，不循环；并在播放时禁用其他按钮
    global playing_test
    playing_test = True
    status_label.config(text="Playing test.mp4 ...", fg="#ffffff")
    # 禁用其它按钮，避免在播放时冲突
    live_btn.config(state=DISABLED)
    browse_btn.config(state=DISABLED)
    test_btn.config(state=DISABLED)


# GUI 元素：直播/浏览按钮与文字说明
live_btn = Button(root, height=1, text='LIVE', width=8, fg='magenta', font=("Calibri", 14, "bold"), command=video_live)
live_btn.place(x=1200, y=20)
text = Label(root, text="  For Live Video", bg="#3a3b3c", fg="#ffffff", font=("Calibri", 20))
text.place(x=1000, y=30)

# 专用按钮：播放项目根目录下的 test.mp4（如果存在）
test_btn = Button(root, height=1, text='TEST', width=8, fg='magenta', font=("Calibri", 14, "bold"), command=play_test_video)
test_btn.place(x=1300, y=20)

browse_btn = Button(root, height=1, width=8, text='VIDEO', fg='magenta', font=("Calibri", 14, "bold"), command=path_select)
browse_btn.place(x=1200, y=90)
text = Label(root, text="To Browse Video", bg="#3a3b3c", fg="#ffffff", font=("Calibri", 20))
text.place(x=1000, y=90)


ttl = Label(root, text="ELDERING MONITERING ", bg="#4f4d4a", fg="#fffbbb", font=("Calibri", 40))
ttl.place(x=100, y=50)

Video_frame = Frame(root, height=720, width=1080, bg="#3a3b3c")
Video_Label = Label(root)
Video_frame.place(x=15, y=200)
Video_Label.place(x=15, y=200)


def calculate_angle(a, b, c):
    """计算由点 a-b-c 形成的夹角（以 b 为顶点），返回角度（度）。

    参数 a,b,c 为 (x,y) 坐标列表（一般为归一化坐标，范围 0-1）。
    使用 np.arctan2 计算角度差并转换为度，并确保角度范围在 [0,180]。
    """
    a = np.array(a)  # 第一个点
    b = np.array(b)  # 中心点
    c = np.array(c)  # 末端点

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# MediaPipe 和全局视频句柄初始化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# 默认使用项目根的 test.mp4 作为输入视频（可通过界面选择其它视频或切换到摄像头）
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
default_video = os.path.join(project_root, 'test.mp4')
video_path = default_video if os.path.exists(default_video) else 0
cap = cv2.VideoCapture(video_path)

# 状态相关变量
counter = 0
stage = None
# 播放模式标志：当按下 TEST 时设置为 True，表示正在以“只播放一次”的方式播放 test.mp4
playing_test = False
# 暂停显示（用于播放结束后停在最后一帧）
paused = False
paused_image = None
last_image = None


## 使用 MediaPipe Pose：检测姿态并在 GUI 中显示结果
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # 如果处于 paused 状态（test 播放结束），直接显示缓存的最后一帧，不再读取视频流
        if paused and paused_image is not None:
            try:
                cv2.imshow('Mediapipe Feed', paused_image)
                # 更新 tkinter 中的图像显示
                img_rgb = cv2.cvtColor(paused_image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb).resize((700, 450), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
                Video_Label["image"] = tk_img
            except Exception:
                pass
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            root.update()
            continue

        ret, frame = cap.read()
        if ret:
            # 每帧保存为 last_image，当播放结束要停在最后一帧时使用
            last_image = frame.copy()
            # 将 BGR 转为 RGB 供 MediaPipe 使用
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 运行姿态检测
            results = pose.process(image)

            # 转回 BGR 以便 OpenCV 显示
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 提取关键点并计算逻辑判断
            try:
                landmarks = results.pose_landmarks.landmark

                # 选择一组关心的关键点（归一化坐标）
                left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]

                # 计算左右侧的角度（以眼-臀-跟为示例）
                angle1 = calculate_angle(left_eye, left_hip, left_heel)
                angle2 = calculate_angle(right_eye, right_hip, right_heel)

                # 在图像上可视化角度值（将归一化坐标乘以大致帧大小）
                cv2.putText(image, str(angle1),
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(angle2),
                            tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # 基于关键点位置与角度做简单状态判断（阈值为经验值）
                # 注意：这些阈值是硬编码的演示值，实际部署需调参或采用更鲁棒的判决方法
                if ((left_eye[0] >= 0.41 and left_eye[0] <= 0.43) and
                        (left_hip[0] >= 0.44 and left_hip[0] <= 0.46) and
                        (left_heel[0] >= 0.41 and left_heel[0] <= 0.43) or
                        (right_eye[0] >= 0.41 and right_eye[0] <= 0.43) and
                        (right_hip[0] <= 0.43 and right_hip[0] >= 0.41) and
                        (right_heel[0] >= 0.37 and right_heel[0] <= 0.39)):

                    if ((left_eye[1] >= 0.24 and left_eye[1] <= 0.33) and
                            (left_hip[1] <= 0.35 and left_hip[1] >= 0.45) and
                            (left_heel[1] <= 0.74 and left_heel[1] >= 0.72) or
                            (right_eye[1] <= 0.30 and right_eye[1] >= 0.24) and
                            (right_hip[1] <= 0.50 and right_hip[1] >= 0.32) and
                            (right_heel[1] >= 0.71 and right_heel[0] <= 0.73)):
                        stage = "safe :)"
                else:
                    # 根据角度差与手指接近程度区分 "悬挂"、"摔倒"、"行走" 等情况
                    if angle1 != angle2 and (angle1 > 170 and angle2 > 170):
                        if (((right_index[0] < 0.70 and right_index[0] > 0.20) and
                             (right_index[1] < 0.56 and right_index[1] > 0.15)) or
                                ((left_index[0] < 0.55 and left_index[0] > 0.18) and
                                 (left_index[1] < 0.56 and left_index[1] > 0.15))):
                            stage = "Hanging on !!"
                        else:
                            stage = "fallen :("

                    elif angle1 != angle2 and (angle1 < 140 or angle2 < 140):
                        stage = "Trying to Walk"
                    elif angle1 != angle2 and ((140 < angle1 < 168) and (140 < angle2 < 168)):
                        stage = "Barely Walking"
                    else:
                        pass

            except Exception:
                # 捕获任何异常（例如 results.pose_landmarks 为 None）并忽略，继续下一帧
                pass

            # 在图像左上角绘制状态框与文字
            cv2.rectangle(image, (0, 0), (350, 125), (245, 117, 16), -1)
            cv2.putText(image, 'Condition: ', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str("Calculating Angles"), (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'STAGE: ', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            # 绘制 MediaPipe 的骨架检测结果
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # OpenCV 窗口显示
            cv2.imshow('Mediapipe Feed', image)

            # 同时把帧转换为 PIL Image 再显示到 Tkinter 的 Label（缩放以适配 GUI）
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(Image.fromarray(image).resize((700, 450), Image.Resampling.LANCZOS))
            Video_Label["image"] = image

            # 按 q 键可以退出循环
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            # 处理帧读取失败（例如文件播放结束或摄像头断流）
            # 如果当前是由 TEST 按钮触发的一次性播放，则停止该播放并恢复摄像头
            try:
                is_file_playback = isinstance(video_path, str) and os.path.exists(video_path)
            except Exception:
                is_file_playback = False

            if playing_test and is_file_playback:
                # 测试视频播放完毕：停在最后一帧（paused），不再循环或切换到摄像头；恢复按钮
                try:
                    paused_image = last_image.copy() if last_image is not None else None
                except Exception:
                    paused_image = last_image
                playing_test = False
                paused = True
                status_label.config(text="test.mp4 playback completed (paused on last frame)", fg="#bbffbb")
                # 恢复按钮可用（允许用户选择其他操作）
                live_btn.config(state=NORMAL)
                browse_btn.config(state=NORMAL)
                test_btn.config(state=NORMAL)
            else:
                # 非测试播放或摄像头断流：循环播放（回到起始帧）以保持演示连续性
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 更新 Tkinter 主循环（使 GUI 响应）
        root.update()

    cap.release()
    cv2.destroyAllWindows()



