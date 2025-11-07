import cv2
import mediapipe as mp
import numpy as np
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk

# 带 GUI 的姿态检测示例（清理后的完整版本）
# - 基于 MediaPipe Pose 提取人体关键点
# - 使用 tkinter 显示视频帧（支持本地视频与摄像头）


# --- 界面初始化 ---
root = Tk()
root.geometry("1920x1080+0+0")
root.state("zoomed")
root.config(bg="#3a3b3c")
root.title("Eldering Monitring")


def path_select():
    """弹出文件选择并打开视频（设置全局 cap）。"""
    global video_path, cap
    video_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(video_path)
    text = Label(root, text="Recorded Video  ", bg="#3a3b3c", fg="#ffffff", font=("Calibri", 20))
    text.place(x=250, y=150)


def video_live():
    """切换到摄像头输入（video_path=0）。"""
    global video_path, cap
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    text = Label(root, text="Live Video Feed", bg="#3a3b3c", fg="#ffffff", font=("Calibri", 20))
    text.place(x=250, y=150)


live_btn = Button(root, height=1, text='LIVE', width=8, fg='magenta', font=("Calibri", 14, "bold"), command=video_live)
live_btn.place(x=1200, y=20)
text = Label(root, text="  For Live Video", bg="#3a3b3c", fg="#ffffff", font=("Calibri", 20))
text.place(x=1000, y=30)

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
    """计算 a-b-c 的夹角（以 b 为顶点）。"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
video_path = 0
cap = cv2.VideoCapture(video_path)

counter = 0
stage = None


## 使用 MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 如果帧读取失败，回到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            root.update()
            continue

        # 转为 RGB 给 MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # 提取关键点（归一化坐标）
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

            angle1 = calculate_angle(left_eye, left_hip, left_heel)
            angle2 = calculate_angle(right_eye, right_hip, right_heel)

            # 在图像上显示角度
            cv2.putText(image, str(angle1), tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(angle2), tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # 根据关键点阈值判断状态（演示用途）
            if ((left_eye[0] >= 0.41 and left_eye[0] <= 0.43) and (left_hip[0] >= 0.44 and left_hip[0] <= 0.46) and (left_heel[0] >= 0.41 and left_heel[0] <= 0.43) or (right_eye[0] >= 0.41 and right_eye[0] <= 0.43) and (right_hip[0] <= 0.43 and right_hip[0] >= 0.41) and (right_heel[0] >= 0.37 and right_heel[0] <= 0.39)):

                if ((left_eye[1] >= 0.24 and left_eye[1] <= 0.33) and (left_hip[1] <= 0.35 and left_hip[1] >= 0.45) and (left_heel[1] <= 0.74 and left_heel[1] >= 0.72) or (right_eye[1] <= 0.30 and right_eye[1] >= 0.24) and (right_hip[1] <= 0.50 and right_hip[1] >= 0.32) and (right_heel[1] >= 0.71 and right_heel[0] <= 0.73)):
                    stage = "safe :)"
            else:
                if angle1 != angle2 and (angle1 > 170 and angle2 > 170):
                    if (((right_index[0] < 0.70 and right_index[0] > 0.20) and (right_index[1] < 0.56 and right_index[1] > 0.15)) or ((left_index[0] < 0.55 and left_index[0] > 0.18) and (left_index[1] < 0.56 and left_index[1] > 0.15))):
                        stage = "Hanging on !!"
                    else:
                        stage = "fallen :("

                elif angle1 != angle2 and (angle1 < 140 or angle2 < 140):
                    stage = "Trying to Walk"
                elif angle1 != angle2 and ((angle1 < 168 and angle1 > 140) and (angle2 < 168 and angle2 > 140)):
                    stage = "Barely Walking"
                else:
                    pass

        except Exception:
            pass

        # 绘制状态框和文字
        cv2.rectangle(image, (0, 0), (350, 125), (245, 117, 16), -1)
        cv2.putText(image, 'Condition: ', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str("Calculating Angles"), (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, 'STAGE: ', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 绘制骨架
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将帧显示在 Tkinter 中（缩放以适配 GUI）
        # 使用 PIL 的 resize + LANCZOS 抗锯齿缩放后转换为 ImageTk
        pil_img = Image.fromarray(image).resize((700, 450), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        Video_Label["image"] = tk_img

        # 按 q 键可以退出循环
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # 更新 Tkinter 主循环（保持 GUI 响应）
        root.update()

    cap.release()
    cv2.destroyAllWindows()
