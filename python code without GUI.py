import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    """计算 a-b-c 的夹角（以 b 为顶点），返回角度值（度）。

    输入通常为归一化坐标（x,y），返回值范围在 0-180。
    """
    a = np.array(a)  # 第一个点
    b = np.array(b)  # 中心点
    c = np.array(c)  # 末端点

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# MediaPipe 绘图工具和姿态模型
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 默认使用名为 'patient.mp4' 的视频文件；可按需修改为其他文件路径或摄像头索引
    # 默认使用项目根的 test.mp4 作为输入视频（可修改为其他文件或摄像头索引）
cap = cv2.VideoCapture('test.mp4')

# 状态变量
counter = 0
stage = None


## 使用 MediaPipe Pose 进行离线/无 GUI 的处理循环
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 如果帧读取失败则跳到下一个循环（或者可以选择重置帧指针）
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 将 BGR 转为 RGB 给 MediaPipe 处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 运行姿态检测
        results = pose.process(image)

        # 转回 BGR 以便用 OpenCV 显示/绘制
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # results.pose_landmarks 可能为 None，需要保护性处理
            landmarks = results.pose_landmarks.landmark

            # 提取感兴趣的关键点（归一化坐标）
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

            # 计算角度
            angle1 = calculate_angle(left_eye, left_hip, left_heel)
            angle2 = calculate_angle(right_eye, right_hip, right_heel)

            # 可视化角度（把归一化坐标乘以近似帧尺寸）
            cv2.putText(image, str(angle1), tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(angle2), tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # 打印关键点以便调试
            print(f"left_eye={left_eye}")
            print(f"right_eye={right_eye}")
            print(f"left_hip={left_hip}")
            print(f"right_hip={right_hip}")
            print(f"left_heel={left_heel}")
            print(f"right_heel={right_heel}")
            print(f"right index={right_index}")
            print(f"left index={left_index}")

            # 简单阈值判断（演示用途）
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
            # 捕获当 results.pose_landmarks 为 None 等异常
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

        # 按 q 键退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 
                       


