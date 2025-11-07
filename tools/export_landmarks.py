"""
export_landmarks.py

用途：使用 MediaPipe Pose 从视频中逐帧提取关键点并保存为 CSV，默认读取项目根目录下的 test.mp4。

输出：CSV 文件，列格式如下：
  frame,time_ms,width,height,
  lm0_x,lm0_y,lm0_z,lm0_vis, lm1_x,lm1_y,lm1_z,lm1_vis, ..., lm32_x,lm32_y,lm32_z,lm32_vis

用法示例（PowerShell）：
  # 建议先在虚拟环境中安装依赖（见 requirements.txt）
  python .\tools\export_landmarks.py --video .\test.mp4 --out .\landmarks_test.csv

参数：
  --video PATH    : 视频路径（默认 ./test.mp4）
  --out PATH      : 输出 CSV 路径（默认 ./landmarks.csv）
  --max-frames N  : 最多处理帧数（可用于测试）
  --skip N        : 每 N 帧抽取一帧（默认 1，即每帧）

注意：需要安装 mediapipe, opencv-python, numpy

"""

import csv
import argparse
import os
import math

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except Exception as e:
    raise ImportError(f"缺少依赖，请先安装 opencv-python, mediapipe, numpy。错误：{e}")


def init_parser():
    p = argparse.ArgumentParser(description='Export MediaPipe Pose landmarks to CSV')
    p.add_argument('--video', type=str, default=os.path.join('..', 'test.mp4'), help='视频路径，默认 ../test.mp4 (相对于 tools 目录)')
    p.add_argument('--out', type=str, default=os.path.join('..', 'landmarks.csv'), help='输出 CSV 路径，默认 ../landmarks.csv')
    p.add_argument('--max-frames', type=int, default=0, help='最多处理多少帧，0 表示全部')
    p.add_argument('--skip', type=int, default=1, help='每隔多少帧抽取 1 帧（默认 1）')
    return p


def make_header(num_landmarks=33):
    header = ['frame', 'time_ms', 'width', 'height']
    for i in range(num_landmarks):
        header += [f'lm{i}_x', f'lm{i}_y', f'lm{i}_z', f'lm{i}_vis']
    return header


def write_row(csv_writer, frame_idx, time_ms, width, height, landmarks):
    # landmarks: list of pose_landmark objects or None
    row = [frame_idx, int(time_ms), width, height]
    if landmarks is None:
        # 填充 NaN
        for _ in range(33):
            row += [math.nan, math.nan, math.nan, math.nan]
    else:
        for lm in landmarks:
            row += [lm.x, lm.y, lm.z, getattr(lm, 'visibility', math.nan)]
    csv_writer.writerow(row)


def export_landmarks(video_path, out_csv, max_frames=0, skip=1):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {video_path}  分辨率: {width}x{height}  FPS: {fps}  总帧数(估计): {total_frames}")

    header = make_header(num_landmarks=33)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with mp_pose.Pose(static_image_mode=False,
                           model_complexity=1,
                           enable_segmentation=False,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as pose:

            frame_idx = 0
            written = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % skip != 0:
                    continue

                time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Convert color
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                else:
                    landmarks = None

                write_row(writer, frame_idx, time_ms, width, height, landmarks)

                written += 1
                if written % 100 == 0:
                    print(f"已写入 {written} 帧 (当前帧 {frame_idx})")

                if max_frames and written >= max_frames:
                    break

    cap.release()
    print(f"导出完成，已写入 {written} 行至 {out_csv}")


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # 默认 video 路径修正：如果传入默认且脚本位于 tools/，则查找上一级的 test.mp4
    vid = args.video
    if vid == os.path.join('..', 'test.mp4'):
        # 转换为相对于项目根路径
        vid = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test.mp4'))

    out_csv = args.out
    if out_csv == os.path.join('..', 'landmarks.csv'):
        out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'landmarks.csv'))

    print(f"输入视频: {vid}\n输出 CSV: {out_csv}")
    export_landmarks(vid, out_csv, max_frames=args.max_frames, skip=args.skip)
