"""
train_frame_classifier.py

用途：对从 MediaPipe 导出的关键点 CSV 做帧级训练，示例使用 RandomForest。

输入：
  --landmarks PATH   : landmarks CSV（默认 ../landmarks.csv）
  --labels PATH      : labels CSV（两列：frame,label），必须提供，或使用 --auto-label 自动生成示例标签
  --out MODEL_PATH   : 输出模型路径（默认 ../model_frame.joblib）

示例（PowerShell）:
  python .\tools\train_frame_classifier.py --landmarks .\landmarks.csv --labels .\labels.csv --out .\model_frame.joblib

备注：如果没有手工标注，可以使用 --auto-label 用简单阈值生成演示标签（仅用于 demo，不用于真实训练）。
"""

import argparse
import os
import math
import json

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
except Exception as e:
    raise ImportError(f"缺少 sklearn/joblib，请先安装 scikit-learn 和 joblib。错误：{e}")


def parse_args():
    p = argparse.ArgumentParser(description='Train a frame-level classifier on MediaPipe landmarks')
    p.add_argument('--landmarks', type=str, default=os.path.join('..', 'landmarks.csv'), help='Landmarks CSV 路径')
    p.add_argument('--labels', type=str, default=None, help='Labels CSV (frame,label) 路径')
    p.add_argument('--out', type=str, default=os.path.join('..', 'model_frame.joblib'), help='输出模型路径')
    p.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    p.add_argument('--random-state', type=int, default=42, help='随机种子')
    p.add_argument('--n-est', type=int, default=100, help='RandomForest 森林中树的数量')
    p.add_argument('--auto-label', action='store_true', help='使用简单阈值自动生成标签（演示用途）')
    return p.parse_args()


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def load_landmarks(path):
    df = pd.read_csv(path)
    return df


def auto_label_from_angles(df):
    # 基于左右眼-臀-跟角度阈值粗略生成标签：如果左右角度都大于 170 -> label 0 (standing/safe)，否则 1 (possible fall)
    labels = []
    for _, row in df.iterrows():
        # 取出需要的 landmark 列
        try:
            le = (row['lm0_x'], row['lm0_y'])
            lh = (row['lm23_x'], row['lm23_y']) if 'lm23_x' in row else (math.nan, math.nan)
            lh_heel = (row['lm29_x'], row['lm29_y']) if 'lm29_x' in row else (math.nan, math.nan)
            re = (row['lm1_x'], row['lm1_y'])
            rh = (row['lm24_x'], row['lm24_y']) if 'lm24_x' in row else (math.nan, math.nan)
            rh_heel = (row['lm30_x'], row['lm30_y']) if 'lm30_x' in row else (math.nan, math.nan)

            angle_l = calculate_angle(le, lh, lh_heel)
            angle_r = calculate_angle(re, rh, rh_heel)
            if np.isnan(angle_l) or np.isnan(angle_r):
                labels.append(0)
            elif angle_l > 170 and angle_r > 170:
                labels.append(0)
            else:
                labels.append(1)
        except Exception:
            labels.append(0)
    return pd.DataFrame({'frame': df['frame'].astype(int), 'label': labels})


def build_features(df):
    """从 landmarks DataFrame 生成特征向量。
    选取的特征：
      - 左右侧 eye/hip/heel 的角度
      - 左右 hip 的 (x,y)
      - 左右 eye 的 (x,y)
      - hip x,y 的帧差（velocity）作为特征（需要前一帧）
    返回 X (numpy array), feature_names
    """
    feats = []
    feature_names = []

    # 为速度计算准备：将 hip 坐标列提取
    hip_l_x = df.get('lm23_x')
    hip_l_y = df.get('lm23_y')
    hip_r_x = df.get('lm24_x')
    hip_r_y = df.get('lm24_y')

    for idx, row in df.iterrows():
        try:
            le = (row['lm0_x'], row['lm0_y'])
            lh = (row['lm23_x'], row['lm23_y'])
            lh_heel = (row['lm29_x'], row['lm29_y'])
            re = (row['lm1_x'], row['lm1_y'])
            rh = (row['lm24_x'], row['lm24_y'])
            rh_heel = (row['lm30_x'], row['lm30_y'])

            angle_l = calculate_angle(le, lh, lh_heel)
            angle_r = calculate_angle(re, rh, rh_heel)

            # 基本坐标特征
            vals = [angle_l, angle_r,
                    lh[0], lh[1], rh[0], rh[1],
                    le[0], le[1], re[0], re[1]]

            # 速度特征（当前 hip - previous hip）
            if idx > 0:
                prev_lh = (hip_l_x.iloc[idx - 1], hip_l_y.iloc[idx - 1])
                prev_rh = (hip_r_x.iloc[idx - 1], hip_r_y.iloc[idx - 1])
                vel_l = (lh[0] - prev_lh[0], lh[1] - prev_lh[1])
                vel_r = (rh[0] - prev_rh[0], rh[1] - prev_rh[1])
            else:
                vel_l = (0.0, 0.0)
                vel_r = (0.0, 0.0)

            vals += [vel_l[0], vel_l[1], vel_r[0], vel_r[1]]

            feats.append(vals)
        except Exception:
            # 若缺失数据，填充 NaN
            feats.append([math.nan] * 14)

    feature_names = ['angle_l', 'angle_r', 'lh_x', 'lh_y', 'rh_x', 'rh_y', 'le_x', 'le_y', 're_x', 're_y',
                     'vel_l_x', 'vel_l_y', 'vel_r_x', 'vel_r_y']

    X = np.array(feats, dtype=float)
    # 用列均值填充 NaN
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X, feature_names


def main():
    args = parse_args()
    lm_path = args.landmarks
    if lm_path == os.path.join('..', 'landmarks.csv'):
        lm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'landmarks.csv'))
    print(f"加载 landmarks: {lm_path}")
    df_lm = load_landmarks(lm_path)

    if args.labels is None and not args.auto_label:
        raise ValueError('必须提供 --labels 或者使用 --auto-label 进行演示标签生成')

    if args.labels:
        labels = pd.read_csv(args.labels)
        # 期望 labels CSV 有 frame,label 两列
        if 'frame' not in labels.columns or 'label' not in labels.columns:
            raise ValueError('labels CSV 需要包含 frame,label 两列')
        labels['frame'] = labels['frame'].astype(int)
    else:
        print('使用 auto-label 生成标签（演示用途）')
        labels = auto_label_from_angles(df_lm)

    # 合并数据（按 frame）
    df = df_lm.copy()
    df['frame'] = df['frame'].astype(int)
    merged = pd.merge(df, labels[['frame', 'label']], on='frame', how='left')
    # 丢弃没有标签的帧
    merged = merged.dropna(subset=['label'])
    merged['label'] = merged['label'].astype(int)

    X, feature_names = build_features(merged)
    y = merged['label'].values

    print(f"样本数: {X.shape[0]}  特征数: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y))>1 else None)

    clf = RandomForestClassifier(n_estimators=args.n_est, random_state=args.random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('\n训练完成，测试集评估:')
    print(classification_report(y_test, y_pred))
    print('混淆矩阵:')
    print(confusion_matrix(y_test, y_pred))

    out_model = args.out
    if out_model == os.path.join('..', 'model_frame.joblib'):
        out_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_frame.joblib'))

    # 保存模型及特征名
    joblib.dump({'model': clf, 'feature_names': feature_names}, out_model)
    print(f"模型已保存到: {out_model}")


if __name__ == '__main__':
    main()
