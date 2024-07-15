import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 指定输入和输出路径
source_video_path = '/home/liubw/test/virtual_human/DINet_optimized/asserts/test_data/origin_videos/test1.mp4'
csv_path = '/home/liubw/test/virtual_human/DINet_optimized/asserts/test_data/landmarks/test1.csv'
output_video_path = '/home/liubw/test/virtual_human/DINet_optimized/asserts/test_data/point_without_audio/test1.mp4'
def get_points(row):
    # print(row.index)
    x_points = [row[' x_{}'.format(i)] for i in range(68)]
    y_points = [row[' y_{}'.format(i)] for i in range(68)]
    return np.column_stack((x_points, y_points))

# 关键点之间的连接
connections = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15), (15,16), # jawline
               (17,18), (18,19), (19,20), (20,21), # right eyebrow
               (22,23), (23,24), (24,25), (25,26), # left eyebrow
               (27,28), (28,29), (29,30), # nose ridge
               (31,32), (32,33), (33,34), (34,35), # nose base
               (36,37), (37,38), (38,39), (39,40), (40,41), (41,36), # right eye
               (42,43), (43,44), (44,45), (45,46), (46,47), (47,42), # left eye
               (48,49), (49,50), (50,51), (51,52), (52,53), (53,54), (54,55), (55,56), (56,57), (57,58), (58,59), (59,48), # outer lip
               (60,61), (61,62), (62,63), (63,64), (64,65), (65,66), (66,67), (67,60)] # inner lip


# 获取源视频的尺寸
cap = cv2.VideoCapture(source_video_path)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 160
height = 160
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# 读取对应的 CSV 文件
df = pd.read_csv(csv_path)

# 创建一个 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 遍历每一帧
for _, row in df.iterrows():
    # 创建一个空白的背景
    plt.figure(figsize=(width/80, height/80), dpi=80)

    # 消除空白区域
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.imshow(np.zeros((height, width, 3)))
    plt.axis('off')

    # 获取关键点
    points = get_points(row)

    # 连接关键点
    for i, j in connections:
        plt.plot(points[[i, j], 0], points[[i, j], 1], 'r', marker='o', markersize=0)  # 设置非常小的标记

    # 将 matplotlib 图像转换为 OpenCV 图像
    plt.draw()
    img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8').reshape(160, 160, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 将图像写入视频
    video.write(img)

    # 关闭图像，释放资源
    plt.close(plt.gcf())

# 释放 VideoWriter
video.release()
