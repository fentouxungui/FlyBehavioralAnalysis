# 在路径图中绘制分析区域-圆形,对照文章绘制出edge区域
# 长直线轨迹是怎么回事
import pandas as pd
import os
import cv2
# import numpy as np
from basics import generate_colors_dict

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

# dvt_output_filepath = 'D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\outputs\\sh'
dvt_output_filepath = 'D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\outputs\\W1118-Rep1'
meta_filename = 'metadata.csv'

meta_filepath = os.path.join(dvt_output_filepath, meta_filename)
meta_df = pd.read_csv(meta_filepath, index_col = 0)
fly_numbers = 6
color_list = generate_colors_dict(range(fly_numbers))
# scaling = 1.6  # 640*480
scaling = 1 # 1024*768
edge_define = 3 * 2
arena_r = 37
arena_scale = 1- edge_define/arena_r

for _, A_entry in meta_df.iterrows():
    position_filepath = os.path.join(dvt_output_filepath,  A_entry['csv'])
    if not os.path.exists(position_filepath):
        print('Sample Escaped: ', A_entry['videoname'])
        continue
    position_df = pd.read_csv(position_filepath)
    # blank_image = np.zeros((768, 1024, 3), np.uint8)
    # blank_image[:, :] = (255, 255, 255)
    video_filepath = os.path.join(os.path.dirname(os.path.dirname(dvt_output_filepath)), 'input-clean-video', os.path.basename(dvt_output_filepath),A_entry['videoname'])
    cap = cv2.VideoCapture(video_filepath)
    _, blank_image = cap.read()
    blank_image = cv2.circle(blank_image, [int(A_entry['x']), int(A_entry['y'])], int(A_entry['r']), color = (255,0,0), thickness=2)
    blank_image = cv2.circle(blank_image, [int(A_entry['x']), int(A_entry['y'])], int(A_entry['r']  * arena_scale), color=(0, 255, 0), thickness=2)
    cv2.imencode('.jpg', blank_image)[1].tofile(f"{dvt_output_filepath}/{A_entry['videoname']}_raw.png")
    for _, row in position_df.iterrows():
        for fly_id in range(fly_numbers):
            cv2.circle(blank_image, (int(row['x' + str(fly_id)] * scaling), int(row['y' + str(fly_id)] * scaling)), 1, color_list[fly_id], -1)
    # cv2.imshow('Image', blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imencode('.jpg', blank_image)[1].tofile(f"{dvt_output_filepath}/{A_entry['videoname']}_motion_trace_circle.png")
    cap.release()

