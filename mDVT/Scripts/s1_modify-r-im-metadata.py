# 使用DVT helper定义arena区域时，使用的是非最小圆，而轨迹未能覆盖该arena的边缘区，
# 使用轨迹至圆心的最大距离重新定义arena区域，另外设置一个经验值，为最小圆的可能半径，最终使用的r要大于或等于该半径。
# 补充，一开始使用非最小圆的好处是，如果轨迹落到最小圆外时，该轨迹不会被滤除掉。

import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

dvt_output_filepath = 'D:\\GitHub_Res\\FlyBehavioralAnalysis\\mDVT\\Data\\RevisedDataForDVT\\w1118-Rep2'
raw_meta_filepath = 'D:\\GitHub_Res\\FlyBehavioralAnalysis\\mDVT\\Data\\RevisedDataForDVT\\w1118-Rep2\\metadata.csv'

meta_df = pd.read_csv(raw_meta_filepath, index_col = 0)
fly_numbers = 6
big_circle_r = 20 #指的是外圆，即最大圆
small_circle_r = 37/2 # 37mm/2 # 内圆
r_scale = small_circle_r/big_circle_r
# scaling = 1.6  # 640*480
scaling = 1  # 1024*768

samples_to_be_removed = []

for index, A_entry in meta_df.iterrows():
    position_filepath = os.path.join(dvt_output_filepath, A_entry['csv'])
    if not os.path.exists(position_filepath):
        print('Sample Escaped: ' + A_entry['videoname'] + ', and will be removed in the new metadata file.')
        continue
    position_df = pd.read_csv(position_filepath)
    max_r = []
    for fly_id in range(fly_numbers):
        max_r.append(pow((position_df['x' + str(fly_id)] * scaling - A_entry['x']) ** 2 + (position_df['y' + str(fly_id)] * scaling - A_entry['y']) ** 2, 0.5).max())
    # 如果果蝇到圆心的最大半径大于metadata中定义的r
    if max(max_r) > A_entry['r']:
        raise Exception('The inferred r is more than the defined r by circle!')
    # 如果果蝇到圆心的最大半径小于 metadata中定义的r乘以r_scale，即内圆的r，那就使用内圆的圆心
    if max(max_r) < A_entry['r'] * r_scale:
        print('Use the scaled r instead of the inferred for it is too small!')
        meta_df.loc[meta_df['videoname'] == A_entry['videoname'], 'r'] = A_entry['r'] * r_scale
    # 否则使用果蝇到圆心的最大半
    else:
        meta_df.loc[meta_df['videoname'] == A_entry['videoname'], 'r'] = max(max_r)

if len(samples_to_be_removed) != 0:
    meta_df = meta_df.drop(samples_to_be_removed)

meta_new_filepath = os.path.join(dvt_output_filepath, 'metadata.csv')
meta_df.to_csv(meta_new_filepath)