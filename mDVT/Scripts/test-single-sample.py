import os
import pandas as pd
import cv2

from basics import (read_yolo_track, remove_outlier_boxes, object_number_check, repair_end_frames, remove_short_trajectory, rename_id,
                    remove_redundant_frames, join_trajectory, update_Trajectory_IDs, simple_fill_all_gaps, format_trajectory_to_DVT,
                    cal_velocity_from_position)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option('display.expand_frame_repr', False)

# 文件路径
# ## 包括yolo的输出结果和metadata.csv文件
#Yolo_outputs_dir = "D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\input-yolo-track-results\\sh"
Yolo_outputs_dir = "D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\input-yolo-track-results\\W1118-Rep1"
# ## 用于track的视频，一般是30min
#clean_video_dir = "D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\input-clean-video\\sh"
clean_video_dir = "D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\input-clean-video\\W1118-Rep1"
# ## 输出结果的存储目录，包括position文件、修正后metadata文件以及arena edge的图示文件、数据质量logs
#output_dir = "D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\outputs\\sh"
output_dir = "D:\\GitHub_Res\\BehavioralVideoAnalysis\\MyScripts\\YoloTrack-to-DVT-scale1\\outputs\\W1118-Rep1"
## 包括yolo的输出结果和metadata.csv文件

# 参数
## 果蝇数目
object_number = 6
## 并行线程数
cores = 4
## 视频长宽
video_width = 1024 #640
video_height = 768 #480

metadata_filepath = os.path.join(Yolo_outputs_dir, "metadata.csv")
meta_df = pd.read_csv(metadata_filepath)

yolo_track_files = [f for f in os.listdir(Yolo_outputs_dir) if (f.endswith('.csv') & f.startswith('20'))]
print("Results directory found!") if os.path.exists(output_dir) else os.mkdir(output_dir)

# 测试，不去除short trajectory, 并且全部smooth填充所有gap。
# 经过比较test1[去除short trajectory和不填充]和test2[不去除short，做填充]，发现test2整体上与DVT结果线性更好！
# 但是考虑到remove extrad id for frames工作量太大，还是决定去除short trajectory
log_filename = os.path.join(output_dir, 's1_logs.csv')

track_file = yolo_track_files[4] # for test

video_filename = track_file[:track_file.rfind(".")] + ".mp4"
cap = cv2.VideoCapture(os.path.join(clean_video_dir, video_filename))
# 经验总结：
# 1. too many frames with more IDs： 可能是果蝇body识别有问题，有很多假阳性！ plot trajectory on video 看一下。
# 2. Warning, Object number check failed, possible object number is 7： 也同上，可能是由于假阳性。

# track_file = yolo_track_files[1]
print(">>> Processing sample: " + track_file + "...")
track_res_path = os.path.join(Yolo_outputs_dir, track_file)
# 如果结果已经存在，则跳过
output_filename = track_file[:track_file.rfind(".")] + "_mp4_cleaned-position.csv"

# 读入track结果
## 基于每个frame检测到的box数目，然后找出数目值出现次数最多的，如果和定义的对象数目一致，则认为检查通过
track_res = read_yolo_track(path=track_res_path, object_number=object_number, mode='.txt')
boxes_initial = track_res.shape[0]

# s1: 去除非目标区的 boxes
## 读入目标区域信息
output_filename_revised = track_file[:track_file.rfind(".csv")] + "_mp4_cleaned-position.csv"
center_x = meta_df.loc[meta_df.csv == output_filename_revised, 'x'].tolist()[0]  # 1024
center_y = meta_df.loc[meta_df.csv == output_filename_revised, 'y'].tolist()[0]  # 768
circle_r = meta_df.loc[meta_df.csv == output_filename_revised, 'r'].tolist()[0]
## 执行去除
track_res = remove_outlier_boxes(trajectoryDF=track_res,
                                 video_width=video_width,
                                 video_height=video_height,
                                 center_x=center_x,
                                 center_y=center_y,
                                 circle_r=circle_r)
boxes_after_remove_outliers = track_res.shape[0]
print(str(boxes_initial - boxes_after_remove_outliers), " boxes removed!")

object_number_check(trajectoryDF=track_res, object_number=object_number, by='frame')

# 如果ID数目等于object number的frames数目不足3/5，则直跳过该样本
frame_counts = track_res['frame'].value_counts().value_counts().to_dict()
if frame_counts[object_number]/sum(frame_counts.values()) < 0.6:
    print("Analysis escaped for low percentage of qualified frames!")
    raise Exception("Analysis escaped for low percentage of qualified frames!")


# 修复两端
track_res = repair_end_frames(trajectoryDF=track_res, object_number=object_number)

# 注意remove_short_trajectory代码需要优化，能不能修改一次就能完成
# rename_id 和 remove_short_trajectory的顺序好像都可以？
track_res, trajectory_group, total_short_trajectory_removed, total_frames_removed = (
    remove_short_trajectory(trajectoryDF=track_res, object_number=object_number, trajectory_length_cut=10))
track_res, trajectory_group = rename_id(trajectoryDF=track_res, trajectoryGroup=trajectory_group, object_number=object_number)

# 两步法去除冗余frames
# 注意：由于用的测试数据中没有多出2个ID的frames，后续group redundant frames时，所有的group都可以找到对应的trajectory
# 先考虑ID数目为最多的，比如8的这些frames，先执行去除，然后再考虑数目为7的
# 有问题
track_res, trajectory_group = remove_redundant_frames(trajectoryDF=track_res, trajectoryGroup=trajectory_group, object_number=object_number, cores=4)

# 两步法 解决ID jump
trajectory_group_solved = join_trajectory(trajectoryDF=track_res, trajectoryGroup=trajectory_group, object_number=object_number)

# update IDs in trajectory
track_res = update_Trajectory_IDs(trajectoryDF=track_res, Solved_trajectoryGroup=trajectory_group_solved)

# 填补gaps
# 两种方式，如果gap长度大于gap_length_cutoff，并且能找到两端frame对应的最近ID是一致的，则使用混合填充，即smooth insert 和 NN fill，否者都使用smooth insert
track_res = simple_fill_all_gaps(trajectoryDF=track_res, gap_length_cutoff=100, cores=4)

# 导出
# step4: reformat
dvt_coordinates = format_trajectory_to_DVT(trajectoryDF=track_res,
                                           object_number=object_number,
                                           video_width=video_width,
                                           video_height=video_height)

# 检查是否有大的跳跃
velocity_df = cal_velocity_from_position(pos=dvt_coordinates, n_inds=object_number, fps=30, scaling_to_mm=1)
print(velocity_df.velocity.max()) # 不要超过500？ 没测试过阈值