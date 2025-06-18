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
# 包括yolo的输出结果和metadata.csv文件
Yolo_outputs_dir = 'D:\\GitHub_Res\\FlyBehavioralAnalysis\\mDVT\\Data\\YoloBytetrackOutputs\\W1118-Rep2'
# ## 用于track的视频，一般是30min
clean_video_dir = 'D:\\GitHub_Res\\FlyBehavioralAnalysis\\mDVT\\Data\\ClippedVideo\\W1118-Rep2'
# 输出结果的存储目录，包括position文件、修正后metadata文件以及arena edge的图示文件、数据质量logs
output_dir = 'D:\\GitHub_Res\\FlyBehavioralAnalysis\\mDVT\\Data\\RevisedDataForDVT\\w1118-Rep2'

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

for track_file in yolo_track_files:
    # track_file = yolo_track_files[1] # for test
    video_filename =  track_file[:track_file.rfind(".")] + ".mp4"
    cap = cv2.VideoCapture(os.path.join(clean_video_dir, video_filename))
    # 经验总结：
    # 1. too many frames with more IDs： 可能是果蝇body识别有问题，有很多假阳性！ plot trajectory on video 看一下。
    # 2. Warning, Object number check failed, possible object number is 7： 也同上，可能是由于假阳性。

    # track_file = yolo_track_files[1]
    print(">>> Processing sample: " + track_file + "...")
    track_res_path = os.path.join(Yolo_outputs_dir, track_file)
    # 如果结果已经存在，则跳过
    output_filename = track_file[:track_file.rfind(".")] + "_mp4_cleaned-position.csv"
    if os.path.exists(os.path.join(output_dir, output_filename)):
        print("Escaped...")
        continue

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
    if frame_counts[object_number] / sum(frame_counts.values()) < 0.6:
        print("Analysis escaped for low percentage of qualified frames, this is probably caused by a poor Yolo detection!")
        continue

    # 修复两端
    # 前端和末端要均有object number个detection
    # 如果不符合，则进行补齐或去除，优先去除短轨迹
    track_res = repair_end_frames(trajectoryDF=track_res, object_number=object_number)

    # rename_id 和 remove_short_trajectory的顺序好像都可以？
    # 去除短轨迹
    track_res, trajectory_group, total_short_trajectory_removed, total_frames_removed = (
        remove_short_trajectory(trajectoryDF=track_res, object_number=object_number, trajectory_length_cut=10))

    # 重命名轨迹
    track_res, trajectory_group = rename_id(trajectoryDF=track_res, trajectoryGroup=trajectory_group, object_number=object_number)

    # 两步法 去除冗余 frames
    # s1, 去除整条冗余轨迹
    # s2, 仅仅在轨迹两端去除冗余frames，优先从大ID的前端去除
    track_res, trajectory_group = remove_redundant_frames(trajectoryDF=track_res, trajectoryGroup=trajectory_group, object_number=object_number, cores=4)

    # 两步法 解决ID jump
    # s1, 基于间隔的frame数目及ORA取值，优先连接置信度高的轨迹
    # s2，基于最短间隔的frames，优先连接长度长的轨迹
    trajectory_group_solved = join_trajectory(trajectoryDF=track_res, trajectoryGroup=trajectory_group, object_number=object_number)

    # update IDs in trajectory
    # 基于trajectory_group_solved中的轨迹连接记录，修正轨迹文件中的轨迹ID
    track_res = update_Trajectory_IDs(trajectoryDF=track_res, Solved_trajectoryGroup=trajectory_group_solved)

    # fill all gaps
    # 两种方式，
    # case1: 如果gap长度大于gap_length_cutoff，并且能找到两端frame对应的最近ID是一致的，则使用混合填充(fill_by_neighbour),
    # 如果最近邻没有gaps，则使用最近邻直接填充. 如果有gaps，则先使用平滑填充方式补齐最近邻的，
    # case2: 对于gap长度大于300的且末端有一个最近邻的考虑使用最近邻的位置进行填充，
    track_res = simple_fill_all_gaps(trajectoryDF=track_res, gap_length_cutoff=100, cores=4)

    # 导出
    # 格式化文件作为DVT input
    dvt_coordinates = format_trajectory_to_DVT(trajectoryDF=track_res,
                                      object_number=object_number,
                                      video_width=video_width,
                                      video_height=video_height)
    # save
    dvt_coordinates.to_csv(os.path.join(output_dir, output_filename), index=True, index_label='position')

    # QC
    # 检查是否有大的跳跃
    velocity_df = cal_velocity_from_position(pos=dvt_coordinates, n_inds=object_number, fps=30, scaling_to_mm=1)
    print("The largest speed is " + str(int(velocity_df.velocity.max())))  # 不要超过500？ 没测试过阈值



    # statistic_summary = pd.DataFrame(dict(video_name=video_filename,
    #                       boxes_initial=boxes_initial,
    #                       boxes_after_remove_outliers=boxes_after_remove_outliers,
    #                       boxes_removed=boxes_initial-boxes_after_remove_outliers,
    #                       frames_has_moreIDs=frames_has_moreIDs,
    #                       frames_has_2_more_IDs_than_previous=frames_has_2_more_IDs_than_previous_frame,
    #                       newIDs_after_trajectory_split=newIDs_solved_after_trajectory_split,
    #                       newIDs_has_more_candidates=newIDs_has_more_candidates,
    #                       gaps_filled=gaps_filled), index=[0])
    # if os.path.exists(log_filename):
    #     statistic_summary.to_csv(log_filename, mode='a', header=False, index=False)
    # else:
    #     statistic_summary.to_csv(log_filename, header=True, index=False)

