import cv2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
import seaborn as sns
import numpy as np
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from tqdm import tqdm
import distinctipy
import math
from ultralytics import YOLO


# matplotlib.use('Qt5Agg')
# matplotlib.use('TKAgg')

# read data
def object_number_check(trajectoryDF, object_number, by = 'frame'):
    # 基于每个frame检测到的box数目，然后找出数目值出现次数最多的，如果和定义的对象数目一致，则认为检查通过
    count_IDs_by_frame = trajectoryDF[by].value_counts().value_counts()
    print("frame counts by ID number: ", count_IDs_by_frame.to_dict())
    possible_object_numbers = count_IDs_by_frame.idxmax()
    if possible_object_numbers == object_number:
        print("Object number check passed...")
    else:
        print("Warning, Object number check failed, possible object number is " + str(possible_object_numbers) + "...")
    pass

def read_yolo_track(path, object_number=None, header=None, mode=None):
    """
    读取Yolo Track 结果，并加入Box Area 和 Inserted列，分别表示每个box的面积和每行是否为人为插入值
    :param path: path to track results file, a csv file
    :return: a pandas dataframe, with columns: 'frame', 'category', 'x', 'y', 'w', 'h', 'id', 'BoxArea', 'Inserted'
    """
    file_name = os.path.basename(path)
    if mode in ['.csv', '.txt']:
        if mode == '.csv':
            track_res = pd.read_csv(path, header=header)
        else:
            track_res = pd.read_csv(path, header=header, sep=" ")
    else:
        if os.path.splitext(file_name)[-1] == '.csv':
            track_res = pd.read_csv(path, header=header)
        elif os.path.splitext(file_name)[-1] == '.txt':
            track_res = pd.read_csv(path, header=header, sep=" ")
        else:
            raise Exception('only support csv or txt file.')
    if header is None:  # 如果数据包括列名
        track_res.columns = ['frame', 'category', 'x', 'y', 'w', 'h', 'id']
    else:
        print('Rename column names escaped! please check the column names manually.')
    box_area = np.multiply(track_res['w'], track_res['h']).to_numpy()
    box_area = np.round(box_area, 7)
    track_res = track_res.assign(BoxArea=box_area)
    track_res = track_res.assign(Inserted=False, Inserted_method='-')
    if object_number is None:
        print("Check object number skipped...")
    else:
        object_number_check(trajectoryDF = track_res, object_number = object_number, by='frame')
    return track_res

def repair_head_frames(trajectoryDF, object_number):
    # 视频第一帧的果蝇数目必须等于object number，# 并且ID编号为从1到object_number，这个没必要吧？
    # 如果小于object number，则用后面的来前向填充，注意后面的必须为6
    # 如果大于6，比如55577这种，则直接报错，
    # 第一个数字5表示第一帧检测到5个，第二个为5表示第二帧检测到5个，第四个为7，表示第四帧检测到7个
    first_frame_id_counts = trajectoryDF.loc[trajectoryDF['frame'] == 1,'frame'].shape[0]
    first_frame_ids = set(trajectoryDF.loc[trajectoryDF['frame'] == 1, 'id'].tolist())
    if first_frame_id_counts == object_number:
        print("First frame check: the first frame has equal detections as the object_number!")
        # if first_frame_ids == set(range(1, object_number + 1)):
        #     raise Exception("Error: IDs from the first frame is unusual!")
    else:
        print("First frame check: Warning, the first frame has unequal detections as the object_number!")
        print("First frame check: Trying to fill the starting gap!")
        current_frame = 2
        # 找到哪一帧有正确数目的果蝇数目，如果过程中发现有多余果蝇数目的，直接报错！
        while True:
            current_frame_id_counts = trajectoryDF.loc[trajectoryDF['frame'] == current_frame,'frame'].shape[0]
            if current_frame_id_counts < object_number:
                current_frame+=1
                continue
            elif current_frame_id_counts == object_number:
                break
            else:
                raise Exception("Error: can not found the corresponding ID in subsequent frames for the gap!")
        # 用后续ID的坐标填充前面的gap
        current_frame_ids = set(trajectoryDF.loc[trajectoryDF['frame'] == current_frame, 'id'].tolist())
        new_IDs = list(current_frame_ids - first_frame_ids)
        if len(new_IDs) == 1:
            print("First frame check: Found the corresponding ID in subsequent frames for the gap!")
            print("First frame check: filling the gap!")
            target_row = trajectoryDF.loc[(trajectoryDF['id'] == list(new_IDs)[0]) & (trajectoryDF['frame'] == current_frame)]
            target_row = target_row.reset_index(drop=True)
            target_row.loc[0,'Inserted'] = True
            target_row.loc[0, 'Inserted_method'] = 'repair_head_frames'
            target_row_replicated = pd.concat([target_row] * (current_frame - 1), ignore_index=True)
            target_row_replicated['frame'] = list(range(1,current_frame))
            trajectoryDF = pd.concat([trajectoryDF, target_row_replicated], ignore_index=True)
            trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
            trajectoryDF = trajectoryDF.reset_index(drop=True)
        else:
            raise Exception("Error: found more than 1 new ID in subsequent frames for the gap!")
    return trajectoryDF

def repair_tail_frames(trajectoryDF, object_number):
    # 视频最后一帧的果蝇数目必须等于object number，
    # 主要原因是后续做轨迹连接时，join_trajectory_by_distance 评估轨迹是否为完整时的依据是末端frame的位置，如果尾端不整齐，会影响这一步
    # 如果小于object number，则用最后出现的轨迹进行延伸
    # 如果大于6，比如457755这种，则直接报错，
    # 最后一个数字5表示第一帧检测到5个，第二个为5表示倒数第二帧检测到5个，倒数第3个为7，这种情况下，多出来的2个轨迹均会被延伸至末尾
    # 方法：从末端开始向前判定box数目
    last_frame_id_counts = trajectoryDF.loc[trajectoryDF['frame'] == trajectoryDF['frame'].max(), 'frame'].shape[0]
    last_frame_ids = set(trajectoryDF.loc[trajectoryDF['frame'] == trajectoryDF['frame'].max(), 'id'].tolist())
    if last_frame_id_counts == object_number:
        print("Last frame check: the last frame has equal detections as the object_number!")
    elif last_frame_id_counts > object_number:
        print("Last frame check: the last frame has more detections than the object_number!")
        print("Last frame check: the shortest trajectory with length less than 100 frames will be removed!")
        # do something to remove the shortest trajectory
        length_of_all_tail_ids = trajectoryDF[trajectoryDF['id'].isin(last_frame_ids)]['id'].value_counts().to_dict()
        ids_to_be_kept = dict(sorted(length_of_all_tail_ids.items(), key=lambda item: item[1], reverse=True)[:object_number])
        ids_to_be_removed = dict(sorted(length_of_all_tail_ids.items(), key=lambda item: item[1], reverse=True)[object_number:])
        # check if all ids to be removed has frames length less than 100, and remove those ids
        for key, value in ids_to_be_removed.items():
            if value > 100:
                print("Warning: trajectory - " + str(key) + " to be removed has " + str(value) + "frames which is is more than 100, please pay a attention!")
            if key in last_frame_ids:
                trajectoryDF = trajectoryDF[trajectoryDF['id'] != key]
    else:
        print("Last frame check: Warning, the last frame has less detections as the object_number!")
        current_frame = (trajectoryDF['frame'].max() - 1)
        # 迭代定位到某一帧，该帧的果蝇ID在加上last_frame_ids刚好有等于或大于果蝇数目的ID，只考虑新多出来的（与最末端ID相比），如果发现有多余果蝇数目的，则均会被延申
        while True:
            current_frame_ids = set(trajectoryDF.loc[trajectoryDF['frame'] == current_frame, 'id'].tolist())
            extra_ids = list(current_frame_ids - last_frame_ids)
            if (len(extra_ids) + last_frame_id_counts) < object_number:
                current_frame = current_frame - 1
                continue
            else:
                break
        if (len(extra_ids) + last_frame_id_counts) > object_number:
            # 对于末端所有ID，统计长度，对于长度最短的轨迹进行去除，其它轨迹进行补齐
            print("Last frame check: Warning: found more trajectory with lost at end!")
            print("Last frame check: the shortest trajectory with frames less than 100 frames will be removed!")
            # 延申extra_ids至末尾
            all_tail_ids = extra_ids + list(last_frame_ids)
            length_of_all_tail_ids = trajectoryDF[trajectoryDF['id'].isin(all_tail_ids)]['id'].value_counts().to_dict()
            ids_to_be_kept = dict(sorted(length_of_all_tail_ids.items(), key=lambda item: item[1], reverse=True)[:object_number])
            ids_to_be_removed = dict(sorted(length_of_all_tail_ids.items(), key=lambda item: item[1], reverse=True)[object_number:])
            # check if all ids to be removed has frames length less than 100, and remove those ids
            for key, value in ids_to_be_removed.items():
                if value > 100:
                    print("Warning: trajectory - " + str(key) + " to be removed has " + str(value) +  "frames which is is more than 100, please pay a attention!")
                if key in last_frame_ids:
                    trajectoryDF = trajectoryDF[trajectoryDF['id'] != key]
            extra_ids_kept = list(set(extra_ids).intersection(set(ids_to_be_kept.keys())))
        for newID in extra_ids_kept:
            target_row = trajectoryDF.loc[(trajectoryDF['id'] == newID) & (trajectoryDF['frame'] == current_frame)]
            target_row = target_row.reset_index(drop=True)
            target_row.loc[0, 'Inserted'] = True
            target_row.loc[0, 'Inserted_method'] = 'repair_tail_frames'
            target_row_replicated = pd.concat([target_row] * (trajectoryDF['frame'].max() - current_frame), ignore_index=True)
            target_row_replicated['frame'] = list(range(current_frame + 1, trajectoryDF['frame'].max() + 1))
            trajectoryDF = pd.concat([trajectoryDF, target_row_replicated], ignore_index=True)
        trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
        trajectoryDF = trajectoryDF.reset_index(drop=True)
    return trajectoryDF

def repair_end_frames(trajectoryDF, object_number):
    trajectoryDF = repair_head_frames(trajectoryDF=trajectoryDF, object_number=object_number)
    trajectoryDF = repair_tail_frames(trajectoryDF=trajectoryDF, object_number=object_number)
    return trajectoryDF

# step1: remove short trajectory

def group_numbers(frames, cut=100):
    """
    对排序好的数字进行分组，如果相邻两个数字的差值小于cut值，则会分配到同一组
    :param frames: a number list
    :param cut: 用于分组时的最小距离，小于或等于这个距离，则被认为同上一个数字是同一组
    :return: a pandas dataframe, with columns: group, min, max, range, middle((min+max)/2)
    """
    res = pd.DataFrame({'raw': frames})
    res['group'] = ((res['raw'].diff() > cut).cumsum())
    res = res.groupby('group').agg(min=('raw', 'min'), max=('raw', 'max')).reset_index()
    res['min'] = res['min'].astype(int)
    res['max'] = res['max'].astype(int)
    res['range'] = (res['max'] - res['min'] + 1).astype(int)
    res['middle'] = np.round((res['min'] + res['max']) / 2).astype(int)
    return res


def parallel_group_trajectory(trajectoryDF, group_by, cut=1, cores=4):
    """
    多线程方式，对 dataframe依据id进行分组，然后得到每个ID所有trajectory【连续frame】的range信息，包括起始位置，终止位置，长度
    return: a pandas dataframe
    """

    def data_process(name, group):
        res = group_numbers(group['frame'], cut=cut)
        res = res.assign(id=name)
        return res

    def applyParallel(dfGrouped, func):
        # 多线程方式，执行某个function，并合并结果
        res = Parallel(n_jobs=cores)(delayed(func)(name, group) for name, group in dfGrouped)
        return pd.concat(res)

    df = trajectoryDF.groupby(group_by)
    result = applyParallel(df, data_process)
    return result

def check_first_frame(trajectoryDF, object_number):
    # 检查一下frame为1时的box数量都否等于object number
    first_frame_id_counts = trajectoryDF.loc[trajectoryDF['frame'] == 1, 'frame'].shape[0]
    if first_frame_id_counts != object_number:
        raise Exception('Error: the first frame has incorrect detection counts!')
    else:
        pass

def remove_short_trajectory(trajectoryDF, object_number, trajectory_length_cut=10, cores=4):
    """
    去除 short trajectory，不去除从frame1起始的，以及以end frame结尾的轨迹！
    :param trajectoryDF: 轨迹文件， a pandas dataframe
    :param object_number: object数目，起始ID的trajectory不会被去除
    :param trajectory_length_cut: 连续 frames 少于 trajectory_cut 的轨迹会被去除
    :param cores: cores used by parallel_group_trajectory
    :return: an updated trajectory, an updated trajectory groups, trajectory removed, frames removed
    """
     # 检查一下frame为1时的box数量都否等于object number
    check_first_frame(trajectoryDF=trajectoryDF, object_number=object_number)
    trajectoryGroup = parallel_group_trajectory(trajectoryDF=trajectoryDF, group_by='id', cores=cores, cut=1)
    trajectoryGroup = trajectoryGroup.reset_index(drop=True)
    # 对于剩余其它grouped trajectory,去除长度短于trajectory_length_cut的
    trajectoryGroup_Short = trajectoryGroup.loc[trajectoryGroup['min'] != 1] # 去除前端起始的轨迹
    trajectoryGroup_Short = trajectoryGroup_Short.loc[trajectoryGroup_Short['max'] != trajectoryDF['frame'].max()]  #去除终止于末端的轨迹
    trajectoryGroup_Short = trajectoryGroup_Short[trajectoryGroup_Short.range <= trajectory_length_cut]
    trajectoryGroup_left = trajectoryGroup[~trajectoryGroup.index.isin(trajectoryGroup_Short.index)]
    total_short_trajectory_removed = trajectoryGroup_Short.shape[0]
    total_frames_removed = trajectoryGroup_Short.range.sum()
    print('Total found ' + str(total_short_trajectory_removed) + ' short trajectory with total ' + str(total_frames_removed) + ' frames, removing...')
    for i in tqdm(range(total_short_trajectory_removed)):
        for j in range(trajectoryGroup_Short.iloc[i]['min'], trajectoryGroup_Short.iloc[i]['max'] + 1):
            trajectoryDF = trajectoryDF[~((trajectoryDF['frame'] == j) & (trajectoryDF['id'] == trajectoryGroup_Short.iloc[i]['id']))]
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    trajectoryGroup_left = trajectoryGroup_left.reset_index(drop=True)
    return trajectoryDF, trajectoryGroup_left, total_short_trajectory_removed, total_frames_removed


def rename_id(trajectoryDF,trajectoryGroup, object_number, cores=4):
    # 检查一下frame为1时的box数量都否等于object number
    check_first_frame(trajectoryDF=trajectoryDF, object_number=object_number)
    # 依据first frame的顺序依次重命名
    trajectoryGroup['parent_id'] = trajectoryGroup['id']
    trajectoryGroup = trajectoryGroup.sort_values(by=['min', 'id']).reset_index(drop=True)
    trajectoryGroup['id'] = list(range(1, trajectoryGroup.shape[0] + 1))
    # rename ids in trajectoryDF
    trajectoryDF['parent_id'] = trajectoryDF['id']
    trajectoryDF['id'] = 0
    for _, row in tqdm(trajectoryGroup.iterrows(), total=trajectoryGroup.shape[0]):
        trajectoryDF.loc[(trajectoryDF['parent_id'] == row['parent_id']) & (trajectoryDF['frame'] >= row['min']) &
        (trajectoryDF['frame'] <= row['max']), 'id'] = row['id']
    if (trajectoryDF['id'] == 0).any():
        raise Exception('Error, the trajectoryDF and trajectoryGroup file are not fully matched!')
    return trajectoryDF, trajectoryGroup

def cal_trajectory_duplication_pct(trajectoryGroup, frameDF):
    # 问题是运行慢，默认用所有trajectory去跑
    print("Calculating duplication percentage...")
    dup_frames = frameDF.loc[frameDF['type'] > 0,'frame'].to_list()
    pct = []
    for _, row in tqdm(trajectoryGroup.iterrows(), total=trajectoryGroup.shape[0]):
        trajectory_frames = list(range(int(row['min']), int(row['max']+ 1)))
        pct.append(sum(x in trajectory_frames for x in dup_frames)/len(trajectory_frames))
    trajectoryGroup['duplication_pct'] = pct
    return trajectoryGroup

def check_trajectory_complexity(trajectoryGroup, frameDF):
    # 判定trajectory的区域是否为简单情况：trajectory从起始到终止区域，没有任何其它新产生的ID或丢失的ID，即该区域的ID是稳定的
    complexity = []
    IDs_included = []
    for _, row in tqdm(trajectoryGroup.iterrows(), total=trajectoryGroup.shape[0]):
        IDs_in_range = frameDF.loc[(frameDF['frame'] >= row['min']) & (frameDF['frame'] <= row['max']), 'IDs']
        IDs_included.append(IDs_in_range.explode().unique().tolist()) #存储所有该frame区间相关的IDs
        # 如果ID组合只有一种
        if IDs_in_range.value_counts().shape[0] == 1:
            complexity.append('simple')
        else:
            complexity.append('complex')
    trajectoryGroup['ids_included'] = IDs_included
    trajectoryGroup['complexity'] = complexity
    trajectoryGroup['id_counts_included'] = trajectoryGroup['ids_included'].apply(len)
    return trajectoryGroup

def collapse_df(df, group_column, value_column):
    df_collapsed = df.groupby(group_column)[value_column].apply(list).rename('IDs').reset_index()
    df_collapsed['Counts'] = df_collapsed['IDs'].apply(len)
    return df_collapsed

def frame_statics(trajectoryDF, object_number):
    df = collapse_df(df=trajectoryDF, group_column='frame', value_column='id')
    df['type'] = 0 #表示ID数目等于object number
    df.loc[df['Counts'] < object_number, 'type'] = -1
    df.loc[df['Counts'] > object_number, 'type'] = 1
    return df

def filter_dict(Adict, cutoff):
    filtered_dict = {key: value for key, value in Adict.items() if value > cutoff}
    return filtered_dict

def find_neighbours_for_trajectory(trajectoryDF, trajectoryGroup, frameDF, object_number,length_max=500, scale = 1.5, overlapped_area_cutoff_for_NN=0.2):
    # 猜测冗余轨迹的最长长度
    frameDF_more = frameDF[frameDF['type'] == 1]
    # 如果没有冗余frames,length_cutoff设为0，只有短于该长度的轨迹才被用于分析是否为冗余轨迹
    if frameDF_more.empty:
        trajectoryGroup['length_cutoff']=0
    else:
        redundancy_trajectory = group_numbers(frames=frameDF_more['frame'].to_list(), cut=1)
        # 设置的这个length cutoff有可能也不合适，默认最大值为500，如果太大有可能会引入太长的gap，这样后续可能无法拼接轨迹！
        length_cutoff = min(length_max, int(redundancy_trajectory['range'].max() * scale)) # scale default 1.5, 去除冗余轨迹时，只考虑长度短于这个的轨迹
        trajectoryGroup['length_cutoff'] = length_cutoff
        print("Redundancy trajectory length cutoff is set to " + str(length_cutoff))
        trajectoryGroup['forward_overlapping_area'] = trajectoryGroup.apply(lambda _: {}, axis=1)
        trajectoryGroup['backward_overlapping_area'] = trajectoryGroup.apply(lambda _: {}, axis=1)
        for index, row in tqdm(trajectoryGroup.iterrows(), total=trajectoryGroup.shape[0]):
            if row['range'] < length_cutoff and row['id'] > object_number:
                # 注意calculate_ORA是计算所有id对之间的比例，后续考虑使用仅仅与目标ID的计算
                trajectoryGroup.at[index, 'forward_overlapping_area'] = (
                    calculate_ORA_single(frame_trajectoryDF=trajectoryDF[trajectoryDF['frame'] == row['min']],
                target_id = row['id']))
                trajectoryGroup.at[index, 'backward_overlapping_area'] = (
                    calculate_ORA_single(frame_trajectoryDF=trajectoryDF[trajectoryDF['frame'] == row['max']],
                                                           target_id=row['id']))
        trajectoryGroup['forward_NN'] = trajectoryGroup['forward_overlapping_area'].apply(lambda row: list(filter_dict(row, overlapped_area_cutoff_for_NN).keys()))
        trajectoryGroup['forward_NN_counts'] = trajectoryGroup['forward_NN'].apply(lambda row: len(row))
        trajectoryGroup['forward_ID_counts'] = trajectoryGroup['forward_overlapping_area'].apply(lambda row: len(row) + 1)
        trajectoryGroup['backward_NN'] = trajectoryGroup['backward_overlapping_area'].apply(lambda row: list(filter_dict(row, overlapped_area_cutoff_for_NN).keys()))
        trajectoryGroup['backward_NN_counts'] = trajectoryGroup['backward_NN'].apply(lambda row: len(row))
        trajectoryGroup['backward_ID_counts'] = trajectoryGroup['backward_overlapping_area'].apply(lambda row: len(row) + 1)
    return trajectoryGroup

def annotate_trajectoryGroup(trajectoryDF, trajectoryGroup, object_number, length_max=500):
    frameDF = frame_statics(trajectoryDF=trajectoryDF, object_number=object_number)
    trajectoryGroup = cal_trajectory_duplication_pct(trajectoryGroup=trajectoryGroup, frameDF=frameDF)
    # trajectoryGroup = check_trajectory_complexity(trajectoryGroup=trajectoryGroup, frameDF=frameDF)
    trajectoryGroup = find_neighbours_for_trajectory(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, frameDF=frameDF,
                                                     object_number=object_number, length_max=length_max)
    trajectoryGroup = trajectoryGroup.reset_index(drop=True)
    return trajectoryGroup

def remove_redundant_frames_by_trajectory(trajectoryDF, trajectoryGroup, object_number, cores=4, length_max=400):
    # 去除一些短长度的trajectory，并且它们很可能是冗余的！为后续remove extra IDs减轻工作量！ 这一步不保证100%准确，即使有误删的也不会有太大麻烦，或者有些该删除的没有被删除，
    # 后续remove extra ID也可以删除
    trajectoryGroup = annotate_trajectoryGroup(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, object_number=object_number, length_max=length_max)
    # simple remove to remove short trajectory that are 100% duplicated and no id change
    trajectoryGroup_dup = trajectoryGroup[trajectoryGroup['id'] > object_number]  # 去除起始IDs
    trajectoryGroup_dup = trajectoryGroup_dup[trajectoryGroup_dup['max'] != trajectoryDF['frame'].max()]  # 去除末端IDs
    trajectoryGroup_dup = trajectoryGroup_dup[trajectoryGroup_dup['range'] <= trajectoryGroup_dup['length_cutoff']]  # 去除长度太长的
    if trajectoryGroup_dup.empty:
        print('Good, no redundant trajectory found, Doing nothing!')
        return trajectoryDF, trajectoryGroup
    else:
        # 第一种去除情况，该轨迹的frames 100%位于重复区域，并且前向或后向frame只有其中一个有1个最近邻。
        trajectoryGroup_dup_p1 = trajectoryGroup_dup[(trajectoryGroup_dup['duplication_pct'] == 1.0) &
                                                  ((trajectoryGroup_dup['forward_NN_counts'] > 0) | (trajectoryGroup_dup['backward_NN_counts'] > 0))]
        # 第二种去除情况，该轨迹的frames 至少30%位于重复区域，并且前向或后向frame都要有至少2个最近邻。
        trajectoryGroup_dup_p2 = trajectoryGroup_dup[((trajectoryGroup_dup['duplication_pct'] > 0.3) & (trajectoryGroup_dup['duplication_pct'] != 1.0)) &
                                                  ((trajectoryGroup_dup['forward_NN_counts'] > 1) &  (trajectoryGroup_dup['backward_NN_counts'] > 1))]
        trajectoryGroup_dup = pd.concat([trajectoryGroup_dup_p1, trajectoryGroup_dup_p2], ignore_index=False)
        trajectoryGroup_left = trajectoryGroup[~trajectoryGroup.index.isin(trajectoryGroup_dup.index)] # 有问题
        print('Attention: ' + str(trajectoryGroup_dup.shape[0]) + ' redundant trajectories bellow will be removed!')
        print(trajectoryGroup_dup)
        for _, row in tqdm(trajectoryGroup_dup.iterrows(), total=trajectoryGroup_dup.shape[0]):
            trajectoryDF = trajectoryDF[~(trajectoryDF['id'] == row['id'])]
        trajectoryDF = trajectoryDF.reset_index(drop=True)
        trajectoryGroup_left.reset_index(drop=True)
        return trajectoryDF, trajectoryGroup_left

def find_corresponding_trajectory_for_redundant_frames(redundant_frameGroup, trajectoryGroup):
    redundant_frameGroup['corresponding_trajectory_IDs_forward'] = redundant_frameGroup.apply(lambda _: [], axis=1)
    redundant_frameGroup['corresponding_trajectory_IDs_backward'] = redundant_frameGroup.apply(lambda _: [], axis=1)
    for index, row in tqdm(redundant_frameGroup.iterrows(), total=redundant_frameGroup.shape[0]):
        redundant_frameGroup.at[index, 'corresponding_trajectory_IDs_forward'] = trajectoryGroup.loc[trajectoryGroup['min'] == row['min'],'id'].to_list()
        redundant_frameGroup.at[index, 'corresponding_trajectory_IDs_backward'] = trajectoryGroup.loc[trajectoryGroup['max'] == row['max'], 'id'].to_list()
    # 检查如果左右两边均无对应的则报错！
    redundant_frameGroup['forward_IDs_counts'] = redundant_frameGroup['corresponding_trajectory_IDs_forward'].apply(lambda row: len(row))
    redundant_frameGroup['backward_IDs_counts'] = redundant_frameGroup['corresponding_trajectory_IDs_backward'].apply(lambda row: len(row))
    redundant_frameGroup_error = redundant_frameGroup[(redundant_frameGroup['forward_IDs_counts'] == 0) & (redundant_frameGroup['backward_IDs_counts'] == 0)]
    if not redundant_frameGroup_error.empty:
        print("Redundant frame group bellow can not found the corresponding trajectory:")
        print(redundant_frameGroup_error)
        # 或者直接加入一个plot function 参数
        raise Exception('Some redundant frame groups can not found the corresponding trajectory! You can view these ranges by plot_range function!')
    return redundant_frameGroup

def remove_redundant_frames_frame_level(trajectoryDF, trajectoryGroup, redundant_frameGroup):
    for index, row in tqdm(redundant_frameGroup.iterrows(), total=redundant_frameGroup.shape[0]):
        # 当匹配到多个ID时，保险起见，用最长的那个！
        # 左边对齐
        if row['forward_IDs_counts'] != 0:
            if row['forward_IDs_counts'] > 1:
                trajectoryGroup_sub = trajectoryGroup[trajectoryGroup.id.isin(row['corresponding_trajectory_IDs_forward'])]
                selected_ID = trajectoryGroup_sub.id[trajectoryGroup_sub['range'].idxmax()] #测试一下啊
            else:
                selected_ID = row['corresponding_trajectory_IDs_forward'][0]
            # trajectory length check must longer than redundant frame group
            if (trajectoryGroup.loc[trajectoryGroup.id == selected_ID, 'range'].to_list()[0] < row['range']):
                raise Exception('The frames length of corresponding trajectory is shorter than the redundant frame group!')
            # modify trajectoryGroup
            trajectoryGroup.loc[trajectoryGroup.id == selected_ID, 'min'] = row['max'] + 1
            # 去除trajectoryGroup中被完全去掉的轨迹
            trajectoryGroup = trajectoryGroup[trajectoryGroup['min'] <= trajectoryGroup['max']]
        # 右边对齐
        else:
            if row['backward_IDs_counts'] > 1:
                trajectoryGroup_sub = trajectoryGroup[trajectoryGroup.id.isin(row['corresponding_trajectory_IDs_backward'])]
                selected_ID = trajectoryGroup_sub.id[trajectoryGroup_sub['range'].idxmax()]
            else:
                selected_ID = row['corresponding_trajectory_IDs_backward'][0]
            print(selected_ID)
            # trajectory length check must longer than redundant frame group
            if (trajectoryGroup.loc[trajectoryGroup.id == selected_ID, 'range'].to_list[0] < row['range']):
                raise Exception('The frames length of corresponding trajectory is shorter than the redundant frame group!')
            # modify trajectoryGroup
            trajectoryGroup.loc[trajectoryGroup.id == selected_ID, 'max'] = row['min'] - 1
        # remove frames on trajectoryDF
        trajectoryDF = trajectoryDF[~((trajectoryDF['id'] == selected_ID) & (trajectoryDF.frame >= row['min']) & (trajectoryDF.frame <= row['max']))]
    return trajectoryDF, trajectoryGroup


def remove_redundant_frames(trajectoryDF, trajectoryGroup, object_number, cores=4):
    # 两步去除冗余frames
    # s1: 在轨迹水平去除冗余的frames
    # 即去除冗余的轨迹，1. 这些轨迹的所有绝大多数frames位于冗余区（> object number）
    # 2. 或者该轨迹的首尾frame均有两个最近邻，并且轨迹长度较短[短于（连续>object_number的序列的长度） * 1.5]，则去除此轨迹
    trajectoryDF, trajectoryGroup = remove_redundant_frames_by_trajectory(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, object_number=object_number, cores=cores)
    # s2: 直接从对应的轨迹上仅仅去除冗余frames，轨迹保留
    trajectoryDF, trajectoryGroup = remove_redundant_frames_by_frames(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, object_number=object_number)
    return trajectoryDF, trajectoryGroup

def remove_redundant_frames_by_frames(trajectoryDF, trajectoryGroup, object_number):
    # 发现所有redundant frame group均位于某个trajectory的左侧对齐。或是某个trajectory的右对齐！
    # 先考虑ID数目为最多的，比如8的这些frames，先执行去除，然后再考虑数目为7的
    # 首先看frame_redundant_group与哪些ID的起始位置是一致的，如果有多个，则用ID编号大的那个，注意长度要小于trajectory的长度！
    # 然后再看frame_redundant_group与哪些ID的终止位置是一致的，如果有多个，则用ID编号小的那个
    # 如果均不一致，则需要报错！
    frameDF = frame_statics(trajectoryDF=trajectoryDF, object_number=object_number)
    count_IDs_by_frame = frameDF['Counts'].value_counts().to_dict()
    numbers_more = sorted([key for key in count_IDs_by_frame.keys() if key > object_number],reverse=True)
    if len(numbers_more) != 0:
        for frame_ID_count in numbers_more:
            print("Processing redundant frames with ID counts equals to " + str(frame_ID_count) + "...")
            Redundant_framesGroup = group_numbers(frameDF.loc[frameDF['Counts'] == frame_ID_count, 'frame'].to_list(), cut=1)
            Redundant_framesGroup = find_corresponding_trajectory_for_redundant_frames(redundant_frameGroup=Redundant_framesGroup, trajectoryGroup=trajectoryGroup)
            trajectoryDF, trajectoryGroup = remove_redundant_frames_frame_level(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, redundant_frameGroup=Redundant_framesGroup)
        # 理论上，去除redundant frames后，每个ID仍旧是唯一的，检查一下
        if not (trajectoryGroup['id'].value_counts() == 1).all():
            raise Exception("Error, some ids are split to multiple parts!")
    else:
        print("No redundant frames found!")
    return trajectoryDF, trajectoryGroup


# def remove_extra_IDs(trajectoryDF, object_number):
#     # 对于每一个有多于object number的frame进行分析，去除ID出现晚的frame
#     # 问题是准确性太差，未考虑实际距离，应该去除参与互动的果蝇的frames
#     frame_df = frame_statics(trajectoryDF=trajectoryDF, object_number=object_number)
#     frame_df_more = frame_df[frame_df['Counts'] > object_number]
#     for _, row in tqdm(frame_df_more.iterrows(), total=frame_df_more.shape[0]):
#         IDs = row['IDs']
#         IDs.sort(reverse=False)
#         IDs_to_remove = IDs[6:]
#         for ID in IDs_to_remove:
#             trajectoryDF = trajectoryDF[~((trajectoryDF['id'] == ID) & (trajectoryDF['frame'] == row['frame']))]
#     trajectoryDF = trajectoryDF.reset_index(drop=True)
#     return trajectoryDF

# step2: remove extra IDs in frame with more IDs

def reverse_01_values(Alist):
    res = list()
    for i in Alist:
        if i == 1:
            res.append(0)
        else:
            res.append(1)
    return res


def find_new_ids(trajectoryDF, current_frame, object_number):
    df_current = trajectoryDF[trajectoryDF.frame == current_frame]
    current_frame_ids = df_current['id'].tolist()
    # 在以上这些ID中，谁出现的少，谁就是new ID.
    # df_previous = trajectoryDF[(trajectoryDF.frame <= current_frame) & trajectoryDF.id.isin(current_frame_ids)]
    # 为了加速以上代码，只考虑前300帧
    if current_frame < 300:
        df_previous = trajectoryDF[(trajectoryDF.frame <= current_frame) & trajectoryDF.id.isin(current_frame_ids)]
    else:
        df_previous = trajectoryDF[(trajectoryDF.frame <= current_frame) & (trajectoryDF.frame > (current_frame - 300)) & trajectoryDF.id.isin(current_frame_ids)]
    all_ids_sorted = df_previous.id.value_counts(sort=True, ascending=False).index.tolist()
    if len(all_ids_sorted) <= object_number:
        raise Exception("IDs from this frame not more than the object_number.")
    del all_ids_sorted[:object_number]
    return all_ids_sorted


def crosstab_data(trajectoryDF, frame, forward_extend=100, backward_extend=100):
    """
    对某段连续区域的frames，整理成01矩阵，表示ID是否出现。
    :param trajectoryDF:
    :param id_column:
    :param frame: 中心位置的frame
    :param forward_extend: 向前延申
    :param backward_extend:向后延申
    :return: a pandas dataframe, filled with 0/1 value
    """
    start = frame - forward_extend
    end = frame + backward_extend
    df_sub = trajectoryDF[(trajectoryDF['frame'] >= start) & (trajectoryDF['frame'] <= end)]
    mat = pd.crosstab(df_sub['id'], df_sub['frame'])
    return mat


def plot_heatmap(mat, file_name):
    """
    对输入的matrix绘制热图并保存。
    :param mat:  a pandas dataframe, filled with 0/1 value, rows are IDs, columns are frames.
    :param save_dir:
    :return: nothing
    """
    plt.figure(figsize=(36, 4))
    sns.heatmap(mat, cbar=False, cmap="YlGnBu", linewidths=0.5, xticklabels=1, yticklabels=1)  # 这一步骤每次好像闪出一个界面，导致绘图慢？
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def remove_extra_IDs_in_each_frame(trajectoryDF, save_dir, object_number, plot_frames=False):
    """
    对于ID数目大于实际数目的帧进行分析，去除多出来的ID[一般都是new ID]，多出来的ID是指同上一帧相比多出来的ID，所有所出来的ID都会被删除！ 这个地方其实有待更多思考，
    是迭代跟上上帧比较好呢？
    :param trajectoryDF:
    :param save_dir: 图片的保存目录，如果plot_frames=True
    :param object_number:
    :param plot_frames: 是否需要绘制ID多于实际数目的frame区域
    :return: a pandas dataframe, updated trajectory dataframe.
    """
    if plot_frames:
        plot_dir = os.path.join(save_dir, 'frames-with-more-IDs')
        print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.makedirs(plot_dir)
    frame_ID_counts = trajectoryDF['frame'].value_counts()
    # 哪些frames检测到的box数目多于object_number
    frame_more = sorted(frame_ID_counts[frame_ID_counts > object_number].index.tolist())
    frames_has_more_than_1_extra_IDs = list()
    print('Total found ' + str(len(frame_more)) + ' frames with more IDs, processing...')
    for current_frame in tqdm(frame_more):
        new_id = find_new_ids(trajectoryDF=trajectoryDF, current_frame=current_frame, object_number=object_number)
        if plot_frames:
            mat = crosstab_data(trajectoryDF, frame=current_frame, forward_extend=100, backward_extend=100)
            plot_heatmap(mat, file_name=f"{plot_dir}/frame-{current_frame}.png")
        if len(new_id) > 1:
            frames_has_more_than_1_extra_IDs.append(current_frame)
        if len(new_id) == 0:  # new_id为空，表示
            raise Exception("New ID not found!")
        for j in new_id:
            trajectoryDF = trajectoryDF[~((trajectoryDF['frame'] == current_frame) & (trajectoryDF['id'] == j))]
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    ## 检查一下remove_extra_IDs_in_each_frame运行后的结果，是否让所有frame的box数目<=object numbers
    if not np.all(trajectoryDF.frame.value_counts().value_counts().index <= object_number):
        print(trajectoryDF.frame.value_counts().value_counts())
        raise Exception("Remove extra IDs failed!")
    return trajectoryDF, len(frame_more), len(frames_has_more_than_1_extra_IDs)


# step3: solve ID jump

def xywh_to_polygon(x, y, w, h):
    """
   将xywh转为多边形，包括四个顶点的坐标
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    return Polygon([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)])

# overlapped_area_ratio 简写为 ORA
def calculate_ORA(frame_trajectoryDF):
    """
    对某一帧，计算每两个box重叠部分的面积比例。结果返回一个DataFrame，其中i表示行，j表示列，每个值为：OverlappedArea(i,j)/BoxArea(i)
    overlapped_area_ratio 简写为 ORA
    :param frame_trajectoryDF:
    :return: A Pandas DataFrame
    """
    # overlapped_area 用于存储各个ID的box之间重叠的面积比
    frame_trajectoryDF.index = frame_trajectoryDF['id'].tolist()
    overlapped_area = pd.DataFrame(columns=frame_trajectoryDF['id'].tolist(), index=frame_trajectoryDF['id'].tolist(), data=0.0)
    for i in overlapped_area.columns.tolist():
        # i表示行
        for j in overlapped_area.index.tolist():
            # j表示列，计算: OverlappedArea(i,j)/Area(i)
            if i != j:
                polygon = xywh_to_polygon(x=frame_trajectoryDF.loc[i, 'x'], y=frame_trajectoryDF.loc[i, 'y'],
                                          w=frame_trajectoryDF.loc[i, 'w'], h=frame_trajectoryDF.loc[i, 'h'])
                other_polygon = xywh_to_polygon(x=frame_trajectoryDF.loc[j, 'x'], y=frame_trajectoryDF.loc[j, 'y'],
                                                w=frame_trajectoryDF.loc[j, 'w'], h=frame_trajectoryDF.loc[j, 'h'])
                intersection = polygon.intersection(other_polygon)
                overlapped_area.loc[i, j] = round(intersection.area / frame_trajectoryDF.loc[i, 'BoxArea'], 3)
    return overlapped_area

def calculate_ORA_single(frame_trajectoryDF, target_id):
    """
    对某一帧，计算目标ID与其它ID的box之间的重叠面积, 重叠面积除以目标ID的box面积
    overlapped_area_ratio 缩写为 ORA
    :param frame_trajectoryDF:
    :return: A Dict
    """
    all_ids = frame_trajectoryDF['id'].tolist()
    if target_id not in all_ids:
        raise Exception('Error, target_id is not in this frame.')
    # set id as index
    frame_trajectoryDF = frame_trajectoryDF.set_index('id', drop=True)
    all_ids.remove(target_id)
    overlapped_area = {}
    target_id_polygon = xywh_to_polygon(x=frame_trajectoryDF.at[target_id, 'x'], y=frame_trajectoryDF.at[target_id, 'y'],
                              w=frame_trajectoryDF.at[target_id, 'w'], h=frame_trajectoryDF.at[target_id, 'h'])
    for i in all_ids:
        other_polygon = xywh_to_polygon(x=frame_trajectoryDF.at[i, 'x'], y=frame_trajectoryDF.at[i, 'y'],
                                        w=frame_trajectoryDF.at[i, 'w'], h=frame_trajectoryDF.at[i, 'h'])
        intersection = target_id_polygon.intersection(other_polygon)
        overlapped_area[i] = round(intersection.area / frame_trajectoryDF.at[target_id, 'BoxArea'], 3)
    return overlapped_area




def find_lost_id(trajectoryDF, object_number, new_id, new_id_start_frame, frames_length_considered, verbose=False):
    # lost_id, frame_space[new id start frame - lost_id last frame], distance_jumped[从lost id 到 new id，跳跃的距离]
    # 首先在start frame和end frame[目标区域，一般小于20frames]之间有lost的那些IDs [object number]，作为candidates。
    lost_id_candidates = list()
    start_frame = new_id_start_frame
    end_frame = start_frame + frames_length_considered - 1
    sub_DF = trajectoryDF[(trajectoryDF['frame'] >= start_frame) & (trajectoryDF['frame'] <= end_frame)]
    for i in range(1, object_number + 1):
        targetID_DF = sub_DF[sub_DF['id'] == i]
        if targetID_DF.shape[0] < frames_length_considered:
            lost_id_candidates.append(i)
    if verbose:
        print("all candidates with lost frames in the new id beginning frames: " + str(lost_id_candidates))
    # 如果candidates数目大于1，首先依据overlapped frames数目进行排序，取重叠数目最少的，然后如果取重叠数目最少的有多个，则继续取距离最近的。
    if len(lost_id_candidates) >= 1:
        # if len(lost_id_candidates) > 1:
        #     print("found more than 1 candidates...")
        # new ID的起始位置
        x1 = trajectoryDF['x'][(trajectoryDF['frame'] == new_id_start_frame) & (trajectoryDF['id'] == new_id)].values[0]
        y1 = trajectoryDF['y'][(trajectoryDF['frame'] == new_id_start_frame) & (trajectoryDF['id'] == new_id)].values[0]
        total_overlapped_frames_list = list()  # new ID与candidate ID在目标区域overlapped frames的数目
        candidate_id_last_frame_list = list()  # candidate id的last frame
        frame_space_list = list()  # candidate id的last frame与new id start frame的间隔
        dis_list = list()  # candidate id的last frame的位置与new id的start frame位置之间的距离
        for j in lost_id_candidates:
            # 先计算在目标区域，overlapped frames的数目
            new_id_total_frames = trajectoryDF['frame'][trajectoryDF['id'] == new_id].tolist()
            candidate_total_frames = trajectoryDF['frame'][trajectoryDF['id'] == j].tolist()
            total_frames_overlapped = len(set(new_id_total_frames) & set(candidate_total_frames))
            total_overlapped_frames_list.append(total_frames_overlapped)
            # 再计算candidate id的last frame的位置与new id的start frame位置之间的距离
            candidate_id_last_frame = new_id_start_frame
            while True:
                target_frame = trajectoryDF[(trajectoryDF['frame'] == candidate_id_last_frame) & (trajectoryDF['id'] == j)]
                if target_frame.empty:  # 修正last frame
                    candidate_id_last_frame = candidate_id_last_frame - 1
                    continue
                else:
                    x2 = target_frame['x'].values[0]
                    y2 = target_frame['y'].values[0]
                    candidate_id_last_frame_list.append(candidate_id_last_frame)
                    frame_space_list.append(new_id_start_frame - candidate_id_last_frame - 1)
                    dis_list.append(round(calculate_distance([x1, y1], [x2, y2]), 3))
                    break
        if verbose:
            print('overlapped frames numbers: ' + str(total_overlapped_frames_list))
            print('candidate id last frame: ' + str(candidate_id_last_frame_list))
            print('candidate id last frame与new id start frame的frames spaced: ' + str(frame_space_list))
            print('candidate id 与new id之间的距离: ' + str(dis_list))
        candidates_count = total_overlapped_frames_list.count(min(total_overlapped_frames_list))
        if candidates_count != 1:  # 如果得到多个candidates，则依据距离匹配。
            # print("Found " + str(candidates_count) + " candidates based on the least overlapped frames, " +
            # "the lost id will be assigned to the nearest object.")
            # 只保留overlapped frames最少的那些值
            least_overlapped_index = np.where(np.array(total_overlapped_frames_list) == min(total_overlapped_frames_list))[0].tolist()
            lost_id_candidates = [lost_id_candidates[x] for x in least_overlapped_index]
            candidate_id_last_frame_list = [candidate_id_last_frame_list[x] for x in least_overlapped_index]
            frame_space_list = [frame_space_list[x] for x in least_overlapped_index]
            dis_list = [dis_list[x] for x in least_overlapped_index]
            # 依据最小距离来判定lost id
            lost_id_index = dis_list.index(min(dis_list))

        else:
            lost_id_index = total_overlapped_frames_list.index(min(total_overlapped_frames_list))  # 否则依据最少overlap来判定
        lost_id = lost_id_candidates[lost_id_index]
        lost_id_last_frame = candidate_id_last_frame_list[lost_id_index]
        frame_space = frame_space_list[lost_id_index]
        distance_jumped = dis_list[lost_id_index]
        return lost_id, lost_id_last_frame, frame_space, distance_jumped, candidates_count
    else:
        print("Warning: Can not found the lost id.")
        return None, None, None, None, 0


def calculate_frame_space(mat, lost_id, start_frame):
    """
    计算lost id与 new id之间的gap。
    :param mat: crosstab_data function 的输出结果
    :param lost_id:
    :param start_frame: new id的起始位置。
    :return: frame space value, and the last frame of lost id
    """
    all_1_frames = mat.columns.values[mat.loc[lost_id] == 1]
    diff = start_frame - all_1_frames
    which_min = diff == min(abs(diff))
    return diff[which_min][0], all_1_frames[which_min][0]


def calculate_coor(x1, x2, n, th, digits=7):
    """
    平滑填充x1和x2中间的缺失值
    :param digits:
    :param x1: 起始位置
    :param x2: 终止位置
    :param n: 缺失值的数目
    :param th: 第几个缺失值
    :return:
    """
    return round(x1 + (x2 - x1) * th / (n + 1), digits)


def smooth_insert(trajectoryDF, new_id, lost_id, new_id_frame, lost_id_frame, digits_xy=6, digits_wh=7):
    """
    ID jump中，对于是由移动过快导致的，且无遮挡的gap，进行平滑填充。填充结果位于dataframe的最后面。
    :param digits_wh:
    :param digits_xy:
    :param trajectoryDF:
    :param new_id: new id of the new trajectory
    :param lost_id:  the lost id
    :param new_id_frame: the start frame of new id
    :param lost_id_frame: the last frame of the lost id
    :return: a pandas dataframe, the updated trajectory.
    """
    blank_frames = range(lost_id_frame + 1, new_id_frame)
    n = len(blank_frames)
    x1 = trajectoryDF['x'][(trajectoryDF['frame'] == lost_id_frame) & (trajectoryDF['id'] == lost_id)].values[0]
    y1 = trajectoryDF['y'][(trajectoryDF['frame'] == lost_id_frame) & (trajectoryDF['id'] == lost_id)].values[0]
    w1 = trajectoryDF['w'][(trajectoryDF['frame'] == lost_id_frame) & (trajectoryDF['id'] == lost_id)].values[0]
    h1 = trajectoryDF['h'][(trajectoryDF['frame'] == lost_id_frame) & (trajectoryDF['id'] == lost_id)].values[0]
    x2 = trajectoryDF['x'][(trajectoryDF['frame'] == new_id_frame) & (trajectoryDF['id'] == new_id)].values[0]
    y2 = trajectoryDF['y'][(trajectoryDF['frame'] == new_id_frame) & (trajectoryDF['id'] == new_id)].values[0]
    w2 = trajectoryDF['w'][(trajectoryDF['frame'] == new_id_frame) & (trajectoryDF['id'] == new_id)].values[0]
    h2 = trajectoryDF['h'][(trajectoryDF['frame'] == new_id_frame) & (trajectoryDF['id'] == new_id)].values[0]
    for i in range(n):
        x = calculate_coor(x1=x1, x2=x2, n=n, th=i + 1, digits=digits_xy)
        y = calculate_coor(x1=y1, x2=y2, n=n, th=i + 1, digits=digits_xy)
        w = calculate_coor(x1=w1, x2=w2, n=n, th=i + 1, digits=digits_wh)
        h = calculate_coor(x1=h1, x2=h2, n=n, th=i + 1, digits=digits_wh)
        box_area = round(w * h, 7)
        df = pd.DataFrame({'frame': int(blank_frames[i]), 'category': int(0), 'x': x, 'y': y, 'w': w, 'h': h, 'id': int(lost_id),
                           'BoxArea': box_area, 'Inserted': True, 'Inserted_method': 'smooth_fill_all_gaps'}, index=[0])
        trajectoryDF = pd.concat([trajectoryDF, df], ignore_index=False)
    return trajectoryDF

def assigned_insert(trajectoryDF, id, lost_from, lost_to, by):
    """
    ID jump中，对于是由移动过快导致的，且无遮挡的gap，进行平滑填充。填充结果位于dataframe的最后面。
    :param trajectoryDF:
    :param id: id of the object
    :param lost_from:  the start frame of gap
    :param lost_to: the end frame of gap
    :param by: the frame used to fill lost frames
    :return: a pandas dataframe, of updated trajectory.
    """
    frame_by = trajectoryDF[(trajectoryDF['frame'] == by) & (trajectoryDF['id'] == id)]
    blank_frames = list(range(lost_from, lost_to + 1))
    df = frame_by.loc[frame_by.index.repeat(len(blank_frames))]
    df['frame'] = blank_frames
    trajectoryDF = pd.concat([trajectoryDF, df], ignore_index=False)
    return trajectoryDF


def extract_frame(cap, current_frame):
    """
    从视频中，提取特定帧的图像
    :param cap:
    :param current_frame:
    :return:
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        raise Exception('Sorry, frame read failed!')


def xywh2xyxy(x, y, w, h):
    """
    将yolo predicted box的坐标形式从x,y,w,h转变为左上角和右下角两个点的坐标。
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


def xyxyn2xyxy(x1, y1, x2, y2, width=1920, height=1080):
    """
    将yolo predicted box的坐标形式从x,y,w,h转变为左上角和右下角两个点的实际坐标[像素点]。
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param width:
    :param height:
    :return:
    """
    x1, x2 = x1 * width, x2 * width
    y1, y2 = y1 * height, y2 * height
    return x1, y1, x2, y2


def crop_image(image, x1, y1, x2, y2):
    """
    从图像中截取box，提供box的左上和右下两个点的坐标[实际像素点]即可。
    starting point: height, and ending point: width
    x1, y1, x2, y2 must be integers
    :param image:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    # starting point: height, and ending point: width
    # x1, y1, x2, y2 must be integers
    return image[int(y1):int(y2), int(x1):int(x2)]


def crop_image_by_xywhn(image, x, y, w, h, width=1920, height=1080):
    """
    从图像中截取box，提供box的左上和右下两个点的坐标[比例位置]即可。
    :param image:
    :param x:
    :param y:
    :param w:
    :param h:
    :param width:
    :param height:
    :return:
    """
    x1, y1, x2, y2 = xywh2xyxy(x=x, y=y, w=w, h=h)
    x1, y1, x2, y2 = xyxyn2xyxy(x1=x1, y1=y1, x2=x2, y2=y2, width=width, height=height)
    image_cropped = crop_image(image=image, x1=x1, y1=y1, x2=x2, y2=y2)
    return image_cropped


def crop_image_by_xywh(image, x, y, w, h):
    """
    从图像中截取box，提供box的左上和右下两个点的坐标[比例位置]即可。
    :param image:
    :param x:
    :param y:
    :param w:
    :param h:
    :param width:
    :param height:
    :return:
    """
    x1, y1, x2, y2 = xywh2xyxy(x=x, y=y, w=w, h=h)
    image_cropped = crop_image(image=image, x1=x1, y1=y1, x2=x2, y2=y2)
    return image_cropped


def extract_and_merge_objects(cap, frame_start, frame_end, trajectoryDF, image_unit_height=100,
                              image_unit_width=100):
    """
    Extract and Merge each object from trajectory file from a frame range
    :param cap: the cap read in by cv2.VideoCapture
    :param frame_start: frame to start
    :param frame_end: frame to stop
    :param trajectoryDF: has frame, x, y, w, h and id columns at least.
    :param image_unit_height: the height for each object
    :param image_unit_width: the width for each object
    :return: a merged image
    """
    trajectoryDF_sub = trajectoryDF[(trajectoryDF['frame'] >= frame_start) &
                                    (trajectoryDF['frame'] <= frame_end)]
    all_ids = trajectoryDF_sub.id.unique().tolist()
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_white = np.zeros((image_unit_height, image_unit_width, 3), dtype=np.uint8)
    img_white[:, :] = (255, 255, 255)  # 白色空白图，用于生成row_index， column_index， and for lost ids
    img_final = img_white.copy()
    # 左上角图片
    cv2.putText(img_final, 'Frame', (0, int(image_unit_height / 2 + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 先生成第一列的第一列图，包括Frame和ID
    for id in all_ids:
        img_index = img_white.copy()
        cv2.putText(img_index, str(id), (image_unit_width - 50, int(image_unit_height / 2 + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img_final = np.concatenate((img_final, img_index), axis=0)
    # 生成每个frame的合并图，并且做大合并
    for frame in list(range(frame_start, frame_end + 1)):
        frame_image = extract_frame(cap=cap, current_frame=frame)
        trajectory_df = trajectoryDF_sub[trajectoryDF_sub.frame == frame]
        frame_ids = trajectory_df.id.unique().tolist()
        trajectory_df = trajectory_df.set_index('id')
        # 生成下一列的第一张图
        img_next = img_white.copy()
        cv2.putText(img_next, str(frame), (0, image_unit_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # 生成下一列的下一张图
        for id in all_ids:
            if id in frame_ids:
                if not isinstance(trajectory_df.loc[id]['x'], float):
                    print("All IDs: " + str(all_ids))
                    print("Frame IDs: " + str(frame_ids))
                    print(trajectoryDF_sub[trajectoryDF_sub['id'] == id])
                    print(trajectory_df)
                if np.all(trajectory_df.x.to_numpy() <= 1) & np.all(trajectory_df.y.to_numpy() <= 1):  # xywhn format
                    image_cropped = crop_image_by_xywhn(image=frame_image, x=trajectory_df.loc[id]['x'],
                                                        y=trajectory_df.loc[id]['y'], w=trajectory_df.loc[id]['w'],
                                                        h=trajectory_df.loc[id]['h'], width=video_width, height=video_height)
                else:
                    image_cropped = crop_image_by_xywh(image=frame_image, x=trajectory_df.loc[id]['x'],
                                                       y=trajectory_df.loc[id]['y'], w=trajectory_df.loc[id]['w'],
                                                       h=trajectory_df.loc[id]['h'])
                # 将照片缩放为指定大小
                img_single = cv2.resize(image_cropped, (image_unit_width, image_unit_height), interpolation=cv2.INTER_NEAREST)
            else:
                img_single = img_white.copy()
            img_next = np.concatenate((img_next, img_single), axis=0)
        img_final = np.concatenate((img_final, img_next), axis=1)
    return img_final


def calculate_distance(x1y1, x2y2):
    x1 = x1y1[0]
    y1 = x1y1[1]
    x2 = x2y2[0]
    y2 = x2y2[1]
    dx = x2 - x1
    dy = y2 - y1
    dis = np.sqrt(dx ** 2 + dy ** 2)
    return dis

def plot_range(trajectoryDF, frame_range, cap, object_number, save_dir, extend_forward = 40, extend_backward = 40, prefix='frame-ranges'):
    plot_dir = os.path.join(save_dir, prefix)
    print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.makedirs(plot_dir)
    for _, row in tqdm(frame_range.iterrows(), total=frame_range.shape[0]):
        frame_start = max(1, row['min'] - extend_forward)
        frame_end = min(trajectoryDF['frame'].max(), row['max'] + extend_backward)
        img = extract_and_merge_objects(cap=cap,
                                        frame_start=frame_start,
                                        frame_end=frame_end,
                                        trajectoryDF=trajectoryDF,
                                        image_unit_height=100,
                                        image_unit_width=100)
        plot_filename = os.path.join(plot_dir, str('frame-from-' + str(frame_start) + '-to-frame-' + str(frame_end) + '.png'))
        cv2.imencode('.jpg', img)[1].tofile(plot_filename)

def plot_ID_jump(trajectoryDF, trajectoryGroup, cap, object_number, save_dir, extend_forward = 40, extend_backward = 40, prefix='frames-with-ID-jump'):
    # 与plot_range的区别是，仅在新ID出现的位置进行绘图，而不是整个range区域，如果两个新ID相差不超过10就会被合并出图
    frame_range = trajectoryGroup.loc[trajectoryGroup['id'] > object_number]
    frame_range = group_numbers(frame_range['min'], cut=10)
    plot_range(trajectoryDF=trajectoryDF, frame_range=frame_range, cap=cap, object_number=object_number,
               save_dir=save_dir, extend_forward=extend_forward, extend_backward=extend_backward, prefix=prefix)


def resolve_ID_jump(trajectoryDF, object_number, cap, save_dir, plot_img=True, max_overlap_ratio=0.2, verbose=False):
    """
    解决ID jump问题，将不连续的trajectory整合为连续的trajectory，不确定的trajectory [frame 中 loss ID 的 max_overlap_ratio > 0.2,
    则保留或者引入gap，后续再结合其他信息解决gap]，最终返回的trajectory为一个相对整洁的trajectory，ID数目是正确的。
    :param trajectoryDF:
    :param object_number:
    :param cap:
    :param save_dir:
    :param plot_img: 是否绘制ID jump区域修正之前和修正之后的图像。
    :param max_overlap_ratio: 当前ID的box与其他ID的box面积的overlap比例，如果大于0.2，则保留gap，否则平滑填充gap，不保留gap。
    :param verbose: 是否在控制台输出每个jump区域的详细信息
    :return:
    """
    plot_dir = os.path.join(save_dir, 'frames-with-ID-jump')
    if plot_img:
        print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.makedirs(plot_dir)
    logs_filepath = os.path.join(save_dir, 'ID-jump-logs.csv')
    total_IDs = trajectoryDF['id'].unique().tolist()
    init_IDs = [i + 1 for i in list(range(object_number))]
    new_IDs = sorted(list(set(total_IDs) - set(init_IDs)))
    IDs_has_more_candidates = 0
    # 注意： 这些new_IDs必须要按照第一次的出现位置进行排序！
    new_IDs_first_position = []
    for new_id in new_IDs:
        new_id_start_index = trajectoryDF[trajectoryDF['id'] == new_id].first_valid_index()
        new_IDs_first_position.append(trajectoryDF['frame'][new_id_start_index])
    new_IDs_sorted = [x for _, x in sorted(zip(new_IDs_first_position, new_IDs))]
    print('Total found ' + str(len(new_IDs)) + ' new IDs, resolving ID jump...')
    # 对于每一个new ID，寻找与其对应的上一个丢失的ID
    for new_id in tqdm(new_IDs_sorted):
        # new ID 最开始出现的位置的index
        new_id_start_index = trajectoryDF[trajectoryDF['id'] == new_id].first_valid_index()
        #  new ID 最开始出现的位置
        new_id_start_frame = trajectoryDF['frame'][new_id_start_index]
        # new ID 最开始出现位置共有多少个ID
        id_counts_start_frame = len(trajectoryDF['id'][trajectoryDF['frame'] == new_id_start_frame].tolist())
        # new ID在当起始frame是否有近邻，计算最大overlap的面积
        overlap_mat = calculate_ORA(frame_trajectoryDF=trajectoryDF[trajectoryDF['frame'] == new_id_start_frame])
        max_ratio = overlap_mat.max(axis=1)[new_id]
        # 绘制此区域各个ID的整合图像，用于人工判定可能交换的ID
        if plot_img:
            img = extract_and_merge_objects(cap=cap,
                                            frame_start=new_id_start_frame - 40,
                                            frame_end=new_id_start_frame + 40,
                                            trajectoryDF=trajectoryDF,
                                            image_unit_height=100,
                                            image_unit_width=100)
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/frame-{new_id_start_frame}-MaxRatio-{max_ratio}.png")
        # new ID的总frame数目
        new_id_frames_length = trajectoryDF[trajectoryDF['id'] == new_id].shape[0]
        # 考虑的区间范围 frame的数目，最大20frames，在此范围内寻找上游丢失的ID
        frames_length_considered = min(20, new_id_frames_length)
        lost_id, lost_id_last_frame, frame_space, distance_jumped, candidates_count = (
            find_lost_id(trajectoryDF=trajectoryDF,
                         object_number=object_number,
                         new_id=new_id,
                         new_id_start_frame=new_id_start_frame,
                         frames_length_considered=frames_length_considered,
                         verbose=verbose))
        if candidates_count == 0:
            # 如果没有找对丢失的id，则仅仅删除当前id
            trajectoryDF = trajectoryDF.loc[trajectoryDF.id != new_id]
            trajectoryDF = trajectoryDF.reset_index(drop=True)
            continue
        else:
            if candidates_count > 1:
                IDs_has_more_candidates += 1
            if verbose:
                print('\nProcessing ID: ' + str(new_id) + ' , which appeared in frame ' + str(new_id_start_frame) + ', Max Overlap Ratio: ' +
                      str(max_ratio) + ', Lost ID: ' + str(lost_id) + ', Frame spaced: ' + str(frame_space) +
                      ', Distance jumped: ' + str(distance_jumped))
            new_id_row_index = trajectoryDF.index[trajectoryDF['id'] == new_id]
            new_id_frames = trajectoryDF.loc[new_id_row_index]['frame'].values
            # 如果lost id后续的frames与new id frames有overlap，则先要去除old id的这些frames
            lost_id_frames_followup_overlapped_index = trajectoryDF.index[(trajectoryDF['id'] == lost_id) & (trajectoryDF['frame'].isin(new_id_frames))]
            if not lost_id_frames_followup_overlapped_index.empty:
                trajectoryDF = trajectoryDF.drop(lost_id_frames_followup_overlapped_index, axis=0)
            # 对于jump情况: 如果这个jump可以修改后，ID数目等于object number
            if (max_ratio < max_overlap_ratio) and (id_counts_start_frame == object_number):
                if frame_space > 0:
                    # 平滑填充插入空出的frame
                    trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF,
                                                 new_id=new_id,
                                                 lost_id=lost_id,
                                                 new_id_frame=new_id_start_frame,
                                                 lost_id_frame=lost_id_last_frame)
                # 修正ID
                trajectoryDF.loc[new_id_row_index, "id"] = lost_id
            else:
                if frame_space <= 0:
                    # 新引入一个gap， 删除newID的第一帧
                    trajectoryDF = trajectoryDF.drop(new_id_row_index[0], axis=0)
                    new_id_row_index = new_id_row_index.delete([0])
                # 修正ID
                trajectoryDF.loc[new_id_row_index, "id"] = lost_id
            trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
            trajectoryDF = trajectoryDF.reset_index(drop=True)
            if plot_img:
                img = extract_and_merge_objects(cap=cap,
                                                frame_start=new_id_start_frame - 40,
                                                frame_end=new_id_start_frame + 40,
                                                trajectoryDF=trajectoryDF,
                                                image_unit_height=100,
                                                image_unit_width=100)
                cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/frame-{new_id_start_frame}-MaxRatio-{max_ratio}-revised.png")
            if verbose:
                print("After this revision:" + str(trajectoryDF.id.value_counts()))
        logs_summary = pd.DataFrame(dict(new_id=new_id,
                                         old_id=lost_id,
                                         frame_start=new_id_start_frame), index=[0])
        if os.path.exists(logs_filepath):
            logs_summary.to_csv(logs_filepath, mode='a', header=False, index=False)
        else:
            logs_summary.to_csv(logs_filepath, header=True, index=False)
    return trajectoryDF, len(new_IDs), IDs_has_more_candidates


# step4: resolve gaps

## 4.1 get and summary all gaps

def get_frames_gaps(frames, max, min=1):
    """
    frames中的缺失值，默认frame从1开始
    :param frames:
    :param max:
    :return:
    """
    return sorted(list(set(range(min, max + 1)) - set(frames)))


def parallel_group_gaps(trajectoryDF, group_by, frame_start=1, cores=4, cut=1):
    """
    多线程方式，对 dataframe依据id进行分组，然后得到每个ID所有trajectory【连续frame】的range信息，包括起始位置，终止位置，长度
    return: a pandas dataframe
    """

    def data_process(name, group):
        res = group_numbers(get_frames_gaps(group['frame'].tolist(), max=trajectoryDF['frame'].max(), min=frame_start), cut=cut)
        res = res.assign(id=name)
        return res

    def applyParallel(dfGrouped, func):
        # 多线程方式，执行某个function，并合并结果
        res = Parallel(n_jobs=cores)(delayed(func)(name, group) for name, group in dfGrouped)
        return pd.concat(res)

    df = trajectoryDF.groupby(group_by)
    result = applyParallel(df, data_process)
    result = result.reset_index(drop=True)
    return result


def summary_gaps(trajectoryDF, object_number, cores=4, max_overlap_ratio=0.2):
    """
    获取所有的gaps，并且统计gap信息，比如最大重叠面积，最近邻居，并做分类，
    simple：gap前后均无最近邻，且无重叠的gap。
    complex: gap前后均无最近邻，但有gap重叠，这种也有可能发生ID switch
    complex-1:有最近邻，且最近邻稳定
    complex-2：有最近邻，但最近邻发生变化。
    :param trajectoryDF:
    :param cores:
    :param max_overlap_ratio: 定义是否有最近邻的阈值
    :return:
    """
    # 第一步，先找出所有连续的gaps： group     min     max  range  middle  id
    gapDF = parallel_group_gaps(trajectoryDF=trajectoryDF, group_by='id', cores=cores)
    print('Total found ' + str(gapDF.shape[0]) + ' gaps, summarise...')
    # gap open overlap-ratio and close overlap-ratio, if both ratio less than a cut off， so smooth insert.
    gap_before_max_ratio = list()
    gap_after_max_ratio = list()
    gap_before_nearest_neighbour = list()
    gap_after_nearest_neighbour = list()
    gap_before_all_neighbours = list()
    gap_after_all_neighbours = list()
    gap_type = list()  # 1. simple; 2. complex; 3. complex-1; 4. complex-2;
    for gap_index in tqdm(range(gapDF.shape[0])):
        gap_before = gapDF['min'].iloc[gap_index] - 1
        gap_after = gapDF['max'].iloc[gap_index] + 1
        id = gapDF['id'].iloc[gap_index]
        gap_range = gapDF['range'].iloc[gap_index]
        # 这个gap所在的frame区间，是否也存在其它gap，如果存在，后续标为complex
        trajectory_sub = trajectoryDF[(trajectoryDF['frame'] >= gapDF['min'].iloc[gap_index]) &
                                      (trajectoryDF['frame'] <= gapDF['max'].iloc[gap_index])]
        row_counts = trajectory_sub[trajectory_sub['id'] != id].shape[0]
        no_gap_counts = gap_range * (object_number - 1)
        if row_counts < no_gap_counts:
            has_overlap = True
        else:
            has_overlap = False
        # 如果gap前一帧，也存在gap，那么就需要使用更前一帧.
        # 但是，如果由不相关的果蝇组成多于两个gap，可能也会导致结果不准确！！！
        frame_trajectoryDF = trajectoryDF[trajectoryDF['frame'] == gap_before]
        frame_object_numbers = frame_trajectoryDF.shape[0]
        while frame_object_numbers < object_number:
            gap_before = gap_before - 1
            frame_trajectoryDF = trajectoryDF[trajectoryDF['frame'] == gap_before]
            frame_object_numbers = frame_trajectoryDF.shape[0]
        overlap_mat = calculate_ORA(frame_trajectoryDF=frame_trajectoryDF)
        before_max_ratio = overlap_mat.max(axis=1)[id]
        gap_before_max_ratio.append(before_max_ratio)
        before_neighbour = overlap_mat.idxmax(axis=1)[id]
        gap_before_nearest_neighbour.append(before_neighbour)
        before_all_neighbours = overlap_mat[id][overlap_mat[id] > max_overlap_ratio].index.tolist()
        before_all_neighbours = [str(i) for i in before_all_neighbours]
        gap_before_all_neighbours.append(','.join(before_all_neighbours))
        # 如果gap后一帧，也存在gap，那么就需要使用更后一帧.
        # 但是，如果由不相关的果蝇组成多于两个gap，可能也会导致结果不准确！！！
        frame_trajectoryDF = trajectoryDF[trajectoryDF['frame'] == gap_after]
        frame_object_numbers = frame_trajectoryDF.shape[0]
        while frame_object_numbers < object_number:
            gap_after = gap_after + 1
            frame_trajectoryDF = trajectoryDF[trajectoryDF['frame'] == gap_after]
            frame_object_numbers = frame_trajectoryDF.shape[0]
        overlap_mat = calculate_ORA(frame_trajectoryDF=frame_trajectoryDF)
        after_max_ratio = overlap_mat.max(axis=1)[id]
        gap_after_max_ratio.append(after_max_ratio)
        after_neighbour = overlap_mat.idxmax(axis=1)[id]
        gap_after_nearest_neighbour.append(after_neighbour)
        after_all_neighbours = overlap_mat[id][overlap_mat[id] > max_overlap_ratio].index.tolist()
        after_all_neighbours = [str(i) for i in after_all_neighbours]
        gap_after_all_neighbours.append(','.join(after_all_neighbours))
        # 引入start_id_counts和end_id_counts，如果缺失的ID数目大于2，则也认为是complex情况？？？？
        if has_overlap:
            gap_type.append('complex')
        elif (before_max_ratio < max_overlap_ratio) and (after_max_ratio < max_overlap_ratio):
            gap_type.append('simple')
        elif before_neighbour != after_neighbour:
            gap_type.append('complex-2')
        else:
            gap_type.append('complex-1')
    gapDF = gapDF.assign(gap_before_max_ratio=gap_before_max_ratio)
    gapDF = gapDF.assign(gap_after_max_ratio=gap_after_max_ratio)
    gapDF = gapDF.assign(gap_before_nearest_neighbour=gap_before_nearest_neighbour)
    gapDF = gapDF.assign(gap_after_nearest_neighbour=gap_after_nearest_neighbour)
    gapDF = gapDF.assign(gap_type=gap_type)
    gapDF = gapDF.assign(gap_before_all_neighbours=gap_before_all_neighbours)
    gapDF = gapDF.assign(gap_after_all_neighbours=gap_after_all_neighbours)
    gapDF = gapDF.sort_values(by='min')
    gapDF = gapDF.reset_index(drop=True)
    return gapDF


## 4.2 resolve gaps with simple type

def fill_gaps_simple(trajectoryDF, gapsDF, cap, save_dir, plot_img=True):
    """
    solve gaps with simple type: fly with gaps has no overlap with other fly.
    :param trajectoryDF:
    :param gapsDF:
    :param cap:
    :param save_dir:
    :param plot_img:
    :return:
    """
    plot_dir = f"{save_dir}/gaps-simple"
    if plot_img:
        print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.mkdir(plot_dir)
    gapsDF_simple = gapsDF[gapsDF['gap_type'] == 'simple']
    gapsDF_simple = gapsDF_simple.reset_index(drop=True)
    print('Total found ' + str(gapsDF_simple.shape[0]) + ' gaps with simple type, filling gaps...')
    for i in tqdm(gapsDF_simple.index):
        id = gapsDF_simple['id'].iloc[i]
        gap_start = gapsDF_simple['min'].iloc[i]
        gap_end = gapsDF_simple['max'].iloc[i]
        if plot_img:
            img = extract_and_merge_objects(cap=cap,
                                            frame_start=gap_start - 40,
                                            frame_end=gap_end + 40,
                                            trajectoryDF=trajectoryDF,
                                            image_unit_height=100,
                                            image_unit_width=100)
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/gap-{gap_start}-to-{gap_end}.png")
        trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF, new_id=id, lost_id=id, new_id_frame=gap_end + 1,
                                     lost_id_frame=gap_start - 1)
        trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
        trajectoryDF = trajectoryDF.reset_index(drop=True)
        if plot_img:
            img = extract_and_merge_objects(cap=cap,
                                            frame_start=gap_start - 40,
                                            frame_end=gap_end + 40,
                                            trajectoryDF=trajectoryDF,
                                            image_unit_height=100,
                                            image_unit_width=100)
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/gap-{gap_start}-to-{gap_end}-revised.png")
        gapsDF = gapsDF[gapsDF['gap_type'] != 'simple']
    print("Congratulations, total " + str(trajectoryDF.Inserted.sum()) + " fake detection boxes are inserted...")
    return trajectoryDF, gapsDF


## 4.3 plot all remain gaps

def generate_colors_dict(ids):
    """
    返回一个dict，name为ids，value为颜色
    :param ids: a id list
    :return: a dict with id as names, and value is colors
    """
    # generate colors, values range from 0 to 1
    ids_colors = distinctipy.get_colors(len(ids))
    # trans values range to 0-255
    for ii in range(len(ids_colors)):
        ids_colors[ii] = tuple(map(lambda x: int(x * 255), ids_colors[ii]))
    colors_dict = dict(zip(ids, ids_colors))
    return colors_dict


def add_rectangle(img, color, x, y, w, h, video_width=1920, video_height=1080):
    """
    使用yolo track的xywhn信息，给原始的raw video frame绘制box，圈出flys
    :param img:
    :param color:
    :param x:
    :param y:
    :param w:
    :param h:
    :param video_width:
    :param video_height:
    :return:
    """
    x_min = math.ceil((float(x) - float(w) / 2) * video_width)
    x_max = math.floor((float(x) + float(w) / 2) * video_width)
    y_min = math.ceil((float(y) - float(h) / 2) * video_height)
    y_max = math.floor((float(y) + float(h) / 2) * video_height)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=3)
    return img


def video_error_region(raw_cap, anno_cap, trajectoryDF, problem_ids, frame_start, frame_end, file_name):
    # 存储一些video信息，用于video输出时的参数设置
    fps = raw_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    trajectoryDF_sub = trajectoryDF[(trajectoryDF['frame'] >= frame_start) & (trajectoryDF['frame'] <= frame_end)]
    # 给所有果蝇设定颜色
    all_ids = trajectoryDF_sub['id'].unique().tolist()
    # 为每一种ID生成以一种对应颜色
    colors_dict = generate_colors_dict(all_ids)
    # output video info
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(file_name, fourcc, fps, (video_width, video_height))
    for Aframe in range(frame_start, frame_end + 1):
        raw_cap.set(cv2.CAP_PROP_POS_FRAMES, Aframe - 1)
        _, raw_frame = raw_cap.read()
        anno_cap.set(cv2.CAP_PROP_POS_FRAMES, Aframe - 1)
        _, annotated_frame = anno_cap.read()
        # 深复制: 标记所有的fly
        raw_frame_label_all = raw_frame.copy()
        # subset track results by frame
        track_framex_all = trajectoryDF[trajectoryDF.frame == Aframe]
        # 标记所有fly
        if not track_framex_all.empty:
            for jj in range(track_framex_all.shape[0]):
                add_rectangle(img=raw_frame_label_all, color=colors_dict[track_framex_all.iloc[jj, 6]],
                              x=track_framex_all.iloc[jj, 2], y=track_framex_all.iloc[jj, 3],
                              w=track_framex_all.iloc[jj, 4], h=track_framex_all.iloc[jj, 5],
                              video_width=video_width, video_height=video_height)
        # 深复制： 仅仅标记有问题的fly
        raw_frame_label_problems = raw_frame.copy()
        # 仅仅提取问题fly的位置
        track_sub = track_framex_all[track_framex_all['id'].isin(problem_ids)]
        if not track_sub.empty:
            for jjj in range(track_sub.shape[0]):
                # 考虑，之后再标记上id信息！
                add_rectangle(img=raw_frame_label_problems, color=colors_dict[track_sub.iloc[jjj, 6]],
                              x=track_sub.iloc[jjj, 2], y=track_sub.iloc[jjj, 3],
                              w=track_sub.iloc[jjj, 4], h=track_sub.iloc[jjj, 5],
                              video_width=video_width, video_height=video_height)
        # 合并四个frame为两行，并且缩放为原视频大小
        img_merged_1 = np.concatenate((raw_frame, raw_frame_label_all), axis=1)
        img_merged_2 = np.concatenate((raw_frame_label_problems, annotated_frame), axis=1)
        img_merged = np.concatenate((img_merged_1, img_merged_2), axis=0)
        img_out = cv2.resize(img_merged, (video_width, video_height))
        video_out.write(img_out)
    video_out.release()
    cv2.destroyAllWindows()


def plot_gaps(trajectoryDF, gapsDF, raw_cap, anno_cap, save_dir, plot_img=True, video_img=True, plot_area=False,
              gaps_joined=False):
    gapsDF = gapsDF.reset_index(drop=True)
    plot_dir = f"{save_dir}/gaps-unsolved"
    print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.mkdir(plot_dir)
    print('Total found ' + str(gapsDF.shape[0]) + ' gaps, plotting...')
    for i in tqdm(gapsDF.index):
        gap_start = gapsDF['min'].iloc[i]
        gap_end = gapsDF['max'].iloc[i]
        if gaps_joined:
            ids = gapsDF['ids'].iloc[i].split(',')
            all_ids = [int(i) for i in ids]
        else:
            id = gapsDF['id'].iloc[i]
            if gapsDF['gap_before_all_neighbours'].iloc[i] != '':
                neighbours_before = [int(i) for i in gapsDF['gap_before_all_neighbours'].iloc[i].split(',')]
            else:
                neighbours_before = []
            if gapsDF['gap_after_all_neighbours'].iloc[i] != '':
                neighbours_after = [int(i) for i in gapsDF['gap_after_all_neighbours'].iloc[i].split(',')]
            else:
                neighbours_after = []
            neighbours = list(set(neighbours_before + neighbours_after))
            all_ids = neighbours + [id.tolist()]
        trackDF_sub = trajectoryDF[trajectoryDF['id'].isin(all_ids)]
        if plot_img:
            img = extract_and_merge_objects(cap=raw_cap,
                                            frame_start=gap_start - 40,
                                            frame_end=gap_end + 40,
                                            trajectoryDF=trackDF_sub,
                                            image_unit_height=100,
                                            image_unit_width=100)
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/gap-{gap_start}-to-{gap_end}.png")
        if video_img:
            video_error_region(raw_cap=raw_cap, anno_cap=anno_cap, trajectoryDF=trajectoryDF, problem_ids=all_ids,
                               frame_start=gap_start - 40, frame_end=gap_end + 40,
                               file_name=f'{plot_dir}/gap-{gap_start}-to-{gap_end}.mp4')
        if plot_area:
            start = gap_start - 40
            end = gap_end + 40
            trackDF_sub = trackDF_sub[(trackDF_sub['frame'] >= start) & (trackDF_sub['frame'] <= end)]
            mat = trackDF_sub.pivot_table(values='BoxArea', index='id', columns='frame', fill_value=0)
            plt.figure(figsize=(18, 0.8))
            sns.heatmap(mat, cbar=True, cmap="YlGnBu", linewidths=0.5, xticklabels=1,
                        yticklabels=1)  # 这一步骤每次好像闪出一个界面，导致绘图慢？
            plt.savefig(f"{plot_dir}/gap-{gap_start}-to-{gap_end}-heatmap.png", bbox_inches='tight')
            plt.close()


def subset_trajectory_by_gap(trajectoryDF, gapsDF, forward_extend=50, backward_extend=50):
    gapsDF = gapsDF.reset_index(drop=True)
    for i in tqdm(gapsDF.index):
        id = gapsDF['id'].iloc[i]
        gap_start = gapsDF['min'].iloc[i]
        gap_end = gapsDF['max'].iloc[i]
        if gapsDF['gap_before_all_neighbours'].iloc[i] != '':
            neighbours_before = [int(i) for i in gapsDF['gap_before_all_neighbours'].iloc[i].split(',')]
        else:
            neighbours_before = []
        if gapsDF['gap_after_all_neighbours'].iloc[i] != '':
            neighbours_after = [int(i) for i in gapsDF['gap_after_all_neighbours'].iloc[i].split(',')]
        else:
            neighbours_after = []
        neighbours = list(set(neighbours_before + neighbours_after))
        all_ids = neighbours + [id.tolist()]
        trackDF_sub = trajectoryDF[trajectoryDF['id'].isin(all_ids)]
        start = gap_start - forward_extend
        end = gap_end + backward_extend
        trackDF_sub = trackDF_sub[(trackDF_sub['frame'] >= start) & (trackDF_sub['frame'] <= end)]
        if i == 0:
            res = trackDF_sub
        else:
            res = pd.concat([res, trackDF_sub], ignore_index=True)
    res = res.drop_duplicates()
    res = res.reset_index(drop=True)
    return res


def extract_object_images(trajectoryDF, cap, save_dir):
    """
    对于位于trajectoryDF中的frame，分别提取所有objects并保存。
    :param trajectoryDF:
    :param cap:
    :param save_dir:
    :return:
    """
    plot_dir = f"{save_dir}/gaps-object-images"
    print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.mkdir(plot_dir)
    print('Total found ' + str(trajectoryDF.shape[0]) + ' objects, extract images...')
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    for j in tqdm(trajectoryDF.index):
        id = trajectoryDF['id'].iloc[j]
        frame = trajectoryDF['frame'].iloc[j]
        x = trajectoryDF['x'].iloc[j]
        y = trajectoryDF['y'].iloc[j]
        w = trajectoryDF['w'].iloc[j]
        h = trajectoryDF['h'].iloc[j]
        frame_image = extract_frame(cap=cap, current_frame=frame)
        image_cropped = crop_image_by_xywhn(image=frame_image, x=x, y=y, w=w, h=h,
                                            width=video_width, height=video_height)
        cv2.imencode('.jpg', image_cropped)[1].tofile(f"{plot_dir}/frame-{frame}-id-{id}.png")


def plt_plot_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.pause(0.0001)  # for for loop not show


def predict_category(trajectoryDF, cap, model_path):
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    model = YOLO(model_path)
    view = list()
    probs = list()
    for j in tqdm(trajectoryDF.index):
        frame = trajectoryDF['frame'].iloc[j]
        x = trajectoryDF['x'].iloc[j]
        y = trajectoryDF['y'].iloc[j]
        w = trajectoryDF['w'].iloc[j]
        h = trajectoryDF['h'].iloc[j]
        frame_image = extract_frame(cap=cap, current_frame=frame)
        image_cropped = crop_image_by_xywhn(image=frame_image, x=x, y=y, w=w, h=h,
                                            width=video_width, height=video_height)
        results = model(image_cropped, verbose=False)
        # plt_plot_image(image_cropped)
        max_index = np.argmax(results[0].probs.data)
        view.append(results[0].names[max_index.tolist()])
        probs.append(round(max(results[0].probs.data).tolist(), 3))
    trajectoryDF = trajectoryDF.assign(view=view)
    trajectoryDF = trajectoryDF.assign(conf=probs)
    return trajectoryDF


def generate_image(img_width, img_height, fill_color, transparency):
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img[:, :] = generate_color_code(color=fill_color, transparency=transparency)
    return img


def generate_color_code(color, transparency):
    rev_trans = 1 - transparency
    if color == "B":
        return (255, 255 * rev_trans, 255 * rev_trans)
    elif color == "G":
        return (255 * rev_trans, 255, 255 * rev_trans)
    elif color == "R":
        return (255 * rev_trans, 255 * rev_trans, 255)
    elif color == "GREY":
        return (255 * rev_trans, 255 * rev_trans, 255 * rev_trans)
    else:
        raise Exception('color must be chosen from B, G, R, GREY, and transparency value should range from 0 to 1.')


def plot_category(trajectoryDF, frame_start, frame_end, img_width, img_height):
    trajectoryDF_sub = trajectoryDF[(trajectoryDF['frame'] >= frame_start) &
                                    (trajectoryDF['frame'] <= frame_end)]
    all_ids = trajectoryDF_sub.id.unique().tolist()
    img_white = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img_white[:, :] = (255, 255, 255)  # 白色空白图，用于生成row_index， column_index， and for lost ids
    # 先生成第一列的第一列图，包括Frame和ID
    for id in all_ids:
        img_index = img_white.copy()
        cv2.putText(img_index, str(id), (img_width - 50, int(img_height / 2 + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if id == all_ids[0]:
            img_final = img_index
        else:
            img_final = np.concatenate((img_final, img_index), axis=0)
    for frame in list(range(frame_start, frame_end + 1)):
        trajectory_df = trajectoryDF_sub[trajectoryDF_sub.frame == frame]
        frame_ids = trajectory_df.id.unique().tolist()
        color_dict = {'cross': 'GREY', 'dorsal': 'B', 'ventral': 'G', 'lateral': 'R'}
        # 生成下一列的下一张图
        for id in all_ids:
            if id in frame_ids:
                view = trajectory_df['view'][trajectory_df['id'] == id].tolist()[0]
                fill_color = color_dict[view]
                img_single = generate_image(img_width=img_width, img_height=img_height, fill_color=fill_color,
                                            transparency=trajectory_df['conf'][trajectory_df['id'] == id].tolist()[0])
            else:
                img_single = img_white.copy()
            if id == all_ids[0]:
                img_next = img_single
            else:
                img_next = np.concatenate((img_next, img_single), axis=0)
        img_final = np.concatenate((img_final, img_next), axis=1)
    return img_final


def merge_adjacent_gaps(gapsDF, distance_cutoff=120):
    """
    如果两个gap之间的距离少于cutoff，并且有overlapped objects[果蝇ids]，就考虑将gap合并到一起
    :return:
    """
    gapsDF = gapsDF.sort_values(by='min')
    gapsDF = gapsDF.reset_index(drop=True)
    # 所有行两两组合，判定是否符合合并的标准
    all_index_combinations = list()
    for i in range(gapsDF.shape[0] - 1):
        for j in range(i + 1, gapsDF.shape[0]):
            first_max = gapsDF['max'].iloc[i]
            first_objects = [gapsDF['id'].iloc[i], gapsDF['gap_before_nearest_neighbour'].iloc[i], gapsDF['gap_after_nearest_neighbour'].iloc[i]]
            first_objects = list(set(first_objects))
            second_min = gapsDF['min'].iloc[j]
            second_objects = [gapsDF['id'].iloc[j], gapsDF['gap_before_nearest_neighbour'].iloc[j], gapsDF['gap_after_nearest_neighbour'].iloc[j]]
            second_objects = list(set(second_objects))
            if (abs(second_min - first_max) < distance_cutoff) and (len(set(first_objects) & set(second_objects)) != 0):
                all_index_combinations.append([i, j])
    # 对两两组合的结果进行进一步合并
    for i in range(len(all_index_combinations) - 1):
        for j in range(i + 1, len(all_index_combinations)):
            set_i = set(all_index_combinations[i])
            set_j = set(all_index_combinations[j])
            if len(set_i & set_j) != 0:
                all_index_combinations[i] = list(set_i | set_j)
                all_index_combinations[j] = list(set_i | set_j)
    # 去除重复的
    for i in reversed(range(len(all_index_combinations))):
        for j in range(i):
            if all_index_combinations[j] == all_index_combinations[i]:
                del all_index_combinations[i]
                break
    # 单独的，未能进入分组的
    all_index_combinations_single = list()
    for i in range(len(all_index_combinations)):
        for j in all_index_combinations[i]:
            all_index_combinations_single.append(j)
    all_index_singles = sorted(list(set(gapsDF.index) - set(all_index_combinations_single)))
    # 先提取单独的
    for i in all_index_singles:
        min = gapsDF['min'].iloc[i]
        max = gapsDF['max'].iloc[i]
        ids = sorted(list({gapsDF['id'].iloc[i], gapsDF['gap_before_nearest_neighbour'].iloc[i],
                           gapsDF['gap_after_nearest_neighbour'].iloc[i]}))
        ids = [str(i) for i in ids]
        id_counts = len(ids)
        ids = ','.join(ids)
        all_gaps = {gapsDF['id'].iloc[i]: [[min, max]]}
        if i == all_index_singles[0]:
            gapsDF_Joined = pd.DataFrame({'min': min, 'max': max, 'ids': ids, 'id_counts': id_counts, 'gap_counts': 1, 'gaps': [all_gaps]}, index=[0])
        else:
            gapsDF_Joined = pd.concat([gapsDF_Joined, pd.DataFrame({'min': min, 'max': max, 'ids': ids, 'id_counts': id_counts, 'gap_counts': 1,
                                                                    'gaps': [all_gaps]}, index=[0])], ignore_index=True)
    # 在提取合并的
    for i in all_index_combinations:
        gapsDF_sub = gapsDF.iloc[i]
        min = gapsDF_sub['min'].min()
        max = gapsDF_sub['max'].max()
        ids = sorted(list(set(gapsDF_sub['id'].tolist()) | set(gapsDF_sub['gap_before_nearest_neighbour'].tolist()) | set(gapsDF_sub['gap_after_nearest_neighbour'].tolist())))
        ids = [str(i) for i in ids]
        id_counts = len(ids)
        ids = ','.join(ids)
        for j in i:
            if j == i[0]:
                all_gaps = {gapsDF['id'].iloc[j]: [[gapsDF['min'].iloc[j], gapsDF['max'].iloc[j]]]}
            else:
                if gapsDF['id'].iloc[j] in all_gaps.keys():
                    all_gaps[gapsDF['id'].iloc[j]].append([gapsDF['min'].iloc[j], gapsDF['max'].iloc[j]])
                else:
                    all_gaps[gapsDF['id'].iloc[j]] = [[gapsDF['min'].iloc[j], gapsDF['max'].iloc[j]]]
        gapsDF_Joined = pd.concat([gapsDF_Joined, pd.DataFrame({'min': min, 'max': max, 'ids': ids, 'id_counts': id_counts, 'gap_counts': len(i),
                                                                'gaps': [all_gaps]}, index=[0])], ignore_index=True)

    gapsDF_Joined = gapsDF_Joined.assign(range=gapsDF_Joined['max'] - gapsDF_Joined['min'] + 1)
    gapsDF_Joined = gapsDF_Joined.sort_values(by='min')
    gapsDF_Joined = gapsDF_Joined.reset_index(drop=True)
    return gapsDF_Joined


# 下一步计划
# 1. plot gaps，gaps两边至少20 frames，并且末端连续的6 frames不能有cross，如果是，则延伸！
# 2. summary gaps，实际考虑的IDs，对于大于2个ID的gap，去除view无改变的ID，如果剩余1个，则标记为无switch，如果剩余大于2个，标记为unsolved，
# 如果剩余2个ID，同只有2个gap的ID一样，如果gap前view不同，且gap前后view均发生改变的，标记为switch，其他情况标记为无switch; 如果gap前view
# 相同，则标记为unsolved。
# 补充，关于view的定义，参考ppt
def extend_cor_region(trajectoryDF, frame_from, frame_to, check_length, cap, model_path):
    # check left
    check_left_ok = check_end_ok(trajectoryDF=trajectoryDF, direction='forward', check_length=check_length)
    while not check_left_ok:
        if (frame_from - 1) not in trajectoryDF['frame'].tolist():
            print("can not extend the forward any more, the frame exceed the max length...")
            break
        else:
            frame_from = frame_from - 1
            indexes_from = trajectoryDF.index[trajectoryDF['frame'] == frame_from].tolist()
            for j in indexes_from:
                trajectoryDF = add_prediction_by_index(trajectoryDF=trajectoryDF, cap=cap, model_path=model_path,
                                                       row_index=j)
            check_left_ok = check_end_ok(trajectoryDF=trajectoryDF, direction='forward', check_length=check_length)
    # check right
    check_right_ok = check_end_ok(trajectoryDF=trajectoryDF, direction='backward', check_length=check_length)
    while not check_right_ok:
        if (frame_to + 1) not in trajectoryDF['frame'].tolist():
            print("can not extend the backward any more, the frame exceed the max length...")
            break
        else:
            frame_to = frame_to + 1
            indexes_to = trajectoryDF.index[trajectoryDF['frame'] == frame_to].tolist()
            for j in indexes_to:
                trajectoryDF = add_prediction_by_index(trajectoryDF=trajectoryDF, cap=cap, model_path=model_path,
                                                       row_index=j)
            check_right_ok = check_end_ok(trajectoryDF=trajectoryDF, direction='backward', check_length=check_length)
    return trajectoryDF, frame_from, frame_to


def check_end_ok(trajectoryDF, check_length, direction=['forward', 'backward'][0]):
    check_res = list()
    trajectoryDF_grouped = trajectoryDF.groupby('id')
    for id, anno in trajectoryDF_grouped:
        if direction == 'forward':
            check_res = check_res + anno['view'][anno['view'] != '-'].tolist()[0:check_length]
        elif direction == 'backward':
            check_res = check_res + anno['view'][anno['view'] != '-'].tolist()[-check_length:]
        else:
            raise Exception("direction must be chosen from forward and backward.")
    if 'cross' in check_res:
        return False
    else:
        return True


def add_prediction_by_index(trajectoryDF, cap, model_path, row_index):
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    model = YOLO(model_path)
    frame = trajectoryDF['frame'].iloc[row_index]
    x = trajectoryDF['x'].iloc[row_index]
    y = trajectoryDF['y'].iloc[row_index]
    w = trajectoryDF['w'].iloc[row_index]
    h = trajectoryDF['h'].iloc[row_index]
    frame_image = extract_frame(cap=cap, current_frame=frame)
    image_cropped = crop_image_by_xywhn(image=frame_image, x=x, y=y, w=w, h=h,
                                        width=video_width, height=video_height)
    results = model(image_cropped, verbose=False)
    max_index = np.argmax(results[0].probs.data)
    trajectoryDF.loc[row_index, 'view'] = results[0].names[max_index.tolist()]
    trajectoryDF.loc[row_index, 'conf'] = round(results[0].probs.data.max().tolist(), 3)
    return trajectoryDF


def cal_view_and_score(ids, trajectoryDF, gap_start, gap_end, view_length=6):
    forward_view = dict()
    forward_score = dict()
    backward_view = dict()
    backward_score = dict()
    for id in ids:
        forward_view[id], forward_score[id] = get_view_and_score(trajectoryDF=trajectoryDF, object_id=id,
                                                                 frame_current=gap_start - 1, direction='forward', view_length=view_length)
        backward_view[id], backward_score[id] = get_view_and_score(trajectoryDF=trajectoryDF, object_id=id,
                                                                   frame_current=gap_end + 1, direction='backward', view_length=view_length)
    return [forward_view], [forward_score], [backward_view], [backward_score]


def get_view_and_score(trajectoryDF, object_id, frame_current, direction=['forward', 'backward'][0], view_length=6):
    views = get_non_cross_view(trajectoryDF=trajectoryDF, object_id=object_id, frame_current=frame_current,
                               direction=direction, view_length=view_length)
    views.reverse()
    view, score = score_list(views)
    return view, score


def get_non_cross_view(trajectoryDF, object_id, frame_current, direction, view_length=6):
    """
    向前或向后获取6个非cross view。
    :param trajectoryDF:
    :param object_id:
    :param frame_current:
    :param direction:
    :return:
    """
    non_cross_view = list()
    while len(non_cross_view) < view_length:
        trajectory_target = trajectoryDF[trajectoryDF['id'] == object_id]
        trajectory_target_frame = trajectory_target[trajectory_target['frame'] == frame_current]
        if trajectory_target_frame.empty:
            if direction == "forward":
                frame_current = frame_current - 1
            elif direction == "backward":
                frame_current = frame_current + 1
            else:
                raise Exception('direction muse chosen from backward and forward.')
            continue
        else:
            view = trajectory_target_frame['view'].tolist()[0]
            if direction == "forward":
                frame_current = frame_current - 1
            elif direction == "backward":
                frame_current = frame_current + 1
            else:
                raise Exception('direction muse chosen from backward and forward.')
            if view == 'cross':
                continue
            else:
                non_cross_view.append(view)
    return non_cross_view


def score_list(Alist):
    """
    对一个list各个元素，基于位置进行打分
    第一个位置1分，第二个位置2分，以此类推，返回分值最大的那个元素，及其分数。
    :param Alist:
    :return:
    """
    keys = list(set(Alist))
    res = dict()
    for i in keys:
        res[i] = sum([j + 1 for j, n in enumerate(Alist) if n == i])
    return max(res, key=res.get), res[max(res, key=res.get)]


def remove_ids_without_gap_and_view_change(trajectoryDF, ids, frame_from, frame_to):
    ids_res = list()
    for id in ids:
        trajectory_sub = trajectoryDF[trajectoryDF['id'].isin([id])]
        trajectory_sub = trajectory_sub[trajectory_sub['frame'] >= frame_from]
        trajectory_sub = trajectory_sub[trajectory_sub['frame'] <= frame_to]
        # 有无gap
        has_gap = not is_continuous(trajectory_sub['frame'].tolist())
        if has_gap:  # has gap
            ids_res.append(id)
        else:  # not have gap
            views = set(trajectory_sub['view'].tolist())
            # 没有cross，或者只有一种非cross view，会被去除
            if 'cross' not in views:
                continue
            else:
                views.remove('cross')
                if len(views) == 1:  # 如果只有一种view，也去除此ID，
                    continue
                elif len(views) == 0:  # 或者只有cross，不可能，应该抛出一个错误！
                    raise Exception("only found cross from the defined frame range for this id.")
                else:
                    ids_res.append(id)
    return ids_res


def is_continuous(lst):
    # 将列表转换为集合
    s = set(lst)
    # 如果集合中的元素数量不等于列表长度，说明有重复元素
    if len(s) != len(lst):
        return False
    # 如果集合中的最小值和最大值之间的元素数量不等于列表长度，说明有缺失元素
    elif max(s) - min(s) + 1 != len(lst):
        return False
    # 如果以上两个条件都不满足，则说明列表是连续的自然数
    else:
        return True


def plot_and_summary_predictions(trajectoryDF, gapsDF, cap, model_path, save_dir, max_extend=200, cor_extend=20,
                                 check_length=5, score_view_length=6, plot_img=True):
    """
    通过对gap附近图像的view预测结果，来判定是否发生了ID switch。
    :param trajectoryDF:
    :param gapsDF:
    :param cap:
    :param model_path:
    :param save_dir:
    :param max_extend:
    :param cor_extend:
    :param check_length:
    :param score_view_length:
    :param plot_img:
    :return:
    """
    if plot_img:
        plot_dir = f"{save_dir}/gaps-complex-with-predictions"
        print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.mkdir(plot_dir)
    print('Total found ' + str(gapsDF.shape[0]) + ' gaps, running...')
    gapsDF = gapsDF.assign(ids_considered={}, forward_view={}, forward_score={},
                           backward_view={}, backward_score={}, ID_switched=False,
                           Solved=False)
    for i in tqdm(gapsDF.index):
        gap_from = gapsDF['min'].iloc[i]
        gap_to = gapsDF['max'].iloc[i]
        ids = gapsDF['ids'].iloc[i].split(',')
        ids = [int(i) for i in ids]
        trajectory_sub = trajectoryDF[trajectoryDF['id'].isin(ids)]
        trajectory_sub = trajectory_sub[trajectory_sub['frame'] > (gap_from - max_extend)]
        trajectory_sub = trajectory_sub[trajectory_sub['frame'] < (gap_to + max_extend)]
        trajectory_sub = trajectory_sub.reset_index(drop=True)
        trajectory_sub = trajectory_sub.assign(view='-', conf='-')
        # 1. 先预测核心区 gap - cor_extend to gap + cor_extend
        frame_from = gap_from - cor_extend
        frame_to = gap_to + cor_extend
        index_from = trajectory_sub.index[trajectory_sub['frame'] == frame_from].min()
        index_to = trajectory_sub.index[trajectory_sub['frame'] == frame_to].max()
        for j in range(index_from, index_to + 1):
            trajectory_sub = add_prediction_by_index(trajectoryDF=trajectory_sub, cap=cap, model_path=model_path,
                                                     row_index=j)
        # 2. 延申核心区
        trajectory_sub, frame_from, frame_to = extend_cor_region(trajectoryDF=trajectory_sub, frame_from=frame_from,
                                                                 frame_to=frame_to, check_length=check_length, cap=cap,
                                                                 model_path=model_path)
        # 3. update gapsDF
        if len(ids) > 2:
            # remove ids without gap and no view change
            ids = remove_ids_without_gap_and_view_change(trajectoryDF=trajectory_sub, ids=ids, frame_from=frame_from,
                                                         frame_to=frame_to)
        gapsDF.loc[i, 'ids_considered'] = [[ids]]
        # fill: forward_view fowrard_score backward_view backward_score， 注意检查gap_start 和 gap_end的位置是否准确！
        # 或许可以再加一个统计值，就是每个view（6个非cross view）的位置和，位置是相对于gap的位置，这个位置和越小，可能结果越靠谱。
        gapsDF.loc[i, 'forward_view'], gapsDF.loc[i, 'forward_score'], gapsDF.loc[i, 'backward_view'], gapsDF.loc[
            i, 'backward_score'] = (cal_view_and_score(ids=ids, trajectoryDF=trajectory_sub, gap_start=gap_from,
                                                       gap_end=gap_to, view_length=score_view_length))
        if len(ids) > 2:  # 如果ids的长度仍旧大于2，则定义此gap无法解决。
            # solved 默认为False，直接去画图就好了
            pass
        elif len(ids) == 1:  # 则认为该gap未发生switch，不管有gap的果蝇到底有没有view转变
            gapsDF.loc[i, 'Solved'] = True
        else:
            # 如果起始view相同，则无法解决
            if gapsDF.loc[i, 'forward_view'][0][ids[0]] == gapsDF.loc[i, 'forward_view'][0][ids[1]]:
                pass
            else:
                # view 均发生变化
                first_view_changed = gapsDF.loc[i, 'forward_view'][0][ids[0]] != gapsDF.loc[i, 'backward_view'][0][
                    ids[0]]
                second_view_changed = gapsDF.loc[i, 'forward_view'][0][ids[1]] != gapsDF.loc[i, 'backward_view'][0][
                    ids[1]]
                if first_view_changed and second_view_changed:
                    gapsDF.loc[i, 'ID_switched'] = True
                gapsDF.loc[i, 'Solved'] = True
        # 4. plot 核心区
        if plot_img:
            # 分类
            gap_counts = gapsDF.loc[i, 'gap_counts']
            fly_counts = gapsDF.loc[i, 'id_counts']
            if gapsDF.loc[i, 'Solved']:
                if gapsDF.loc[i, 'ID_switched']:
                    cat_dir = str(gap_counts) + '-gaps-' + str(fly_counts) + '-flys-Switched'
                else:
                    cat_dir = str(gap_counts) + '-gaps-' + str(fly_counts) + '-flys-NoSwitch'
            else:
                cat_dir = str(gap_counts) + '-gaps-' + str(fly_counts) + '-flys-UnSolved'

            plot_dir_final = f'{plot_dir}/{cat_dir}'
            if not os.path.exists(plot_dir_final):
                os.mkdir(plot_dir_final)
            # 绘图
            img1 = extract_and_merge_objects(cap=cap,
                                             frame_start=frame_from,
                                             frame_end=frame_to,
                                             trajectoryDF=trajectory_sub,
                                             image_unit_height=100,
                                             image_unit_width=100)
            img2 = plot_category(trajectoryDF=trajectory_sub,
                                 frame_start=frame_from,
                                 frame_end=frame_to,
                                 img_width=100,
                                 img_height=100)
            img = np.concatenate((img1, img2), axis=0)
            # 极低概率出现文件名一样的gaps，两个gap的起始和终止位置一样。
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir_final}/gap-from-{gap_from}-to-{gap_to}-annotated.png")
    return gapsDF


def fill_gaps_complex(trajectoryDF, gapsDF, mode=['auto', 'manual'][0], view_score_cutoff=10):
    fake_detected_boxes_before = trajectoryDF.Inserted.sum()
    gapsDF_solved = gapsDF[gapsDF['Solved']]
    if mode == 'auto':
        gapsDF_solved = check_score(df=gapsDF_solved, cutoff=view_score_cutoff)
    elif mode == 'manual':
        pass
    else:
        raise Exception("mode must be chosen from auto or manual.")
    gapsDF_Unsolved = gapsDF.loc[list(set(gapsDF.index.tolist()) - set(gapsDF_solved.index.tolist()))]
    gapsDF_solved = gapsDF_solved.sort_values(by='min', ascending=False)
    gapsDF_solved = gapsDF_solved.reset_index(drop=True)
    for i in tqdm(gapsDF_solved.index):
        id_switched = gapsDF_solved['ID_switched'].iloc[i]
        # 如果ID switch发生了，先修正ID，只gap之后的ID
        if id_switched:
            ids = gapsDF_solved['ids_considered'].iloc[i][0][0]
            gap_end = gapsDF_solved['max'].iloc[i]
            frames_ids_first = (trajectoryDF['frame'] > gap_end) & (trajectoryDF['id'] == ids[0])
            frames_ids_second = (trajectoryDF['frame'] > gap_end) & (trajectoryDF['id'] == ids[1])
            trajectoryDF.loc[frames_ids_first, 'id'] = ids[1]
            trajectoryDF.loc[frames_ids_second, 'id'] = ids[0]
        # fill gaps
        for id, gap_ranges in gapsDF_solved['gaps'].iloc[i].items():
            for gap_range in gap_ranges:
                trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF, new_id=id, lost_id=id, new_id_frame=int(gap_range[1] + 1),
                                             lost_id_frame=int(gap_range[0] - 1))
    trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    fake_detected_boxes_inserted = trajectoryDF.Inserted.sum() - fake_detected_boxes_before
    print("Congratulations, total " + str(fake_detected_boxes_inserted) + " fake detection boxes are inserted...")
    return trajectoryDF, gapsDF_Unsolved


def check_score(df, cutoff):
    index_qualified = list()
    for i in df.index:
        if (min(df['forward_score'].loc[i][0].values()) >= cutoff) & (min(df['backward_score'].loc[i][0].values()) >= cutoff):
            index_qualified.append(i)
    return df.loc[index_qualified]


def format_trajectory_to_DVT(trajectoryDF, object_number, video_width=1920, video_height=1080, format=['xywhn', 'xywh'][0]):
    # columns to be: position,x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5
    # position correspond to frame,but should start from 0
    total_frames = trajectoryDF['frame'].max()
    for i in range(object_number):
        df = trajectoryDF[trajectoryDF['id'] == i + 1]
        df = df[['frame', 'x', 'y']]
        if format == 'xywhn':
            df['x'] = df['x'] * video_width
            df['y'] = df['y'] * video_height
        df = df.set_index('frame', drop=True)
        df = df.rename(columns={'x': 'x' + str(i), 'y': 'y' + str(i)})
        if i == 0:
            result = df
        else:
            result = pd.concat([result, df], axis=1)
    result = result.sort_values(['frame'])
    if not np.all(result.index == list(range(1, total_frames + 1))):  # 检查所有frame均存在，并且是排序好的。
        raise Exception("格式化出错，检查发现frame不完整。")
    result.reset_index(drop=True, inplace=True)
    return result


def fill_all_gaps(trajectoryDF, plot_img, save_dir, cap, cores=4):
    # 对于规则的trajectory中的gaps，全部使用smooth insert方法进行填充
    plot_dir = os.path.join(save_dir, 'gaps-filled')
    if plot_img:
        print("Plots saving directory already existed!") if os.path.exists(plot_dir) else os.makedirs(plot_dir)
    gapsDF = parallel_group_gaps(trajectoryDF=trajectoryDF, group_by='id', cores=cores)
    print('Total found ' + str(gapsDF.shape[0]) + ' gaps, filling all gaps...')
    for i in tqdm(gapsDF.index):
        id = gapsDF['id'].iloc[i]
        gap_start = gapsDF['min'].iloc[i]
        gap_end = gapsDF['max'].iloc[i]
        gap_length = gap_end - gap_start + 1
        frame_max = trajectoryDF['frame'].max()
        print('frame-' + str(gap_start) + '-to-' + str(gap_end) + '-id-' + str(id) + '-length-' + str(gap_length))
        if plot_img:
            img = extract_and_merge_objects(cap=cap,
                                            frame_start=max([1, gap_start - 40]),
                                            frame_end=min([gap_end + 40, frame_max]),
                                            trajectoryDF=trajectoryDF,
                                            image_unit_height=100,
                                            image_unit_width=100)
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/frame-{gap_start}-to-{gap_end}-id-{id}-length-{gap_length}.png")

        # 如果gap起始于第一帧
        if gap_start == 1:
            print("Warnings: A object is not detected at the first frame!")
            print("The frame in which this object first appear: " + str(gap_end + 1))
            trajectoryDF = assigned_insert(trajectoryDF=trajectoryDF,
                                           id=id,
                                           lost_from=gap_start,
                                           lost_to=gap_end,
                                           by=gap_end + 1)
        # 如果gap结束于最后一帧
        elif gap_end == frame_max:
            trajectoryDF = assigned_insert(trajectoryDF=trajectoryDF,
                                           id=id,
                                           lost_from=gap_start,
                                           lost_to=gap_end,
                                           by=gap_start - 1)
        else:
            trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF,
                                         new_id=id,
                                         lost_id=id,
                                         new_id_frame=gap_end + 1,
                                         lost_id_frame=gap_start - 1)
        if plot_img:
            img = extract_and_merge_objects(cap=cap,
                                            frame_start=max([1, gap_start - 40]),
                                            frame_end=min([gap_end + 40, frame_max]),
                                            trajectoryDF=trajectoryDF,
                                            image_unit_height=100,
                                            image_unit_width=100)
            cv2.imencode('.jpg', img)[1].tofile(f"{plot_dir}/frame-{gap_start}-to-{gap_end}-id-{id}-revised.png")

    # check filling results
    if not np.all(trajectoryDF.id.value_counts() == trajectoryDF.frame.max()):
        raise Exception("not all ids has enough values as the frame length.")
    trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
    trajectoryDF.reset_index(drop=True)
    return trajectoryDF, gapsDF.shape[0]


def filter_and_split_trajectory(trajectoryDF, object_number, short_trajectory_cutoff=10, gap_cutoff_for_split=15, cores=4):
    """
    s1: 如果某个连续的trajectory(内部没有gap)长度短于short_trajectory_cutoff，该trajectory会被删除
    s2: 如果某个new ID中存在gap，gap长度大于split_newID(默认15frames),那么此new ID会被分割为两部分。因为发现某些new ID可能也是多个不同ID的混合体
    s3: 按照id出现的先后顺序，对id进行重命名
    :param trajectoryDF:
    :param short_trajectory_cutoff: 如果某个连续的trajectory(内部没有gap)长度短于short_trajectory_cutoff，该trajectory会被删除
    :param gap_cutoff_for_split: 如果某个new ID中存在gap，gap长度大于split_newID(默认15frames),那么此new ID会被分割为两部分。因为发现某些new ID可能也是多个ID的混合体
    :param cores: cores used by function - parallel_group_trajectory
    :return: dataframe
    """
    # s1
    trajectoryDF, total_short_trajectory_removed, total_frames_removed = remove_short_trajectory(trajectoryDF = trajectoryDF, object_number=object_number,
                                                                                                 trajectory_cut=short_trajectory_cutoff, cores=cores)
    # s2
    grouped_trajectory = parallel_group_trajectory(trajectoryDF=trajectoryDF, group_by='id', cores=cores, cut=gap_cutoff_for_split)
    # 如果启示的object number old id也可以被分割，则依据object_number将old ID分为两组, 小于object_number的和，大于object_number的，因为后面要求old id要大于new id
    grouped_trajectory_A = grouped_trajectory[grouped_trajectory.id <= object_number]
    if grouped_trajectory_A.empty:
        raise Exception("ID must start from 1 to " +  str(object_number) + '!')
    # 起始ID必须为从1到object number
    effective_ids = grouped_trajectory_A.id.unique().tolist()
    effective_ids.sort()
    if not effective_ids == list(range(1, object_number + 1)):
        print("Initial ids:" + str(effective_ids))
        raise Exception("Initial IDs are not as expected as from 1 to " + object_number + ".")
    grouped_trajectory_B = grouped_trajectory[grouped_trajectory.id > object_number]
    if grouped_trajectory_B.empty:
        print("No new IDs found!")
        return trajectoryDF, 0
    else:
        newIDs_counts = len(set(grouped_trajectory_B.id.to_list()))
        grouped_trajectory_A = grouped_trajectory_A.sort_values(by=['min', 'id'], ignore_index=True)
        grouped_trajectory_A = grouped_trajectory_A[grouped_trajectory_A.index > (object_number - 1)]  # 去除起始的那些
        grouped_trajectory_B = grouped_trajectory_B.sort_values(by=['min', 'id'], ignore_index=True)
        grouped_trajectory = pd.concat([grouped_trajectory_A, grouped_trajectory_B], ignore_index=True)
        new_id_start = grouped_trajectory.id.max()
        grouped_trajectory['new_id'] = list(range(new_id_start + 1, new_id_start + grouped_trajectory.shape[0] + 1))
        for i in grouped_trajectory.index:
            if grouped_trajectory.id.iloc[i] != grouped_trajectory.new_id.iloc[i]:
                frame_from = grouped_trajectory['min'][i]
                frame_to = grouped_trajectory['max'][i]
                # 注意，新ID一定要大于旧ID，否则可能会修改出错
                if grouped_trajectory.id.iloc[i] > grouped_trajectory.new_id.iloc[i]:
                    print(grouped_trajectory)
                    raise Exception('New ID must bigger than the old ID.')
                trajectoryDF.loc[(trajectoryDF['frame'] >= frame_from) & (trajectoryDF['frame'] <= frame_to) &
                                 (trajectoryDF['id'] == grouped_trajectory.id.iloc[i]), 'id'] = grouped_trajectory.new_id.iloc[i]
        trajectoryDF = trajectoryDF.reset_index(drop=True)
        return trajectoryDF, total_short_trajectory_removed, total_frames_removed, newIDs_counts


def remove_outlier_boxes(trajectoryDF, video_width, video_height, center_x, center_y, circle_r):
    trajectoryDF['raw_x'] = trajectoryDF['x'] * video_width
    trajectoryDF['raw_y'] = trajectoryDF['y'] * video_height
    trajectoryDF['dis_to_center'] = pow(((trajectoryDF['raw_x'] - center_x) ** 2 + (trajectoryDF['raw_y'] - center_y) ** 2), 0.5)
    trajectoryDF = trajectoryDF[trajectoryDF.dis_to_center < circle_r]
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    return trajectoryDF


def zoom_box(ATensorBox, zoom_factor=1.5):
    ATensorBox[:, 2] = ATensorBox[:, 2] * zoom_factor
    ATensorBox[:, 3] = ATensorBox[:, 3] * zoom_factor
    ATensorBox[:, 2][~(ATensorBox[:, 0] * 2 > ATensorBox[:, 2])] = ATensorBox[:, 0][
        ~(ATensorBox[:, 0] * 2 > ATensorBox[:, 2])]
    ATensorBox[:, 3][~(ATensorBox[:, 1] * 2 > ATensorBox[:, 3])] = ATensorBox[:, 1][
        ~(ATensorBox[:, 1] * 2 > ATensorBox[:, 3])]
    return ATensorBox


def crop_images_by_boxes(boxes, image):
    imgs = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = xywh2xyxy(x=boxes[i, 0], y=boxes[i, 1], w=boxes[i, 2], h=boxes[i, 3])
        imgs.append(crop_image(image=image, x1=x1, y1=y1, x2=x2, y2=y2))
    return imgs


def save_predictions(YoloPredictions, frame, IDs, filename):
    predictions = []
    probs = []
    for i in range(len(YoloPredictions)):
        probs.append(round(float(YoloPredictions[i].probs.data.max()), 3))
        # Use Tensor.cpu() to copy the tensor to host memory first.
        max_index = np.argmax(YoloPredictions[i].probs.data.cpu())
        predictions.append(YoloPredictions[i].names[max_index.tolist()])
    res = {'frame': frame, 'ids': IDs, 'predictions': predictions, 'probs': probs}
    df = pd.DataFrame(res)
    if frame == 1:
        df.to_csv(filename, index=False, header=True, sep='\t')
    else:
        df.to_csv(filename, index=False, mode='a', header=False, sep='\t')


def iter_join_trajectory_by_spaced_frame(trajectoryDF, trajectoryGroup, iterations_and_ORA_cutoff = {5: 0.8, 10: 0.5, 30: 0.2, 50: 0.1, 100: 0.0001}):
    # {5: 0.8, 10: 0.5, 30: 0.2, 50: 0.1, 100: 0.0001} 解读，按照里面的顺序设置connect_trajectory_by_spaced_frame中的参数iterations为5ORA_cutoff为0.8，以此类推。
    initial_trajectories = trajectoryGroup.shape[0]
    for iterations, ORA_cutoff in iterations_and_ORA_cutoff.items():
        print('Use Iteration ' + str(iterations) + " and ORA cutoff " + str(ORA_cutoff) + ':')
        initial_trajectories_2 = trajectoryGroup.shape[0]
        for frames_spaced in range(iterations):
            join_trajectory_df = connect_trajectory_by_spaced_frame(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, frame_spaced=frames_spaced)
            trajectoryGroup = join_trajectory_by_ORA(trajectoryJoinINFO=join_trajectory_df, trajectoryGroup=trajectoryGroup, ORA_cutoff=ORA_cutoff)
        print("Total " + str(initial_trajectories_2 - trajectoryGroup.shape[0]) + " trajectories removed!")
    print("Finally, total " + str( initial_trajectories - trajectoryGroup.shape[0]) + "  trajectories removed!")
    return trajectoryGroup

def join_trajectory(trajectoryDF, trajectoryGroup, object_number, iterations_and_ORA_cutoff={5: 0.8, 10: 0.5, 30: 0.2, 50: 0.1, 100: 0.0001}):
    trajectoryGroup = add_xywh_to_trajectoryDF(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup)
    # two steps to join trajectory
    # s1: 按照ORA以及间隔的frames数目，分优先级连接轨迹
    trajectoryGroup = iter_join_trajectory_by_spaced_frame(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, iterations_and_ORA_cutoff=iterations_and_ORA_cutoff)
    # s2: 连接相隔距离最短的轨迹
    trajectoryGroup = join_trajectory_by_distance(trajectoryDF=trajectoryDF, trajectoryGroup=trajectoryGroup, object_number=object_number)
    return trajectoryGroup

def connect_by_trajectory_distance(trajectoryDF, trajectoryGroup, frames_spaced_cutoff=1500):
    # 在join_trajectory之后，仍存在少量碎片轨迹，这时将frames间隔最少的轨迹串联到一起，不在考虑重叠面积，但会输出间隔的frame数目和中心点距离
    # 构建一个from_to_matrix，每行表示，从行名到列名的最短距离，如果某个trajectory找到多个最近上游，则使用xy计算距离最近的那个
    # 行名是非终止的ID，列名是非起始的ID
    trajectoryGroup = trajectoryGroup.sort_values(['range'], ascending=False)
    frame_max = trajectoryDF['frame'].max()
    to_IDs= trajectoryGroup['id'][trajectoryGroup['min'] != 1].to_list()
    from_IDs = trajectoryGroup['id'][trajectoryGroup['max'] != frame_max].to_list()
    spaced_frames_mat = pd.DataFrame(columns=to_IDs, index=from_IDs)
    for i in from_IDs:
        for j in to_IDs:
            if i == j:
                spaced_frames_mat.loc[i,j] = frame_max
            else:
                spaced_frames_mat.loc[i,j] = abs(trajectoryGroup.loc[trajectoryGroup['id'] == i, 'max'].to_list()[0] - trajectoryGroup.loc[trajectoryGroup['id'] == j, 'min'].to_list()[0])
    From_IDs = []
    To_IDs = []
    frames_spaced = []
    for a_from_ID in spaced_frames_mat.index:
        raw_dict = spaced_frames_mat.loc[a_from_ID].to_dict()
        sorted_dict = dict(sorted(raw_dict.items(), key=lambda item: item[1]))
        # 另外，先到先得，如果a和b都是与A最近，若a先出现，则设定a与A相连，b只能与第二最近的相连
        for a_to_ID_candidate in list(sorted_dict.keys()):
            if a_to_ID_candidate not in To_IDs:
                From_IDs.append(a_from_ID)
                To_IDs.append(a_to_ID_candidate)
                frames_spaced.append(sorted_dict[a_to_ID_candidate])
                break
    paired_trajectory_df = pd.DataFrame()
    paired_trajectory_df['From_ID'] = From_IDs
    paired_trajectory_df['To_ID'] = To_IDs
    paired_trajectory_df['Frames_spaced'] = frames_spaced #代码未经过测试
    paired_trajectory_df['ORA_of_selected_ID'] = 1 # 设置最大重叠面积为1，伪数据，方便后续做join_trajectory
    paired_trajectory_df['ID_with_max_ORA'] = paired_trajectory_df['To_ID'] #伪数据，方便后续做join_trajectory
    if any(paired_trajectory_df['Frames_spaced'] > frames_spaced_cutoff):
        print("Attention: some joined trajectory has spaced frames more than " + str(frames_spaced_cutoff), ", and these will be removed")
        print(paired_trajectory_df[paired_trajectory_df['Frames_spaced'] >= frames_spaced_cutoff])
    paired_trajectory_df = paired_trajectory_df[paired_trajectory_df['Frames_spaced'] < frames_spaced_cutoff]
    return paired_trajectory_df

def join_trajectory_by_distance(trajectoryDF, trajectoryGroup, object_number):
    join_trajectory_df = connect_by_trajectory_distance(trajectoryGroup=trajectoryGroup, trajectoryDF=trajectoryDF)
    trajectoryGroup = join_trajectory_by_ORA(trajectoryJoinINFO=join_trajectory_df, trajectoryGroup=trajectoryGroup, ORA_cutoff=0)
    trajecotryGroup_full = trajectoryGroup[(trajectoryGroup['min'] == 1) & (trajectoryGroup['max'] == trajectoryDF.frame.max())]
    trajecotryGroup_left = trajectoryGroup[~((trajectoryGroup['min'] == 1) & (trajectoryGroup['max'] == trajectoryDF.frame.max()))]
    if trajecotryGroup_full.shape[0] != object_number:
        print('Please check the connected trajectory bellow:')
        print(join_trajectory_df)
        raise Exception('Error: join trajectory by distance failed. the final trajectory counts not equals to object number!')
    if not trajecotryGroup_left.empty:
        print('Please check the unconnected trajectory bellow:')
        print(trajecotryGroup_left)
    print("Congratulations, all trajectories were solved!")
    return trajectoryGroup

def extract_coordinate(trajectoryDF, frame, id):
    # 基于frame和id从trajectory提取坐标位置
    row = trajectoryDF[(trajectoryDF['id'] == id) & (trajectoryDF['frame'] == frame)]
    try:
        return {'x': row['x'].tolist()[0], 'y': row['y'].tolist()[0], 'w': row['w'].tolist()[0], 'h': row['h'].tolist()[0]}
    except IndexError as e:
        print("Can not found frame " + str(frame) + " id " + str(id))
        print(f"Caught an IndexError: {e}")

def add_xywh_to_trajectoryDF(trajectoryDF, trajectoryGroup):
    trajectoryGroup['min_xywh'] = trajectoryGroup.apply(lambda _: {}, axis=1)
    trajectoryGroup['max_xywh'] = trajectoryGroup.apply(lambda _: {}, axis=1)
    for index, row in trajectoryGroup.iterrows():
        trajectoryGroup.at[index, 'min_xywh'] = extract_coordinate(trajectoryDF=trajectoryDF, frame=row['min'], id=row['id'])
        trajectoryGroup.at[index, 'max_xywh'] = extract_coordinate(trajectoryDF=trajectoryDF, frame=row['max'], id=row['id'])
    return trajectoryGroup


def join_trajectory_by_ORA(trajectoryJoinINFO, trajectoryGroup, ORA_cutoff=0.2):
    trajectoryJoinINFO = trajectoryJoinINFO[trajectoryJoinINFO['ORA_of_selected_ID'] > ORA_cutoff]
    # trajectoryJoinINFO 按照后端ID的大小进行排序倒序
    trajectoryJoinINFO = trajectoryJoinINFO.sort_values(['ID_with_max_ORA'], ascending=False)
    # print(str(trajectoryJoinINFO.shape[0]) + " trajectories will be removed!")
    # update trajectoryGroup and trajectoryDF
    if 'Included_trajectory_ID' not in trajectoryGroup.columns: #第一次做connect
        trajectoryGroup['Included_trajectory_ID'] = trajectoryGroup['id']
        for index, row in trajectoryJoinINFO.iterrows():
            # 把 trajectoryGroup id 列的id进行改正，
            trajectoryGroup.loc[trajectoryGroup['id'] == row['ID_with_max_ORA'], 'id'] = row['From_ID']
        # 然后把trajectoryGroup内同一ID的进行合并
        trajectoryGroup = trajectoryGroup.groupby('id').apply(lambda x: pd.Series({
            'min': x['min'].min(),
            'max': x['max'].max(),
            'range': x['max'].max() - x['min'].min() + 1,
            'parent_id': x['parent_id'].to_list(),
            'Included_trajectory_ID': x['Included_trajectory_ID'].to_list(),
            'min_xywh': x['min_xywh'][x['min'].idxmin()],
            'max_xywh': x['max_xywh'][x['max'].idxmax()],
        }),include_groups=False)
    else:
        for index, row in trajectoryJoinINFO.iterrows():
            # 把 trajectoryGroup id 列的id进行改正，
            trajectoryGroup.loc[trajectoryGroup['id'] == row['ID_with_max_ORA'], 'id'] = row['From_ID']
        # 然后把trajectoryGroup内同一ID的进行合并
        trajectoryGroup = trajectoryGroup.groupby('id').apply(lambda x: pd.Series({
            'min': x['min'].min(),
            'max': x['max'].max(),
            'range': x['max'].max() - x['min'].min() + 1,
            'parent_id': flattened_list(x['parent_id'].to_list()),
            'Included_trajectory_ID': flattened_list(x['Included_trajectory_ID'].to_list()),
            'min_xywh': x['min_xywh'][x['min'].idxmin()],
            'max_xywh': x['max_xywh'][x['max'].idxmax()],
        }),include_groups=False)
    trajectoryGroup = trajectoryGroup.reset_index(drop=False)
    return trajectoryGroup

def flattened_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def connect_trajectory_by_spaced_frame(trajectoryDF, trajectoryGroup, frame_spaced=0):
    paired_trajectory = {}
    for _, row in trajectoryGroup.iterrows():
        start_frames = [row['max'] + 1 + frame_spaced, row['max'] + 1 - frame_spaced]
        if any(trajectoryGroup['min'].isin(start_frames)):
            paired_trajectory[row['id']] = trajectoryGroup.loc[trajectoryGroup['min'].isin(start_frames), 'id'].to_list()
    paired_trajectory_df = pd.DataFrame()
    paired_trajectory_df['From_ID'] = paired_trajectory.keys()
    paired_trajectory_df['To_ID'] = paired_trajectory.values()
    paired_trajectory_df['From_Frame'] = 0
    paired_trajectory_df['To_Frame'] = paired_trajectory_df.apply(lambda _: {}, axis=1)
    paired_trajectory_df['Spaced_Frames'] = paired_trajectory_df.apply(lambda _: {}, axis=1)
    paired_trajectory_df['overlapped_area_pct'] = paired_trajectory_df.apply(lambda _: {}, axis=1)
    for index, row in paired_trajectory_df.iterrows():
        From_Frame = trajectoryGroup.loc[trajectoryGroup['id'] == row['From_ID'], 'max'].to_list()[0]
        paired_trajectory_df.at[index, 'From_Frame'] = From_Frame
        xywh1 =  trajectoryGroup.loc[trajectoryGroup['id'] == row['From_ID'], 'max_xywh'].to_list()[0]
        overlap_area = {}
        to_frame_dict = {}
        spaced_frame_dict = {}
        for To_ID in row['To_ID']:
            to_frame_dict[To_ID] = trajectoryGroup.loc[trajectoryGroup['id'] == To_ID, 'min'].to_list()[0]
            spaced_frame_dict[To_ID] = to_frame_dict[To_ID] - From_Frame - 1
            xywh2 = trajectoryGroup.loc[trajectoryGroup['id'] == To_ID, 'min_xywh'].to_list()[0]
            overlap_area[To_ID] = calculate_ORA_pair(xywh1, xywh2)
        paired_trajectory_df.at[index, 'To_Frame'] = to_frame_dict
        paired_trajectory_df.at[index, 'Spaced_Frames'] = spaced_frame_dict
        paired_trajectory_df.at[index, 'overlapped_area_pct'] = overlap_area
    paired_trajectory_df['ID_with_max_ORA'] = paired_trajectory_df['overlapped_area_pct'].apply(lambda x: max(x, key=x.get))
    paired_trajectory_df['ORA_of_selected_ID'] = paired_trajectory_df['overlapped_area_pct'].apply(lambda row: max(row.values()))
    return paired_trajectory_df

def calculate_ORA_pair(xywh1, xywh2):
    """
    对某一帧，计算两个box之间的重叠面积除以两个box占的总面积 OverlappedArea/(area_A + area_B - OverlappedArea)
    overlapped_area_ratio 缩写为 ORA
    :return: A float value
    """
    polygon_1 = xywh_to_polygon(x=xywh1['x'], y=xywh1['y'],w=xywh1['w'], h=xywh1['h'])
    polygon_2 = xywh_to_polygon(x=xywh2['x'], y=xywh2['y'],w=xywh2['w'], h=xywh2['h'])
    intersection = polygon_1.intersection(polygon_2)
    overlapped_area_ratio = round(intersection.area / (polygon_1.area + polygon_2.area - intersection.area), 3)
    return overlapped_area_ratio



def update_Trajectory_IDs(trajectoryDF, Solved_trajectoryGroup):
    # 首先仅保留full trajectory
    Solved_trajectoryGroup_full = Solved_trajectoryGroup[(Solved_trajectoryGroup['min'] == 1) & (Solved_trajectoryGroup['max'] == trajectoryDF['frame'].max())]
    # 去除未解决的trajectory
    trajectoryGroup_unSolved = Solved_trajectoryGroup[~((Solved_trajectoryGroup['min'] == 1) & (Solved_trajectoryGroup['max'] == trajectoryDF['frame'].max()))]
    for index, row in trajectoryGroup_unSolved.iterrows():
        trajectoryDF = trajectoryDF[trajectoryDF['id'] != row['id']]

    for index, row in Solved_trajectoryGroup_full.iterrows():
        for A_ID in row['Included_trajectory_ID']:
            trajectoryDF.loc[trajectoryDF['id'] == A_ID, 'id'] = row['id']
    # 去除重复的
    trajectoryDF = trajectoryDF.drop_duplicates(subset=['frame', 'id'])
    return trajectoryDF

def simple_fill_all_gaps(trajectoryDF, gap_length_cutoff=300, cores=4):
    gapDF = parallel_group_gaps(trajectoryDF=trajectoryDF, group_by='id', cores=cores, cut=1)
    print("Total found " + str(gapDF.shape[0]) + " gaps, and please check the top 10 longest gaps bellow:")
    print(gapDF.sort_values(by='range', ascending=False)[:10])
    for _, row in tqdm(gapDF.iterrows(), total=gapDF.shape[0]):
        trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF,  new_id=row['id'], lost_id=row['id'], new_id_frame=row['max'] + 1, lost_id_frame=row['min'] - 1)
    # trajectoryDF = pd.concat([trajectoryDF, trajectory_new], axis=1)
    trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    return trajectoryDF


def cal_velocity_from_position(pos, n_inds, fps, scaling_to_mm=1):
    # 计算移动速度， scaling_to_mm: 1 pixel to ? mm
    # pos： dataframe DVT format coordinates
    l_list = []
    xlabel_list = []
    ylabel_list = []
    for i in range(n_inds):
        xlabel_list.append('x' + str(int(i)))
        ylabel_list.append('y' + str(int(i)))
    for xlabel in xlabel_list:
        xlabel_p = xlabel + '_p'
        pos[xlabel] = pos[xlabel] * scaling_to_mm
        pos[xlabel_p] = pos[xlabel].shift(1)
    for ylabel in ylabel_list:
        ylabel_p = ylabel + '_p'
        pos[ylabel] = pos[ylabel] * scaling_to_mm
        pos[ylabel_p] = pos[ylabel].shift(1)
    # pos = pos.fillna(method = 'bfill')
    # pos = pos.fillna(method = 'ffill')
    pos = pos.bfill()
    pos = pos.ffill()

    def smooth(x, window_len=11, window='hanning'):
        # window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
        s = np.r_[2 * x[0] - x[window_len - 1::-1], x,
                  2 * x[-1] - x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len:-window_len + 1]

    def dist_cost(x1, x2, x3, x4):
        return np.sqrt(np.power(x1 - x2, 2) + np.power(x3 - x4, 2))

    for i in range(n_inds):
        xlabel = 'x' + str(int(i))
        ylabel = 'y' + str(int(i))
        xlabel_p = xlabel + '_p'
        ylabel_p = ylabel + '_p'
        move_label = 'move' + str(int(i))
        pos[move_label] = pos.apply(lambda row: dist_cost(row[xlabel], row[xlabel_p], row[ylabel], row[ylabel_p]),
                                    axis=1)
        pos[move_label] = pos[move_label] * scaling_to_mm
        pos[move_label] = pos[move_label] * fps
        smoothed_velocity_list = smooth(np.array(list(pos[move_label])), window_len=30)
        # frame_list=pos.index/fps/60 # in minute
        # plt.plot(frame_list, pos[move_label], alpha=0.5,color='blue',label='velocity')
        # plt.plot(frame_list, smoothed_velocity_list, alpha=0.5,color='red',label='smoothed velocity')
        # plt.xlabel('Time(min)', fontsize=10)
        # plt.ylabel('Velocity(mm/sec)', fontsize=10)
        # plt.title(xlabel+ylabel+' motion velocity')
        # plt.tight_layout()
        # plt.show()
        velocity_dict = {'frame': list(pos.index + 1),
                         'id': [i + 1] * len(smoothed_velocity_list),
                         'velocity': smoothed_velocity_list}
        velocity_df = pd.DataFrame(velocity_dict)
        if i == 0:
            velocity_res = velocity_df
        else:
            velocity_res = pd.concat([velocity_res, velocity_df], ignore_index=True)
    velocity_res.loc[velocity_res['velocity'] < 0, 'velocity'] = 0
    return velocity_res


def simple_fill_all_gaps(trajectoryDF, gap_length_cutoff=100, cores=4):
    gapDF = parallel_group_gaps(trajectoryDF=trajectoryDF, group_by='id', cores=cores, cut=1)
    print("Total found " + str(gapDF.shape[0]) + " gaps, and please check the top 10 longest gaps bellow:")
    print(gapDF.sort_values(by='range', ascending=False)[:10])
    print("Annotating the gaps...")
    gapDF = annotate_gaps(trajectoryDF=trajectoryDF, gapsDF=gapDF, overlapped_area_cutoff_for_NN=0)
    gapDF = find_best_NN_for_gaps(gapsDF=gapDF)
    print("Filling all gaps...")
    for index, row in tqdm(gapDF.iterrows(), total=gapDF.shape[0]):
        if row['range'] < gap_length_cutoff:
            trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF, new_id=row['id'], lost_id=row['id'], new_id_frame=row['max'] + 1, lost_id_frame=row['min'] - 1)
        elif np.isnan(row['best_NN']):
            print("Long gaps bellow will be filled by smooth insert!")
            print(gapDF.iloc[index])
            trajectoryDF = smooth_insert(trajectoryDF=trajectoryDF, new_id=row['id'], lost_id=row['id'], new_id_frame=row['max'] + 1, lost_id_frame=row['min'] - 1)
        else:
            trajectoryDF = fill_by_neighbour(trajectoryDF=trajectoryDF, frame_from=row['min'], frame_to=row['max'], gap_id=row['id'], NN_id=row['best_NN'], cores=cores)
    trajectoryDF = trajectoryDF.sort_values(by=['frame', 'id'])
    trajectoryDF = trajectoryDF.reset_index(drop=True)
    return trajectoryDF


def fill_by_neighbour(trajectoryDF, frame_from, frame_to, gap_id, NN_id, cores=4):
    # 如果 NN 无gap
    NN_frames = trajectoryDF[(trajectoryDF['id'] == NN_id) & (trajectoryDF['frame'] >= frame_from) & (trajectoryDF['frame'] <= frame_to)]
    NN_frames = NN_frames.assign(id=gap_id)
    if NN_frames.shape[0] != (frame_to - frame_from + 1):
     # 如果 NN 有gap，混合填充
        NN_gapsDF = parallel_group_gaps(trajectoryDF=NN_frames, group_by='id', frame_start=NN_frames['frame'].min(), cores=cores, cut=1)
        for index, row in NN_gapsDF.iterrows():
            NN_frames = smooth_insert(trajectoryDF=NN_frames, new_id=row['id'], lost_id=row['id'], new_id_frame=row['max'] + 1, lost_id_frame=row['min'] - 1)
    trajectoryDF = pd.concat([trajectoryDF, NN_frames], ignore_index=False)
    return trajectoryDF


def annotate_gaps(trajectoryDF, gapsDF, overlapped_area_cutoff_for_NN=0):
    gapsDF['forward_overlapping_area'] = gapsDF.apply(lambda _: {}, axis=1)
    gapsDF['backward_overlapping_area'] = gapsDF.apply(lambda _: {}, axis=1)
    for index, row in tqdm(gapsDF.iterrows(), total=gapsDF.shape[0]):
        # 注意calculate_ORA是计算所有id对之间的比例，后续考虑使用仅仅与目标ID的计算
        gapsDF.at[index, 'forward_overlapping_area'] = (
            calculate_ORA_single(frame_trajectoryDF=trajectoryDF[trajectoryDF['frame'] == row['min'] - 1],
                                 target_id=row['id']))
        gapsDF.at[index, 'backward_overlapping_area'] = (
            calculate_ORA_single(frame_trajectoryDF=trajectoryDF[trajectoryDF['frame'] == row['max'] + 1],
                                 target_id=row['id']))
    gapsDF['forward_NN'] = gapsDF['forward_overlapping_area'].apply(lambda row: list(filter_dict(row, overlapped_area_cutoff_for_NN).keys()))
    gapsDF['forward_NN_counts'] = gapsDF['forward_NN'].apply(lambda row: len(row))
    gapsDF['forward_NN_nearest'] = gapsDF['forward_overlapping_area'].apply(lambda row: max(row, key=row.get))
    gapsDF['forward_ID_counts'] = gapsDF['forward_overlapping_area'].apply(lambda row: len(row) + 1)
    gapsDF['backward_NN'] = gapsDF['backward_overlapping_area'].apply(lambda row: list(filter_dict(row, overlapped_area_cutoff_for_NN).keys()))
    gapsDF['backward_NN_counts'] = gapsDF['backward_NN'].apply(lambda row: len(row))
    gapsDF['backward_NN_nearest'] = gapsDF['backward_overlapping_area'].apply(lambda row: max(row, key=row.get))
    gapsDF['backward_ID_counts'] = gapsDF['backward_overlapping_area'].apply(lambda row: len(row) + 1)
    return gapsDF


def find_best_NN_for_gaps(gapsDF):
    gapsDF['best_NN'] = gapsDF.apply(lambda _: {}, axis=1)
    for index, row in tqdm(gapsDF.iterrows(), total=gapsDF.shape[0]):
        if row['forward_NN_nearest'] == row['backward_NN_nearest']:
            gapsDF.at[index, 'best_NN'] = row['forward_NN_nearest']
        elif (row['forward_NN_nearest'] in row['backward_NN']) & (row['backward_NN_nearest'] not in row['forward_NN']):
            gapsDF.at[index, 'best_NN'] = row['forward_NN_nearest']
        elif (row['forward_NN_nearest'] not in row['backward_NN']) & (row['backward_NN_nearest'] in row['forward_NN']):
            gapsDF.at[index, 'best_NN'] = row['backward_NN_nearest']
        elif (row['forward_NN_nearest'] in row['backward_NN']) & (row['backward_NN_nearest'] in row['forward_NN']):
            # 使用最大ORA
            forward_NN_ORA_sum = row['forward_overlapping_area'][row['forward_NN_nearest']] + row['backward_overlapping_area'][row['forward_NN_nearest']]
            backward_NN_ORA_sum = row['forward_overlapping_area'][row['backward_NN_nearest']] + row['backward_overlapping_area'][row['backward_NN_nearest']]
            if forward_NN_ORA_sum > backward_NN_ORA_sum:
                gapsDF.at[index, 'best_NN'] = row['forward_NN_nearest']
            else:
                gapsDF.at[index, 'best_NN'] = row['forward_NN_nearest']
        else:
            gapsDF.at[index, 'best_NN'] = np.nan
            # raise Exception('Error: can not find the nearest NN, probably this gaps is not caused by object cross!')
    return gapsDF