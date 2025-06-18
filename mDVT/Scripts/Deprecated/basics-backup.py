import cv2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import numpy as np
from joblib import Parallel, delayed

matplotlib.use('Qt5Agg')

def df_range_to_list(df, min_column = 'min_extended', max_column = 'max_extended'):
    res = list()
    for i in range(0, df.shape[0]):
        res += list(range(df[min_column][i], df[max_column][i] + 1))
    return res

def slide_window_all_equal(AList, window_size = 40):
    if len(AList) <=  window_size:
        raise Exception('Input list must has length more than window size!')
    res = list()
    for i in range(len(AList) - window_size + 1):
        sub = AList[i:(i + window_size)]
        res.append(all_list_element_equal(sub))
    return res

def slide_window_all_equal_two(AList, BList, window_size = 40):
    """
    通过slide window形式获取问题区域，注意问题区域的长度要小于window_size才可以，如果问题区域长度超过设置，就需要合并相邻的问题区域！
    此外，默认前40个元素均相等！
    :param AList:
    :param BList:
    :param window_size:
    :return:
    """
    if len(AList) <=  window_size or len(BList) <= window_size:
        raise Exception('Input list must has length more than window size!')
    if len(AList) !=  len(BList):
        raise Exception('Input list muse be equal length!')
    res = [True for i in range(window_size - 1)]
    for i in range(len(AList) - window_size + 1):
        subA = all_list_element_equal(AList[i:(i + window_size)])
        subB = all_list_element_equal(AList[i:(i + window_size)])
        if subA and subB:
            res.append(True)
        else:
            res.append(False)
    return res

def all_list_element_equal(AList):
    if AList.count(AList[0]) == len(AList):
        return True
    else:
        return False

def group_error_frames(frameDF, ids_column = 'ids', ids_counts_column = 'id_counts',
                       window_size = 40, group_cut = 1, extend_frames = (5, 5), object_number = 20):
    """
    用于对frameInfo中，提取ids发生改变的区域（frames），进行分组和延申。注意frameDF里的frame列必须是连续的！
    默认最开始40frames帧ids组合和数量均是正确的。非问题区域的IDs数目应为object_number
    只有当id_counts和ids连续在40frames中均不发生改变，才被认为是正确状态！连续40帧的定义，指的是单个ID最大的Gap。
    :param frameDF:
    :param ids_column:
    :param group_cut:
    :param extend_frames:
    :return: a grouped error regions in pandas DataFrame
    """
    # frame_not_equal = list(frameDF[ids_column].iloc[1:].values != frameDF[ids_column].iloc[:-1].values)
    # frame_not_equal.insert(0, False)
    # # 同上一个frame比较，发生变化的frame
    # frame_changed = frameDF[frame_not_equal]['frame'].tolist()
    com_res = slide_window_all_equal_two(frameDF[ids_counts_column].tolist(), frameDF[ids_column].tolist(), window_size = window_size)
    problem_frames = frameDF['frame'][[not i for i in com_res]].tolist()
    frame_error_group = group_numbers(problem_frames, cut = group_cut)
    # frame_error_group = frame_error_group.assign(min_extended=frame_error_group['min'] - extend_frames[0])
    # frame_error_group = frame_error_group.assign(max_extended=frame_error_group['max'] + extend_frames[1])
    # frame_error_group = frame_error_group.assign(range_extended=frame_error_group['max_extended'] -
    #                                                             frame_error_group['min_extended'] + 1)
    # check all other non-problem frames whether all has specified IDs count.
    frames_in_error_region = df_range_to_list(df = frame_error_group, min_column = 'min_extended', max_column = 'max_extended')
    if np.all(frameDF[~np.in1d(frameDF['frame'].values, frames_in_error_region)]['id_counts'] == object_number):
        pass
    else:
        raise Exception('Sorry, There are frames not in error region still has wrong ID count!')
    return frame_error_group

def count_gaps(frames):
    """
    对于排序好的整数，统计不连续的位置有多少个
    :param frames:
    :return: an integer
    """
    result = [x - y for x, y in zip(frames[1:], frames[:-1])]
    counts = sum(1 for i in result if i != 1)
    return counts

def calculate_distance_matrix(df):
    """
    基于pandas DataFrame中的列x、y坐标值，计算每两个点之间的距离，值越大，两点间的距离也越大
    :param df: a trajectory from a specified frame
    :return: a distance matrix
    """
    distance_matrix = squareform(pdist(df[['x', 'y']]))
    np.fill_diagonal(distance_matrix, 1)  # 对角线的值设为1
    distance_dataframe = pd.DataFrame.from_records(distance_matrix)  # numpy array to pandas dataframe
    distance_dataframe.index = df['id'].tolist()
    distance_dataframe.columns = df['id'].tolist()
    return distance_dataframe

def parallel_summary_frame(df, group_by, cores = 32):
    """
    多线程方式，对DataFrame进行分组，然后分别对每组计算每两个box之间overlap区域的面积比，统计每帧出现的ID数目
    return: an area overlap matrix
    """
    def data_process(name, group):
        # 对每一dataframe进行怎样的操作
        group.index = group['id'].tolist()
        overlapped_area_ratio_matrix = calculate_overlapped_area_ratio(frame_trajectoryDF=group)
        max_ratio = overlapped_area_ratio_matrix.max(axis=None)
        ids = ','.join(map(str, sorted(group['id'])))
        id_counts = group.shape[0],
        return pd.DataFrame({'frame': name, 'ids': ids, 'id_counts': id_counts, 'maxRatio': max_ratio}, index=[name])

    def applyParallel(dfGrouped, func):
        # 多线程方式，执行某个function，并合并结果
        res = Parallel(n_jobs=cores)(delayed(func)(name, group) for name, group in dfGrouped)
        return pd.concat(res)

    df = df.groupby(group_by)
    result = applyParallel(df, data_process)
    # 检查frame列是否是连续的
    if np.all(np.diff(result['frame'].to_numpy()) == 1):
        pass
    else:
        print("Attention! frames are not continuous...")
    return result

def parallel_summary_ids(df, group_by, cores = 4):
    """
    多线程方式，对DataFrame进行分组后，统计每个ID的起始时间、总frame数目、gap数目等信息
    return: Pandas DataFrame
    """
    def data_process(name, group):
        # 对每一dataframe进行怎样的操作
        return pd.DataFrame({'id': name, 'start': min(group['frame']), 'end': max(group['frame']),
                             'total_frames': group.shape[0], 'total_gaps': count_gaps(group['frame'].tolist())},
                            index=[name])

    def applyParallel(dfGrouped, func):
        # 多线程方式，执行某个function，并合并结果
        res = Parallel(n_jobs=cores)(delayed(func)(name, group) for name, group in dfGrouped)
        return pd.concat(res)

    df = df.groupby(group_by)
    result = applyParallel(df, data_process)
    return result

def plot_error_region_heatmap(trajectoryDF, errorDF, save_dir, frame_space = 100):
    """

    :param trajectoryDF:
    :param errorDF:
    :return:
    """
    for i in range(len(errorDF)):
        print("Processing Group:" + str(i) + "...")
        # 对于问题区域绘制热图
        plot_data = crosstab_data(df = trajectoryDF, id_column = "id", frame_column = "frame",
                                 frame_middle = errorDF['middle'].iloc[i], frame_space = frame_space)
        plt.figure(figsize=(36, 4))
        sns.heatmap(plot_data, cbar=False, cmap="YlGnBu", linewidths=0.5, xticklabels=1, yticklabels=1) #这一步骤每次好像闪出一个界面，导致绘图慢？
        plt.savefig(f"{save_dir}\\frame-{errorDF['middle'].iloc[i]}.png", bbox_inches='tight')
        plt.close()

def video_error_region_integrated(raw_cap, anno_cap, trajectoryDF, errorDF, save_dir, frame_space = 100):
    # 存储一些video信息，用于video输出时的参数设置
    fps = raw_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for i in range(len(errorDF)):
        plot_data = crosstab_data(df=trajectoryDF, id_column="id", frame_column="frame",
                                  frame_middle=errorDF['middle'].iloc[i], frame_space=frame_space)
        # 提取出错的果蝇id
        problem_ids = list(plot_data.index[plot_data.apply(lambda x: x.sum(), axis=1) != plot_data.shape[1]])
        # 给所有果蝇设定颜色
        all_ids = plot_data.index.tolist()
        # 提取所有出错位置的frames
        problem_frames = plot_data.columns.tolist()
        # 为每一种ID生成以一种对应颜色
        colors_dict = generate_colors_dict(all_ids)
        # output video info
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(f'{save_dir}\\frame-{errorDF['middle'].iloc[i]}.mp4', fourcc, fps, (video_width, video_height))
        # 逐帧标记果蝇
        for j in problem_frames:
            raw_cap.set(cv2.CAP_PROP_POS_FRAMES, j - 1)
            _, raw_frame = raw_cap.read()
            anno_cap.set(cv2.CAP_PROP_POS_FRAMES, j - 1)
            _, annotated_frame = anno_cap.read()
            # 深复制: 标记所有的flys
            raw_frame_label_all = raw_frame.copy()
            # subset track results by frame
            track_framex_all = trajectoryDF[trajectoryDF.frame == j]
            if not track_framex_all.empty:
                for jj in range(track_framex_all.shape[0]):
                    add_rectangle(img=raw_frame_label_all, color=colors_dict[track_framex_all.iloc[jj, 6]],
                                  x=track_framex_all.iloc[jj, 2], y=track_framex_all.iloc[jj, 3],
                                  w=track_framex_all.iloc[jj, 4], h=track_framex_all.iloc[jj, 5],
                                  video_width=video_width, video_height=video_height)
            # 深复制： 仅仅标记有问题的flys
            raw_frame_label_problems = raw_frame.copy()
            # 仅仅提取问题flys的位置
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

def plot_error_region_objects(cap, trajectoryDF, frame_error_groupDF, save_dir):
    for i in range(frame_error_groupDF.shape[0]):
        img = extract_and_merge_objects(cap=cap, frame_start=frame_error_groupDF.iloc[i]['min_extended'],
                                        frame_end=frame_error_groupDF.iloc[i]['max_extended'],
                                        trajectoryDF=trajectoryDF,
                                        image_unit_height=100,
                                        image_unit_width=100)
        # cv2.imwrite filename not support chinese path
        # cv2.imwrite(filename = f"{save_dir}\\frame-{frame_error_groupDF.iloc[i]['middle']}-images.png", img=img) # Not work
        cv2.imencode('.jpg', img)[1].tofile(f"{save_dir}\\frame-{frame_error_groupDF.iloc[i]['middle']}-images.png")

def remove_extra_IDs_in_trajectory(trajectoryDF, frameDF, object_number = 20):
    print("Check the id counts distribution bellow!")
    print(frameDF['id_counts'].value_counts())
    if any(frameDF['id_counts'] - object_number > 1):
        raise Exception("检测到id组合和id数目同时发生变化！无法继续使用以下代码分析！")
    frame_more = frameDF[frameDF['id_counts'] > object_number]
    track_res_updated = trajectoryDF.copy(deep=True)
    frame_info_updated = frameDF.copy(deep=True)
    for i in range(frame_more.shape[0]):
        current_frame = frame_more['frame'].iloc[i]
        new_id = find_new_ids(trajectoryDF = track_res_updated, current_frame = current_frame)
        if len(new_id) != 1:
            print(f"Frame {current_frame} has {len(new_id)} more IDs than previous frame...")
        for j in new_id:
            track_res_updated = track_res_updated[
                ~((track_res_updated['frame'] == current_frame) & (track_res_updated['id'] == j))]
            track_res_sub = track_res_updated[track_res_updated['frame'] == current_frame]
            ids = ','.join(map(str, sorted(track_res_sub['id'])))
            track_res_sub.index = track_res_sub['id'].tolist()
            overlapped_area_ratio_matrix = calculate_overlapped_area_ratio(frame_trajectoryDF=track_res_sub)
            max_ratio = overlapped_area_ratio_matrix.max(axis=None)
            frame_info_updated.loc[frame_info_updated['frame'] == current_frame, 'id_counts'] = len(track_res_sub['id'])
            frame_info_updated.loc[frame_info_updated['frame'] == current_frame,'ids'] = ids
            frame_info_updated.loc[frame_info_updated['frame'] == current_frame, 'maxRatio'] = max_ratio
    print("ID counts distribution after remove!")
    print(frame_info_updated['id_counts'].value_counts())
    return track_res_updated,frame_info_updated

def find_lost_ids(trajectoryDF, current_frame):
    df_current = trajectoryDF[trajectoryDF.frame == current_frame]
    df_previous = trajectoryDF[trajectoryDF.frame == (current_frame - 1)]
    current_frame_ids = df_current['id'].tolist()
    previous_frame_ids = df_previous['id'].tolist()
    lost_ids = list(set(previous_frame_ids) - set(current_frame_ids))
    return lost_ids

def plot_trajectory_distribution(rangeDF):
    plt.hist(rangeDF.range, bins=30, color='skyblue', alpha=0.8, log=True)
    plt.show()  # box area distribution

def parallel_range_to_gap(rangeDF, cores = 4):
    """

    return: Pandas DataFrame
    """
    def data_process(name, group):
        min2max = group['min'].values[1:] - 1
        max2min = group['max'].values + 1
        max2min = np.delete(max2min, len(max2min) - 1)
        range = min2max - max2min + 1
        return pd.DataFrame({'id': name, 'min': max2min, 'max': min2max, 'range': range})

    def applyParallel(dfGrouped, func):
        # 多线程方式，执行某个function，并合并结果
        res = Parallel(n_jobs=cores)(delayed(func)(name, group) for name, group in dfGrouped)
        return pd.concat(res)
    # remove ids without gaps
    id_counts_df = rangeDF['id'].value_counts()
    ids_with_gaps = id_counts_df.index[id_counts_df.values > 1]
    df = rangeDF[rangeDF['id'].isin(ids_with_gaps)]
    dfGrouped = df.groupby('id')
    result = applyParallel(dfGrouped, data_process)
    return result

def calculate_scores(mat, old_ids, new_id):
    sequence_new_id = reverse_01_values(mat.loc[new_id].values.tolist())
    scores = dict()
    for i in range(len(old_ids)):
        scores[old_ids[i]] = 0
    for j in old_ids:
        input_sequence = mat.loc[j].values.tolist()
        match_score = score_sequence(input_sequence, sequence_new_id)
        scores[j] = match_score
    return scores

def score_sequence(AList, BList):
    res = 0
    for i in range(len(AList)):
        if AList[i] == BList[i]:
            res+=1
    return res

def statics_gap(target_id, neighbour_ids, cap, YOLO_model, gap_start, gap_end):
    target_forward_view, target_forward_score = get_view_and_score(cap=cap, YOLO_model=YOLO_model, object_id=target_id,
                                                                   frame_current=gap_start - 1, direction='forward',
                                                                   view_length=6)
    target_backward_view, target_backward_score = get_view_and_score(cap=cap, YOLO_model=YOLO_model, object_id=target_id,
                                                                     frame_current=gap_end + 1, direction='backward',
                                                                     view_length=6)
    view_statics_dict = {'object_id': target_id, 'type': 'target', 'forward_view': target_forward_view,
                         'forward_score': target_forward_score,'backward_view': target_backward_view,
                         'backward_score': target_backward_score}
    view_statics = pd.DataFrame(data=view_statics_dict, index=[0])
    for A_neighbour in neighbour_ids:
        neighbour_forward_view, neighbour_forward_score = get_view_and_score(cap=cap, YOLO_model=YOLO_model,
                                                                             object_id=A_neighbour,
                                                                             frame_current=gap_start - 1,
                                                                             direction='forward', view_length=6)
        neighbour_backward_view, neighbour_backward_score = get_view_and_score(cap=cap, YOLO_model=YOLO_model,
                                                                               object_id=A_neighbour,
                                                                               frame_current=gap_end + 1,
                                                                               direction='backward', view_length=6)
        neighbour_view_statics_dict = {'object_id': A_neighbour, 'type': 'neighbour',
                                       'forward_view': neighbour_forward_view, 'forward_score': neighbour_forward_score,
                                       'backward_view': neighbour_backward_view, 'backward_score': neighbour_backward_score}
        neighbour_view_statics = pd.DataFrame(data=neighbour_view_statics_dict, index=[0])
        view_statics = pd.concat([view_statics, neighbour_view_statics], axis=0)
        view_statics.reset_index(drop = True)
    return view_statics

def get_view_and_score(cap, YOLO_model, object_id, frame_current, direction, view_length):
    views = get_non_cross_view(cap=cap, YOLO_model=YOLO_model, object_id=object_id,frame_current=frame_current,
                               direction=direction, view_length=view_length)
    views.reverse()
    view, score = score_list(views)
    return view, score

def get_possible_neighbours_from_gapDF(gap_before, gap_end):
    if gap_before != '':
        neighbours_before = [int(i) for i in gap_before.split(',')]
    else:
        neighbours_before = []
    if gap_end != '':
        neighbours_after = [int(i) for i in gap_end.split(',')]
    else:
        neighbours_after = []
    neighbour_candidates = list(set(neighbours_before + neighbours_after))
    return neighbour_candidates

def get_real_neighbour(trajectoryDF, possible_neighbours, YOLO_model, forward_extend = 5, backward_extend = 5):
    # 如果neighbours数目大于1，首先判定哪个是真正的发生cross的果蝇
    # 判定方法，gap以及gap前后各5frames作为目标区域，如果该区域一个cross也没有，则删除此neighbour candidate.
    trajectory_neighbours = trajectoryDF[trajectoryDF['id'].isin(possible_neighbours)]
    trajectory_neighbours = trajectory_neighbours[(trajectory_neighbours['frame'] >= (gap_start - forward_extend)) &
                                                  (trajectory_neighbours['frame'] <= (gap_end + backward_extend))]
    trajectory_neighbours = predict_category(trajectoryDF=trajectory_neighbours, cap=cap,model_path=YOLO_model)

    neighbour_real = list()
    for id, data in trajectory_neighbours.groupby('id'):
        if 'cross' in data['view'].tolist():
            neighbour_real.append(id)
    if len(neighbour_real) != 1:
        print('has more than one neighbour!')
    return neighbour_real

def calculate_trajectory_mapping_scores(mat, old_ids, new_id):
    """
    对于一个新产生的trajectory，寻找与其对应的先前的trajectory。
    :param mat:
    :param old_ids: 所有旧IDs。
    :param new_id: 新trajectory对应的ID。
    :return: a pandas dataframe, with scores for each old id. id with the maximum value should be the lost id.
    """
    sequence_new_id = reverse_01_values(mat.loc[new_id].values.tolist())
    scores = dict()
    for i in range(len(old_ids)):
        scores[old_ids[i]] = 0
    for j in old_ids:
        input_sequence = mat.loc[j].values.tolist()
        match_score = score_sequence(input_sequence, sequence_new_id)
        scores[j] = match_score
    return scores
