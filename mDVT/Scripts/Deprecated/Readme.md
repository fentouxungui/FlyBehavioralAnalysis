# functions介绍

## ``resolve_ID_jump``

对于new ID出现的frame，先看一下这些new ID果蝇有无近邻（max overlap ratio, or xy distance），1. 如果没有近邻[area overlap ratio < 0.2，这个值的确定比较artificial, 我是在观测了一些new ID产生的图片来设定的，图片见frame_ID_dump目录]，并且loss的帧很少[少于10帧]，则直接将其轨迹并入到ID丢失，且距离最近的果蝇上，并对gap做平滑处理，2. 如有近邻，并入到ID丢失，且距离最近的果蝇上，并保留或人为制造一个gap。最终得到果蝇轨迹是一个标准的ID×frame的矩阵，值是xy坐标轴，含gap，即值为缺失值！

找对应ID的算法【注意，该算法被弃用，不适合所有情况】

![image-20240924092126235](../Readme_Images/image-20240924092126235.png)

![image-20240924182334849](../Readme_Images/image-20240924182334849.png)

算法有问题，不能解决这种情况！改进函数，右端延申终止于new id结束的地方或100frame的地方，取最小值。

frame space：

发现新问题，改进后的算法解决不了以下情况：

![image-20241030190553722](../Readme_Images/image-20241030190553722.png)

由于6有一个很长的缺失，图中未能显示，实际上是6变成了14！

**再次改进后的算法：**

len = min(new_id length,20)区域必须要有gap【只考虑1到object numbers的ID】，去除无gap的。 或者说只考虑那些在new id起始位置及后面20frame的区域有gap的ID，如果有gap的ID多于1个，则再且综合考虑位置new id 起始frame的坐标与有gap的其它ID的最近距离，来判定是哪个id lost。

![image-20241031105837989](../Readme_Images/image-20241031105837989.png)

发现还是有问题：上面这种情况会识别错误，

加入overlap的frame数目的比较，选择overlap数据少的那个ID。而不是依据距离了！

![image-20241031133233410](../Readme_Images/image-20241031133233410.png)

发现的新问题：上面这种情况无法结果，会有两个candidates，然后依据最近距离指定即可，对于是否要保留起始位置的gap时，**补充：如果new id起始frame处的果蝇数目少于object number，则保留gap**。





## ``summary_gaps``

> 获取所有的gaps，并且统计gap信息，比如最大重叠面积，最近邻居，并做分类，simple：无最近邻，complex-1:有最近邻，且最近邻稳定，complex-2：
> 有最近邻，但最近邻发生变化。

注意： 按照算法的漏洞，上面这个也会被标记为simple，发生overlap的两只果蝇都发生了轨迹的丢失！计算overlap max ratio时有漏洞，frame中的另一只果蝇是丢失的！导致较低的ratio值。**不过，也没有问题**！

![image-20240930145405886](../Readme_Images/image-20240930145405886.png)











## ``find_problem_regions``

用于解决gap问题，去除潜在的ID switch。

> 此函数将将所有frames区分为稳态帧和非稳态帧， 从而找出非稳态区域（由连续的非稳态帧组成），非稳态区域是问题的发生区。我们仅需要在非稳态区找出问题并进行纠正即可。其中``window_size``参数的设定尤为重要。
>
> 稳态区：id组合可以稳定连续地出现至少40帧而不发生变化，并且id数目也均为20只。其它情况均为非稳态区。

**功能**

用于从frame info中，提取出非稳态区域。非稳态区域由连续的非稳态帧组成。

单个非稳态区域可以包括一种或多种类型的problem，比如两个相邻的gap，或一个gap与一个相邻的ID jump。一般仅仅在gap的地方才可能发生ID switch。如果两个problem发生位置间隔超过40 frames，即第一个problem结束的位置与第二个problem开始的位置间隔超过40 frames，则会被分开为两个非稳态区域。

**如何判定当前帧是否为稳态**

从当前帧开始往前，共取40帧[``windown_size``]，如果这40帧的ids组合均相同，并且ids数量均为20[``object_number``]，则判定该帧为稳态帧，否则为非稳态帧！即稳态代表当前问题区域的状态至少可以稳定传播40帧。

在最开始的40 - 1 frames [``windown_size``]中，默认前40-1帧，均为稳态帧。

**为什么使用滑窗，而不是仅仅当前帧**

有时尽管当前帧与上一帧相同的IDs组合，以及正确的object数目，但也可能是错误帧。从单帧水平不能判定该帧处于稳态还是非稳态。

**window_size 参数**



**注意**

1. 要在使用完``remove_short_trajectory``和``remove_extra_IDs_in_trajectory``之后再使用本函数。此时轨迹中仅存在gap，object数目均少于object数目。

2. ``windown_size``参数很**重要**，要大于gap的长度！
3. 每一个problem region的前40-1帧和后40-1帧其实均为稳态！稳态不代表结果都是对的，仅代表结果是稳定的，错误轨迹也可以稳定传播。当然，视频最开始的40帧的轨迹必须是正确的！或许这个function叫做``find_unstable_regions``会更为准确！

非稳态区域的长度分布

# 名词解释

## ID jump 和 ID switch 的区别

ID jump是指一只果蝇的ID发生变化，通常是由于其掉落、飞行等迅速移动导致。少数情况是被由相邻果蝇遮挡导致。

ID switch是指两只果蝇移动轨迹相交，发生遮挡，导致原来的ID发生交换，被遮挡果蝇可能伴随发生ID jump。

**对于由相邻果蝇遮挡导致的ID jump，可以将新旧ID的轨迹合并成gap，变成一个gap问题！**而对于没有邻居的ID jump则直接合并新旧ID，并平滑填充gap区域即可。一般先解决ID jump问题，变为一个标准的ID×frame的矩阵后，再解决gap问题。



## 常用参数名

``trajectoryDF``：trajectory file, A pandas data frame, with columns: ``['frame', 'category', 'x', 'y', 'w', 'h', 'id', 'BoxArea']``, and can be read in by function `` read_yolo_track``.

``frame_trajectoryDF``: 提取``trajectoryDF``中的某一帧得到的pandas data frame。



``frameDF``：frame level summary file,  A pandas data frame, with columns:``['frame', 'maxRatio', 'ids', 'id_counts']``. ``maxRatio``为可选的。

``rangeDF``:  range file,  A pandas data frame, with columns:``[grop, from, to, range]``. grouped frames, usually used for plot interested regions.  Can be id range, problem region range etc.

``object_number``: 果蝇的实际数目



希望Yolo可以识别单只果蝇腹面，单只果蝇背面，单只果蝇侧面，两只果蝇交叉，

或者更详细，单只雄果蝇背面、腹部，侧面，单只雌果蝇背面、腹部、侧面，以及两只果蝇交叉。











