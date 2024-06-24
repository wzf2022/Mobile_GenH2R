``` bash
conda activate genh2r
cd /share1/haoran/HRI/GenH2R
```
## environment
验证：在 10.210.5.7 上
``` bash
CUDA_VISIBLE_DEVICES=4 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray False policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_cartesian_simultaneous_dart/3 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_cartesian_simultaneous_dart/3 policy.pointnet2.pretrained_suffix iter_80000
```
``` bash
success rate: 79/144=0.5486111111111112
contact rate: 28/144=0.19444444444444445
   drop rate: 34/144=0.2361111111111111
timeout rate: 3/144=0.020833333333333332
average done frame        : 7425.041666666667
average success done frame: 7747.962025316456
average success num steps : 57.177215189873415
average success           : 0.22168269230769228
evaluting uses 92.3297905921936 seconds
```
### 保证 determinism
#### 先验证在一样的 action 下 pybullet 结果是否一样
生成 action 并存储，在 10.210.5.12
``` bash
CUDA_VISIBLE_DEVICES=3,4,5,6 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train policy.name cartesian policy.cartesian.staged False policy.demo_dir data/demo/s0/train/cartesian_simultaneous
success rate: 389/720=0.5402777777777777
contact rate: 75/720=0.10416666666666667
   drop rate: 89/720=0.12361111111111112
timeout rate: 167/720=0.23194444444444445
average done frame        : 6625.251388888889
average success done frame: 5111.521850899743
average success num steps : 37.05398457583548
average success           : 0.3278853632478632
evaluting uses 445.65417218208313 seconds
CUDA_VISIBLE_DEVICES=3,4,5,6 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test policy.name cartesian policy.cartesian.staged False policy.demo_dir data/demo/s0/test/cartesian_simultaneous
success rate: 86/144=0.5972222222222222
contact rate: 24/144=0.16666666666666666
   drop rate: 14/144=0.09722222222222222
timeout rate: 20/144=0.1388888888888889
average done frame        : 5555.194444444444
average success done frame: 4793.127906976744
average success num steps : 34.68604651162791
average success           : 0.377071047008547
evaluting uses 98.8398802280426 seconds
```
读取 action
``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train policy.name offline policy.offline.demo_dir data/demo/s0/train/cartesian_simultaneous
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test policy.name offline policy.offline.demo_dir data/demo/s0/test/cartesian_simultaneous
10.210.5.12, 10.210.5.13 与前面一致
10.210.5.6, 10.210.5.7 不一致
CUDA_VISIBLE_DEVICES=4 python -m evaluate evaluate.scene_ids "(499,)" evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/s0/test/cartesian_simultaneous env.verbose True
```
原因是 PyKDL 给出的结果不一样
#### 用 pybullet 做 solver 试试
在 10.210.5.12 上生成
``` bash
CUDA_VISIBLE_DEVICES=3,4,5,6 RAY_DEDUP_LOGS=0 python -m evaluate env.panda.IK_solver pybullet evaluate.setup s0 evaluate.split train policy.name cartesian policy.cartesian.staged False policy.demo_dir data/demo/s0/train/cartesian_simultaneous_pybullet
success rate: 335/720=0.4652777777777778
contact rate: 70/720=0.09722222222222222
   drop rate: 64/720=0.08888888888888889
timeout rate: 251/720=0.3486111111111111
average done frame        : 7630.7861111111115
average success done frame: 5147.737313432835
average success num steps : 37.31044776119403
average success           : 0.28107297008547005
evaluting uses 494.74023389816284 seconds
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate env.panda.IK_solver pybullet evaluate.setup s0 evaluate.split test policy.name cartesian policy.cartesian.staged False policy.demo_dir data/demo/s0/test/cartesian_simultaneous_pybullet
success rate: 63/144=0.4375
contact rate: 24/144=0.16666666666666666
   drop rate: 14/144=0.09722222222222222
timeout rate: 43/144=0.2986111111111111
average done frame        : 6995.625
average success done frame: 5043.587301587301
average success num steps : 36.55555555555556
average success           : 0.26779754273504275
evaluting uses 130.98769521713257 seconds
```
``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate env.panda.IK_solver pybullet evaluate.setup s0 evaluate.split test policy.name offline policy.offline.demo_dir data/demo/s0/test/cartesian_simultaneous_pybullet
10.210.5.7, 10.210.5.12, 10.210.5.13, 10.210.5.14, 10.210.5.18 上一致
10.210.5.9 上不一致
CUDA_VISIBLE_DEVICES=4 python -m evaluate env.panda.IK_solver pybullet evaluate.scene_ids "(754,)" evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/s0/test/cartesian_simultaneous env.verbose True
在 260 step 时，尽管环境状态完全一样，IK solver 的输入也完全一样，但是结果还是不一样
```

record video 会导致结果变化，在 11 上有这个问题，在 7 上没有这个问题
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_original_cartesian_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000 policy.demo_dir data/models/s0/omg_original_cartesian_simultaneous_dart/0 policy.demo_structure flat policy.record_ego_video True policy.record_third_person_video True
```
### env/panda RobotKinematics 里面的东西有时间重写一下
pos_action 也可以用这个来算，orientation 可以是 None
### lxm 报错
problem
``` bash
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found (required by /home/haoran/anaconda3/envs/genh2r/lib/python3.8/site-packages/lxml/etree.cpython-38-x86_64-linux-gnu.so)
```
solution
``` bash
pip uninstall lxml
pip install lxm
```
### release 条件要增加
碰撞问题复现
``` bash
CUDA_VISIBLE_DEVICES=4 python -m evaluate evaluate.scene_ids "[107]" evaluate.use_ray False policy.name cartesian env.visualize True env.show_trajectory True policy.cartesian.staged False
```
### config
有没有更合理的 config 写法？分布式的？
首先把所有名字统一成小写
### action space
感觉 3d translation+3d rotation vector 也非常合理，好像比 euler 还要合理一些，可以换一下试试，改一下 action_type 的命名
## data
### dexycb
ycb 物体的位置可以换一下，搞成 data/assets/objects/ycb/002_master_chef_can，区分一下来源

相应地，scene npz 里要存一下物体的路径，而不只是名字
### hand mesh
不能直接用 handover-sim 的 hand，要学一下用 mano_pybullet 生成默认 beta 的 hand mesh
看看 HOI4D 需不需要每个场景都按照 beta 生成一个 mesh，还是也直接用默认 beta
### t0 (synthetic)
#### objects and grasps
重新根据 acronym 和 ShapeNet 处理一遍，要做的应该只有 convex decomposition 和 SDF generation（看看是否需要 SDF，根据 OMG-Planner 的重构情况）

在处理的时候注意也要保留 texture 信息
#### hand pose
重新配一下 DexGraspNet，看看能不能与 genh2r 配在同一个环境里

重新生成 hand pose 时看看能不能搞成 deterministic 的

原先 DexGraspNet 好像有个小 bug，可以确认一下，去提个 issue

需要加入对左手的支持，如果 DexGraspNet 不支持的话就镜像翻转物体，生成完 hand pose 再翻回来
#### 轨迹生成
scene data 的空间占用相当大，需要改变存储方式

目前是物体的 (T, N, 6) 和手的 (T, 51)

对于 synthetic 来说，可以改成存储物体的 (T, N, 6) 和 object_to_hand (6,)，hand_theta (45,)，但每次读出来都需要额外花时间处理
或者存储 (T, N, 6) 和手的 (T, 6)，(45,)。花费二倍空间，但是不需要额外处理

这部分最终想好怎么做之后需要相应地改环境和 process_dexycb
### t1 (HOI4D)
整理一下之前的导入代码
## demo
### 重构 OMG Planner
### landmark
### show target grasp
``` bash
OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 env.visualize True policy.show_target_grasp True

python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name cartesian policy.cartesian.staged False policy.cartesian.verbose True env.visualize True policy.show_target_grasp True

CUDA_VISIBLE_DEVICES=0,1,2,3 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.demo_dir data/demo/s0/train/omg_cartesian_simultaneous
success rate: 557/720=0.7736111111111111
contact rate: 40/720=0.05555555555555555
   drop rate: 37/720=0.05138888888888889
timeout rate: 86/720=0.11944444444444445
average done frame        : 9169.047222222222
average success done frame: 8771.319569120287
average success num steps : 65.10951526032316
average success           : 0.25170213675213676
evaluting uses 774.2132182121277 seconds
```
## imitation
加入 flow 和 prediction

现在读取数据是单线程的，可以加个 ray 搞成多线程。但需要保持 determinism。

``` bash
# train
CUDA_VISIBLE_DEVICES=0 python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_cartesian_simultaneous train.data.seed 0 train.run_dir data/models/tmp
# evaluate
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/cartesian_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000
```
