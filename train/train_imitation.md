conda activate genh2r
cd /share1/haoran/HRI/GenH2R

``` bash
CUDA_VISIBLE_DEVICES=6 nohup python -m train.train_imitation train.data.demo_dir data/demo/pointnet2/s0/simultaneous/train train.run_dir data/models/tmp > log/tmp.txt 2>&1 &
data/models/tmp/iter_80000.pth

CUDA_VISIBLE_DEVICES=3,4,5,6 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/tmp policy.pointnet2.pretrained_suffix iter_80000
success rate: 9/144=0.0625
contact rate: 44/144=0.3055555555555556
   drop rate: 74/144=0.5138888888888888
timeout rate: 17/144=0.11805555555555555
average done frame        : 6244.548611111111
average success done frame: 5751.666666666667
average success num steps : 34.77777777777778
average success           : 0.5576410256410257
evaluting uses 122.54272055625916 seconds
```

# demo generation
## debug
``` bash
CUDA_VISIBLE_DEVICES=4 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True env.visualize True

CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True policy.seed 0 policy.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0
success rate: 529/720=0.7347222222222223

CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name offline policy.offline.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.scene_ids "[950]" evaluate.use_ray False policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True policy.seed 0

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.scene_ids "(946, 947, 948, 950)" evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.scene_ids "(950,)" evaluate.use_ray False policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True policy.seed 0 policy.demo_dir data/demo/tmp env.verbose True policy.verbose True

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.scene_ids "(950,)" evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/tmp env.verbose True policy.verbose True

bug：在 reached 之后还继续 dart，导致在生成 demo 的时候有 dart，但是在 load demo 的时候没有
```
``` bash
CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True policy.seed 0 policy.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0
success rate: 562/720=0.7805555555555556
contact rate: 32/720=0.044444444444444446
   drop rate: 99/720=0.1375
timeout rate: 27/720=0.0375
average done frame        : 7225.079166666666
average success done frame: 7173.526690391459
average success num steps : 52.88256227758007
average success           : 0.4482671776621955
evaluting uses 581.14240193367 seconds
CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name offline policy.offline.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0
298, 591 diff
```
bug：判定 reached 之前还可能有 dart，这一步没有被算在 reach 里
``` bash
CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.scene_ids "(298,)" evaluate.use_ray False policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True policy.seed 0 policy.demo_dir data/demo/tmp policy.save_state True env.verbose True policy.verbose True

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.scene_ids "(298,)" evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/tmp env.verbose True policy.verbose True
```
``` bash
CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True policy.seed 0 policy.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0
success rate: 562/720=0.7805555555555556
contact rate: 32/720=0.044444444444444446
   drop rate: 99/720=0.1375
timeout rate: 27/720=0.0375
average done frame        : 7225.079166666666
average success done frame: 7173.526690391459
average success num steps : 52.88256227758007
average success           : 0.4482671776621955
evaluting uses 577.4184799194336 seconds
CUDA_VISIBLE_DEVICES=2,3,4,5 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name offline policy.offline.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0

CUDA_VISIBLE_DEVICES=2 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name offline policy.offline.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart/0 env.visualize True ENV.show_trajectory True
```

``` bash
CUDA_VISIBLE_DEVICES=3 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.demo_dir data/demo/tmp policy.record_third_person_video True evaluate.scene_ids "[951, 996, 953, 917]"
```
# s0
## train_pointnet2_simultaneous_dart
### demo
``` bash
nohup bash scripts/gen_demo_for_different_seeds.sh \
   0,1,2,3,4,5,6,7 \
   data/demo/s0/train/pointnet2_simultaneous_dart \
   "2000 1000 19000" \
   "evaluate.setup s0 evaluate.split train policy.name pointnet2 policy.pointnet2.pretrained_dir /share1/haoran/HRI/generalizable_handover/output/t450_bc_omg_replan5_landmark_smooth0.08_use_hand_flow0_pred0_wd0.0001_pred0.5_300w_13s_no_accum_3 policy.pointnet2.pretrained_source handoversim policy.dart True" \
   > log/demo_s0_train_pointnet2_simultaneous_dart.txt 2>&1 &
11 2526347
```
### train
``` bash
CUDA_VISIBLE_DEVICES=0 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart train.data.seed 0 train.run_dir data/models/s0/pointnet2_simultaneous_dart/0 > log/train_s0_pointnet2_simultaneous_dart_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart train.data.seed 1 train.run_dir data/models/s0/pointnet2_simultaneous_dart/1 > log/train_s0_pointnet2_simultaneous_dart_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart train.data.seed 2 train.run_dir data/models/s0/pointnet2_simultaneous_dart/2 > log/train_s0_pointnet2_simultaneous_dart_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart train.data.seed 3 train.run_dir data/models/s0/pointnet2_simultaneous_dart/3 > log/train_s0_pointnet2_simultaneous_dart_3.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/pointnet2_simultaneous_dart train.data.seed 4 train.run_dir data/models/s0/pointnet2_simultaneous_dart/4 > log/train_s0_pointnet2_simultaneous_dart_4.txt 2>&1 &
```
### evaluate
``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/pointnet2_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/pointnet2_simultaneous_dart/1 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=2,3,4 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 24 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/pointnet2_simultaneous_dart/2 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=2,3,4 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 24 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/pointnet2_simultaneous_dart/3 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=2,3,4 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 24 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/pointnet2_simultaneous_dart/4 policy.pointnet2.pretrained_suffix iter_80000
```
results
```
0
success rate: 116/144=0.8055555555555556
contact rate: 7/144=0.04861111111111111
   drop rate: 21/144=0.14583333333333334
timeout rate: 0/144=0.0
average done frame        : 6383.694444444444
average success done frame: 6474.862068965517
average success num steps : 47.41379310344828
average success           : 0.5020106100795756
evaluting uses 96.29419279098511 seconds
1
success rate: 118/144=0.8194444444444444
contact rate: 9/144=0.0625
   drop rate: 17/144=0.11805555555555555
timeout rate: 0/144=0.0
average done frame        : 6382.430555555556
average success done frame: 6411.940677966101
average success num steps : 46.94915254237288
average success           : 0.5068507170795306
evaluting uses 103.42110300064087 seconds
2
success rate: 117/144=0.8125
contact rate: 11/144=0.0763888888888889
   drop rate: 16/144=0.1111111111111111
timeout rate: 0/144=0.0
average done frame        : 6361.229166666667
average success done frame: 6447.717948717948
average success num steps : 47.23076923076923
average success           : 0.5040986193293885
evaluting uses 123.44273948669434 seconds
3
success rate: 112/144=0.7777777777777778
contact rate: 11/144=0.0763888888888889
   drop rate: 21/144=0.14583333333333334
timeout rate: 0/144=0.0
average done frame        : 6435.138888888889
average success done frame: 6549.660714285715
average success num steps : 48.044642857142854
average success           : 0.49625686813186815
evaluting uses 124.30080842971802 seconds
4
success rate: 109/144=0.7569444444444444
contact rate: 7/144=0.04861111111111111
   drop rate: 27/144=0.1875
timeout rate: 1/144=0.006944444444444444
average done frame        : 6424.013888888889
average success done frame: 6485.688073394495
average success num steps : 47.58715596330275
average success           : 0.5011778405081156
evaluting uses 125.88619327545166 seconds

```
## train_omg_original_simultaneous_dart (the stored action is joint)
``` bash
CUDA_VISIBLE_DEVICES=2 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner python -m evaluate evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.demo_dir data/demo/tmp evaluate.use_ray False env.visualize True
```
### demo
``` bash
nohup bash scripts/gen_demo_for_different_seeds.sh \
   2,3,4,5,6,7 \
   data/demo/s0/train/omg_original_simultaneous_dart \
   "0 1000 19000" \
   "OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner" \
   "evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.dart True" \
   > log/demo_s0_train_omg_original_simultaneous_dart.txt 2>&1 &
```
### train
``` bash
CUDA_VISIBLE_DEVICES=4 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_simultaneous_dart train.data.seed 0 train.run_dir data/models/s0/omg_original_simultaneous_dart/0 > log/train_s0_omg_original_simultaneous_dart_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_simultaneous_dart train.data.seed 1 train.run_dir data/models/s0/omg_original_simultaneous_dart/1 > log/train_s0_omg_original_simultaneous_dart_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_simultaneous_dart train.data.seed 2 train.run_dir data/models/s0/omg_original_simultaneous_dart/2 > log/train_s0_omg_original_simultaneous_dart_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_simultaneous_dart train.data.seed 3 train.run_dir data/models/s0/omg_original_simultaneous_dart/3 > log/train_s0_omg_original_simultaneous_dart_3.txt 2>&1 &
```
### evaluate
``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_original_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000
```
## train_omg_simultaneous_dart (the stored action is joint)
``` bash
CUDA_VISIBLE_DEVICES=7 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy python -m evaluate evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.demo_dir data/demo/tmp evaluate.use_ray False env.visualize True
```
### demo
``` bash
nohup bash scripts/gen_demo_for_different_seeds.sh \
   4,5,6,7 \
   data/demo/s0/train/omg_simultaneous_dart \
   "0 1000 19000" \
   "OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy" \
   "evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.dart True" \
   > log/demo_s0_train_omg_simultaneous_dart.txt 2>&1 &
nohup bash scripts/gen_demo_for_different_seeds.sh \
   2,3,4,5 \
   data/demo/s0/train/omg_simultaneous_dart \
   "12000 1000 12000" \
   "OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy" \
   "evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.dart True" \
   > log/demo_s0_train_omg_simultaneous_dart_12000.txt 2>&1 &
6 86130
16 116575
```
### train
``` bash
CUDA_VISIBLE_DEVICES=2 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_simultaneous_dart train.data.seed 0 train.run_dir data/models/s0/omg_simultaneous_dart/0 > log/train_s0_omg_simultaneous_dart_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_simultaneous_dart train.data.seed 1 train.run_dir data/models/s0/omg_simultaneous_dart/1 > log/train_s0_omg_simultaneous_dart_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_simultaneous_dart train.data.seed 2 train.run_dir data/models/s0/omg_simultaneous_dart/2 > log/train_s0_omg_simultaneous_dart_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_simultaneous_dart train.data.seed 3 train.run_dir data/models/s0/omg_simultaneous_dart/3 > log/train_s0_omg_simultaneous_dart_3.txt 2>&1 &
```
## train_omg_original_cartesian_simultaneous_dart
### demo
``` bash
nohup bash scripts/gen_demo_for_different_seeds.sh \
   2,3,4,5,6,7 \
   data/demo/s0/train/omg_original_cartesian_simultaneous_dart \
   "0 1000 19000" \
   "OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner" \
   "evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.dart True" \
   > log/demo_s0_train_omg_original_cartesian_simultaneous_dart.txt 2>&1 &
```
### train
``` bash
CUDA_VISIBLE_DEVICES=4 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_cartesian_simultaneous_dart train.data.seed 0 train.run_dir data/models/s0/omg_original_cartesian_simultaneous_dart/0 > log/train_s0_omg_original_cartesian_simultaneous_dart_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_cartesian_simultaneous_dart train.data.seed 1 train.run_dir data/models/s0/omg_original_cartesian_simultaneous_dart/1 > log/train_s0_omg_original_cartesian_simultaneous_dart_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_cartesian_simultaneous_dart train.data.seed 2 train.run_dir data/models/s0/omg_original_cartesian_simultaneous_dart/2 > log/train_s0_omg_original_cartesian_simultaneous_dart_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_original_cartesian_simultaneous_dart train.data.seed 3 train.run_dir data/models/s0/omg_original_cartesian_simultaneous_dart/3 > log/train_s0_omg_original_cartesian_simultaneous_dart_3.txt 2>&1 &
```
### evaluate
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_original_cartesian_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000 policy.demo_dir data/models/s0/omg_original_cartesian_simultaneous_dart/0 policy.demo_structure flat policy.record_ego_video True policy.record_third_person_video True
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_original_cartesian_simultaneous_dart/1 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_original_cartesian_simultaneous_dart/2 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_original_cartesian_simultaneous_dart/3 policy.pointnet2.pretrained_suffix iter_80000
```
``` bash
0
success rate: 83/144=0.5763888888888888
contact rate: 28/144=0.19444444444444445
   drop rate: 32/144=0.2222222222222222
timeout rate: 1/144=0.006944444444444444
average done frame        : 7033.284722222223
average success done frame: 7413.385542168675
average success num steps : 54.65060240963855
average success           : 0.24774145299145298
evaluting uses 317.62040972709656 seconds
1
success rate: 79/144=0.5486111111111112
contact rate: 31/144=0.2152777777777778
   drop rate: 32/144=0.2222222222222222
timeout rate: 2/144=0.013888888888888888
average done frame        : 7068.916666666667
average success done frame: 7415.2405063291135
average success num steps : 54.65822784810127
average success           : 0.2357238247863248
evaluting uses 90.48953008651733 seconds
2
success rate: 82/144=0.5694444444444444
contact rate: 29/144=0.2013888888888889
   drop rate: 31/144=0.2152777777777778
timeout rate: 2/144=0.013888888888888888
average done frame        : 7208.201388888889
average success done frame: 7375.463414634146
average success num steps : 54.353658536585364
average success           : 0.2464177350427351
evaluting uses 94.98426151275635 seconds
3
success rate: 80/144=0.5555555555555556
contact rate: 31/144=0.2152777777777778
   drop rate: 31/144=0.2152777777777778
timeout rate: 2/144=0.013888888888888888
average done frame        : 7052.625
average success done frame: 7386.1
average success num steps : 54.4875
average success           : 0.23995299145299143
evaluting uses 89.3283760547638 seconds
```
## train_omg_cartesian_simultaneous_dart
### demo
``` bash
nohup bash scripts/gen_demo_for_different_seeds.sh \
   2,3,4,5 \
   data/demo/s0/train/omg_cartesian_simultaneous_dart \
   "0 1000 19000" \
   "OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy" \
   "evaluate.setup s0 evaluate.split train policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ONLY_NEAREST_GRASP True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.dart True" \
   > log/demo_s0_train_omg_cartesian_simultaneous_dart.txt 2>&1 &
```
### train
``` bash
CUDA_VISIBLE_DEVICES=4 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_cartesian_simultaneous_dart train.data.seed 0 train.run_dir data/models/s0/omg_cartesian_simultaneous_dart/0 > log/train_s0_omg_cartesian_simultaneous_dart_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_cartesian_simultaneous_dart train.data.seed 1 train.run_dir data/models/s0/omg_cartesian_simultaneous_dart/1 > log/train_s0_omg_cartesian_simultaneous_dart_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_cartesian_simultaneous_dart train.data.seed 2 train.run_dir data/models/s0/omg_cartesian_simultaneous_dart/2 > log/train_s0_omg_cartesian_simultaneous_dart_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/omg_cartesian_simultaneous_dart train.data.seed 3 train.run_dir data/models/s0/omg_cartesian_simultaneous_dart/3 > log/train_s0_omg_cartesian_simultaneous_dart_3.txt 2>&1 &
```
### evaluate
on 10.210.5.7
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_cartesian_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000 policy.demo_dir data/models/s0/omg_cartesian_simultaneous_dart/0 policy.demo_structure flat policy.record_ego_video True policy.record_third_person_video True
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_cartesian_simultaneous_dart/1 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_cartesian_simultaneous_dart/2 policy.pointnet2.pretrained_suffix iter_80000
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/omg_cartesian_simultaneous_dart/3 policy.pointnet2.pretrained_suffix iter_80000
```
``` bash
0
success rate: 86/144=0.5972222222222222
contact rate: 31/144=0.2152777777777778
   drop rate: 26/144=0.18055555555555555
timeout rate: 1/144=0.006944444444444444
average done frame        : 7094.055555555556
average success done frame: 7427.011627906977
average success num steps : 54.83720930232558
average success           : 0.2560699786324786
evaluting uses 343.7916781902313 seconds
1
success rate: 90/144=0.625
contact rate: 23/144=0.1597222222222222
   drop rate: 31/144=0.2152777777777778
timeout rate: 0/144=0.0
average done frame        : 7171.368055555556
average success done frame: 7584.055555555556
average success num steps : 55.94444444444444
average success           : 0.26043002136752136
evaluting uses 93.86153364181519 seconds
2
success rate: 95/144=0.6597222222222222
contact rate: 22/144=0.1527777777777778
   drop rate: 24/144=0.16666666666666666
timeout rate: 3/144=0.020833333333333332
average done frame        : 7464.236111111111
average success done frame: 7698.557894736842
average success num steps : 56.78947368421053
average success           : 0.26908760683760685
evaluting uses 95.94236946105957 seconds
3
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
## train_cartesian_simultaneous_dart
``` bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True policy.name cartesian policy.cartesian.staged False
success rate: 420/720=0.5833333333333334
contact rate: 60/720=0.08333333333333333
   drop rate: 84/720=0.11666666666666667
timeout rate: 156/720=0.21666666666666667
average done frame        : 6685.622222222222
average success done frame: 5273.695238095238
average success num steps : 38.33571428571429
average success           : 0.5944080586080586
evaluting uses 342.9767882823944 seconds
```
### demo
``` bash
nohup bash scripts/gen_demo_for_different_seeds.sh \
   0,1,2,3,4,5,6,7 \
   data/demo/s0/train/cartesian_simultaneous_dart \
   "0 1000 19000" \
   "" \
   "evaluate.setup s0 evaluate.split train policy.name cartesian policy.cartesian.staged False policy.dart True" \
   > log/demo_s0_train_cartesian_simultaneous_dart.txt 2>&1 &
nohup bash scripts/gen_demo_for_different_seeds.sh \
   0,1,2,3,4,5,6,7 \
   data/demo/s0/train/cartesian_simultaneous_dart \
   "5000 1000 5000" \
   "" \
   "evaluate.setup s0 evaluate.split train policy.name cartesian policy.cartesian.staged False policy.dart True" \
   > log/demo_s0_train_cartesian_simultaneous_dart_5000.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3,4,5,6 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train policy.name cartesian policy.cartesian.staged False policy.demo_dir data/demo/s0/train/cartesian_simultaneous evaluate.num_runners 32
```
### train
``` bash
CUDA_VISIBLE_DEVICES=0 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/cartesian_simultaneous_dart train.data.seed 0 train.run_dir data/models/s0/cartesian_simultaneous_dart/0 > log/train_s0_cartesian_simultaneous_dart_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/cartesian_simultaneous_dart train.data.seed 1 train.run_dir data/models/s0/cartesian_simultaneous_dart/1 > log/train_s0_cartesian_simultaneous_dart_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/cartesian_simultaneous_dart train.data.seed 2 train.run_dir data/models/s0/cartesian_simultaneous_dart/2 > log/train_s0_cartesian_simultaneous_dart_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m train.train_imitation train.data.demo_dir data/demo/s0/train/cartesian_simultaneous_dart train.data.seed 3 train.run_dir data/models/s0/cartesian_simultaneous_dart/3 > log/train_s0_cartesian_simultaneous_dart_3.txt 2>&1 &
```
### evaluate
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/cartesian_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000 policy.demo_dir data/models/s0/cartesian_simultaneous_dart/0 policy.demo_structure flat policy.record_ego_video True policy.record_third_person_video True

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/cartesian_simultaneous_dart/1 policy.pointnet2.pretrained_suffix iter_80000

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/cartesian_simultaneous_dart/2 policy.pointnet2.pretrained_suffix iter_80000

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/cartesian_simultaneous_dart/3 policy.pointnet2.pretrained_suffix iter_80000
```
``` bash
0
success rate: 86/144=0.5972222222222222
contact rate: 20/144=0.1388888888888889
   drop rate: 24/144=0.16666666666666666
timeout rate: 14/144=0.09722222222222222
average done frame        : 5248.215277777777
average success done frame: 4973.988372093023
average success num steps : 36.02325581395349
average success           : 0.3687622863247863
evaluting uses 82.87949681282043 seconds
1
success rate: 84/144=0.5833333333333334
contact rate: 17/144=0.11805555555555555
   drop rate: 25/144=0.1736111111111111
timeout rate: 18/144=0.125
average done frame        : 5599.097222222223
average success done frame: 4990.928571428572
average success num steps : 36.107142857142854
average success           : 0.3594262820512821
evaluting uses 89.40417909622192 seconds
2
success rate: 90/144=0.625
contact rate: 21/144=0.14583333333333334
   drop rate: 17/144=0.11805555555555555
timeout rate: 16/144=0.1111111111111111
average done frame        : 5898.291666666667
average success done frame: 5369.1
average success num steps : 39.144444444444446
average success           : 0.36691826923076926
evaluting uses 89.14651203155518 seconds
3
success rate: 89/144=0.6180555555555556
contact rate: 19/144=0.13194444444444445
   drop rate: 18/144=0.125
timeout rate: 18/144=0.125
average done frame        : 5669.388888888889
average success done frame: 5087.955056179775
average success num steps : 36.96629213483146
average success           : 0.3762077991452991
evaluting uses 87.49499869346619 seconds
```