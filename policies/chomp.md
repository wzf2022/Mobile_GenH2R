# s0
## destination planning
### debug
#### visualize
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=12 env.panda.IK_solver=PyKDL policy=chomp chomp.know_destination=True chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.65 demo_dir=tmp/debug_chomp/s0/destination_planning demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,9,10,11,12,13,14,15,16] chomp.show_filter_results=True
```
#### test success rate
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.know_destination=True chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.65
success rate: 615/720=0.8541666666666666
contact rate: 15/720=0.020833333333333332
   drop rate: 47/720=0.06527777777777778
timeout rate: 43/720=0.059722222222222225
average done frame        : 6513.131944444444
average success done frame: 6161.178861788618
average success num steps : 45.08943089430894
average success           : 0.4494113247863248
evaluting uses 412.5963418483734 seconds
```
#### add dart
``` bash
CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.know_destination=True chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.65 dart=True
success rate: 606/720=0.8416666666666667
contact rate: 22/720=0.030555555555555555
   drop rate: 41/720=0.05694444444444444
timeout rate: 51/720=0.07083333333333333
average done frame        : 7489.065277777778
average success done frame: 7205.369636963696
average success num steps : 53.113861386138616
average success           : 0.37522991452991455
evaluting uses 2849.728663921356 seconds
```
#### test demo generation
``` bash
CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=24 env.panda.IK_solver=PyKDL policy=chomp chomp.know_destination=True chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.65 chomp.action_type=ego_cartesian demo_dir=data/demo/s0/train/chomp/destination_planning
success rate: 615/720=0.8541666666666666
contact rate: 17/720=0.02361111111111111
   drop rate: 46/720=0.06388888888888888
timeout rate: 42/720=0.058333333333333334
average done frame        : 6509.666666666667
average success done frame: 6174.9544715447155
average success num steps : 45.2
average success           : 0.4485061965811966
evaluting uses 603.1339976787567 seconds

CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=8 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/destination_planning show_target_grasp=True demo_dir=tmp/debug_chomp/s0/destination_planning_load demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,10,11,12,13]
```
#### test demo generation with ee orient loss
``` bash
CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=24 env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.know_destination=True chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.65 chomp.action_type=ego_cartesian demo_dir=data/demo/s0/train/chomp/destination_planning_with_ee_orient_loss
success rate: 629/720=0.8736111111111111
contact rate: 15/720=0.020833333333333332
   drop rate: 34/720=0.04722222222222222
timeout rate: 42/720=0.058333333333333334
average done frame        : 6638.640277777778
average success done frame: 6304.538950715421
average success num steps : 46.1764705882353
average success           : 0.45000790598290596
evaluting uses 609.550891160965 seconds

CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=8 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/destination_planning_with_ee_orient_loss show_target_grasp=True demo_dir=tmp/debug_chomp/s0/destination_planning_with_ee_orient_loss_load demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,10,11,12,13]
```
## dense planning
### debug
#### visualize
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=12 env.panda.IK_solver=PyKDL policy=chomp chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 demo_dir=tmp/debug_chomp/s0/dense_planning demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,9,10,11,12,13,14,15,16] chomp.show_filter_results=True
```
#### test success rate
``` bash
CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=24 env.panda.IK_solver=PyKDL policy=chomp chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13
success rate: 595/720=0.8263888888888888
contact rate: 15/720=0.020833333333333332
   drop rate: 58/720=0.08055555555555556
timeout rate: 52/720=0.07222222222222222
average done frame        : 7602.759722222222
average success done frame: 7322.2
average success num steps : 54.003361344537815
average success           : 0.36099209401709403
evaluting uses 2918.506758213043 seconds
```
#### add dart
``` bash
CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=24 env.panda.IK_solver=PyKDL policy=chomp chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 dart=True
success rate: 599/720=0.8319444444444445
contact rate: 20/720=0.027777777777777776
   drop rate: 52/720=0.07222222222222222
timeout rate: 49/720=0.06805555555555555
average done frame        : 8130.402777777777
average success done frame: 7911.477462437396
average success num steps : 58.52587646076795
average success           : 0.32570769230769236
evaluting uses 5060.9836983680725 seconds
```
#### test demo generation
``` bash
CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.action_type=ego_cartesian demo_dir=data/demo/s0/train/chomp/dense_planning
success rate: 581/720=0.8069444444444445
contact rate: 13/720=0.018055555555555554
   drop rate: 74/720=0.10277777777777777
timeout rate: 52/720=0.07222222222222222
average done frame        : 7618.663888888889
average success done frame: 7367.209982788296
average success num steps : 54.352839931153184
average success           : 0.3497042735042735
evaluting uses 2191.6293087005615 seconds

CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/dense_planning show_target_grasp=True demo_dir=tmp/debug_chomp/s0/dense_planning_load demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,10,11,12,13]
```
### generate
#### smooth 0.08
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=56 setup=s0 split=train env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0. chomp.replan_period=0.13 chomp.use_endpoints=False chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/dense_planning_dart start_seed=1000 end_seed=20000 step_seed=1000
success rate: 11998/14400=0.8331944444444445
contact rate: 328/14400=0.02277777777777778
   drop rate: 709/14400=0.04923611111111111
timeout rate: 1363/14400=0.09465277777777778
average success done frame   : 8082.567261210202
average success reached frame: 6478.1888648108015
average success num steps    : 59.85205867644608
```
#### smooth 0.1
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=s0 split=train env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0. chomp.replan_period=0.13 chomp.use_endpoints=False chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/dense_planning_smooth_0.1_dart start_seed=0 end_seed=20000 step_seed=1000
```
#### smooth 0.12
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=s0 split=train env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0. chomp.replan_period=0.13 chomp.use_endpoints=False chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/dense_planning_smooth_0.12_dart start_seed=0 end_seed=20000 step_seed=1000
```
## landmark planning
### debug
#### visualize
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=12 env.panda.IK_solver=PyKDL policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 demo_dir=tmp/debug_chomp/s0/landmark_planning demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,9,10,11,12,13,14,15,16] chomp.show_filter_results=True
```
#### test success rate
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4
success rate: 602/720=0.8361111111111111
contact rate: 17/720=0.02361111111111111
   drop rate: 44/720=0.06111111111111111
timeout rate: 57/720=0.07916666666666666
average done frame        : 7331.595833333334
average success done frame: 6920.102990033222
average success num steps : 50.91029900332226
average success           : 0.3911004273504274
evaluting uses 949.2466230392456 seconds
```
#### add dart
``` bash
CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 dart=True
success rate: 601/720=0.8347222222222223
contact rate: 19/720=0.02638888888888889
   drop rate: 46/720=0.06388888888888888
timeout rate: 54/720=0.075
average done frame        : 7915.506944444444
average success done frame: 7634.261231281198
average success num steps : 56.42096505823627
average success           : 0.3445950854700855
evaluting uses 2905.878100156784 seconds
```
#### test demo generation
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.action_type=ego_cartesian demo_dir=data/demo/s0/train/chomp/landmark_planning
success rate: 599/720=0.8319444444444445
contact rate: 18/720=0.025
   drop rate: 47/720=0.06527777777777778
timeout rate: 56/720=0.07777777777777778
average done frame        : 7403.120833333333
average success done frame: 7018.5943238731215
average success num steps : 51.68948247078464
average success           : 0.38284839743589744
evaluting uses 1104.830097913742 seconds

CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/landmark_planning show_target_grasp=True demo_dir=tmp/debug_chomp/s0/landmark_planning_load demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,10,11,12,13]
```
#### test demo generation dart
``` bash
CUDA_VISIBLE_DEVICES=3,4,6,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/landmark_planning_dart
success rate: 593/720=0.8236111111111111
contact rate: 19/720=0.02638888888888889
   drop rate: 52/720=0.07222222222222222
timeout rate: 56/720=0.07777777777777778
average done frame        : 7984.934722222222
average success done frame: 7712.672849915683
average success num steps : 57.02192242833052
average success           : 0.33504038461538466
evaluting uses 3117.621987104416 seconds

CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/landmark_planning_dart show_target_grasp=True demo_dir=tmp/debug_chomp/s0/landmark_planning_dart_load demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,10,11,12,13]
```
#### test demo generation with ee orient loss
``` bash
CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.action_type=ego_cartesian demo_dir=data/demo/s0/train/chomp/landmark_planning_with_ee_orient_loss
success rate: 606/720=0.8416666666666667
contact rate: 12/720=0.016666666666666666
   drop rate: 45/720=0.0625
timeout rate: 57/720=0.07916666666666666
average done frame        : 7600.388888888889
average success done frame: 7198.402640264027
average success num steps : 53.03465346534654
average success           : 0.37568098290598295
evaluting uses 974.6689636707306 seconds

CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/landmark_planning_with_ee_orient_loss show_target_grasp=True demo_dir=tmp/debug_chomp/s0/landmark_planning_with_ee_orient_loss_load demo_structure=flat record_third_person_video=True scene_ids=[5,6,7,8,10,11,12,13]
```
#### test demo generation with ee orient loss dart
``` bash
CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/landmark_planning_with_ee_orient_loss_dart
success rate: 607/720=0.8430555555555556
contact rate: 17/720=0.02361111111111111
   drop rate: 32/720=0.044444444444444446
timeout rate: 64/720=0.08888888888888889
average done frame        : 8323.034722222223
average success done frame: 7929.749588138386
average success num steps : 58.67874794069193
average success           : 0.32887275641025643
evaluting uses 4177.543011426926 seconds
```
### generate
#### smooth 0.08
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=56 setup=s0 split=train env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/landmark_planning_dart start_seed=1000 end_seed=20000 step_seed=1000
success rate: 12079/14400=0.8388194444444445
contact rate: 320/14400=0.022222222222222223
   drop rate: 667/14400=0.04631944444444445
timeout rate: 1334/14400=0.0926388888888889
average success done frame   : 7896.573805778624
average success reached frame: 6288.2481993542515
average success num steps    : 58.42437287854955
```
#### smooth 0.1
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=56 setup=s0 split=train env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/landmark_planning_smooth_0.1_dart start_seed=0 end_seed=1000 step_seed=1000
success rate: 12399/14400=0.8610416666666667
contact rate: 319/14400=0.022152777777777778
   drop rate: 700/14400=0.04861111111111111
timeout rate: 982/14400=0.06819444444444445
average success done frame   : 7234.2945398822485
average success reached frame: 5587.609484635857
average success num steps    : 53.32841358174046
```
#### smooth 0.12
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=s0 split=train env.panda.IK_solver=PyKDL policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian dart=True demo_dir=data/demo/s0/train/chomp/landmark_planning_smooth_0.12_dart start_seed=0 end_seed=20000 step_seed=1000

python -m train.data_provider obj_pose_pred_frame_num=0 seed=0 demo_dir=data/demo/s0/train/chomp/landmark_planning_smooth_0.12_dart
success rate: 12365/14400=0.8586805555555556
contact rate: 320/14400=0.022222222222222223
   drop rate: 790/14400=0.05486111111111111
timeout rate: 925/14400=0.0642361111111111
average success done frame   : 6766.564658309745
average success reached frame: 5110.214314597654
average success num steps    : 49.72689041649818

CUDA_VISIBLE_DEVICES=0,2,3,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=s0 split=train num_runners=32 env.panda.IK_solver=PyKDL policy=offline offline.demo_dir=data/demo/s0/train/chomp/landmark_planning_smooth_0.12_dart/0 show_target_grasp=True demo_dir=tmp/debug_chomp/s0/landmark_planning_smooth_0.12_dart_0_load demo_structure=flat record_third_person_video=True record_ego_video=True offline.check_input_pcd=True scene_ids=[5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23] overwrite_demo=True
```
# t0
## destination planning
### generate 
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.know_destination=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/destination_planning
success rate: 20614/56560=0.36446251768033944
contact rate: 18248/56560=0.3226308345120226
   drop rate: 14287/56560=0.2525990099009901
timeout rate: 3409/56560=0.06027227722772277
average success done frame   : 10460.208887164063
average success reached frame: 8704.941690113516
average success num steps    : 77.70874163190065
average success              : 0.07123254814492438
evaluting uses 27734.867873191833 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=8 end_traj_idx=16 env.panda.IK_solver=PyKDL env.stop_moving_time=8 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.know_destination=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/destination_planning
success rate: 20109/56560=0.35553394625176804
contact rate: 18503/56560=0.32713932107496463
   drop rate: 14321/56560=0.2532001414427157
timeout rate: 3626/56560=0.06410891089108911
average success done frame   : 10452.45800387886
average success reached frame: 8704.801730568402
average success num steps    : 71.10492814162812
average success              : 0.06969947638994668
evaluting uses 7417.305030822754 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=16 end_traj_idx=24 env.panda.IK_solver=PyKDL env.stop_moving_time=8 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.know_destination=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/destination_planning
success rate: 20511/56560=0.36264144271570015
contact rate: 18426/56560=0.32577793493635077
   drop rate: 14185/56560=0.2507956152758133
timeout rate: 3437/56560=0.060767326732673266
average success done frame   : 10459.651552825313
average success reached frame: 8707.901857539857
average success num steps    : 78.22870654770611
average success              : 0.07089217440974867

CUDA_VISIBLE_DEVICES=2 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=8 setup=t0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.panda.IK_solver=PyKDL env.stop_moving_time=8 policy=offline offline.demo_dir=data/demo/t0/train/chomp/destination_planning show_target_grasp=True demo_dir=tmp/debug_chomp/t0/destination_planning_load demo_structure=flat record_third_person_video=True
```
## dense planning
### debug
``` bash
CUDA_VISIBLE_DEVICES=3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=t0 split=train start_object_idx=0 end_object_idx=1000 start_traj_idx=0 end_traj_idx=2 num_runners=32 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 chomp.replan_period=0.13
```
### generate
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=56 setup=t0 split=train start_traj_idx=0 end_traj_idx=5 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning
success rate: 22476/35350=0.6358132956152758
contact rate: 7486/35350=0.21176803394625177
   drop rate: 2864/35350=0.08101838755304101
timeout rate: 2524/35350=0.0714002828854314
average success done frame   : 5675.762146289376
average success reached frame: 4375.0759031856205
average success num steps    : 36.975751913151804
average success              : 0.3582679708410401
evaluting uses 40969.968943834305 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=56 setup=t0 split=train start_traj_idx=5 end_traj_idx=10 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=10 end_traj_idx=15 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning
success rate: 22239/35350=0.6291089108910891
contact rate: 7602/35350=0.21504950495049505
   drop rate: 2856/35350=0.0807920792079208
timeout rate: 2652/35350=0.07502121640735503
average success done frame   : 5706.319528755789
average success reached frame: 4402.86051531094
average success num steps    : 41.874095058231035
average success              : 0.3530114220433032
evaluting uses 84670.30779314041 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=15 end_traj_idx=20 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning
success rate: 22414/35350=0.6340594059405941
contact rate: 7582/35350=0.21448373408769447
   drop rate: 2938/35350=0.08311173974540312
timeout rate: 2415/35350=0.06831683168316832
average success done frame   : 5725.53060587133
average success reached frame: 4421.217988757026
average success num steps    : 42.012715267243685
average success              : 0.35485229246001526
evaluting uses 83168.74366092682 seconds
```
### generate smooth 0.12
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12
success rate: 36121/56560=0.6386315417256011
contact rate: 11681/56560=0.20652404526166904
   drop rate: 4987/56560=0.08817185289957567
timeout rate: 3770/56560=0.06665487977369165
average success done frame   : 4969.402148334763
average success reached frame: 3679.3587940533207
average success num steps    : 36.203122837130756
average success              : 0.394556286040692
evaluting uses 113216.45863580704 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=8 end_traj_idx=16 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12
success rate: 35957/56560=0.6357319660537483
contact rate: 11819/56560=0.20896393210749648
   drop rate: 4823/56560=0.08527227722772278
timeout rate: 3961/56560=0.07003182461103254
average success done frame   : 4996.411936479684
average success reached frame: 3708.8991295158107
average success num steps    : 36.41057374085713
average success              : 0.3914440390055489
evaluting uses 124788.13992404938 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=16 end_traj_idx=24 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12
success rate: 35938/56560=0.6353960396039604
contact rate: 11929/56560=0.21090876944837342
   drop rate: 4921/56560=0.0870049504950495
timeout rate: 3771/56560=0.06667256011315417
average success done frame   : 4997.94387556347
average success reached frame: 3696.8246702654574
average success num steps    : 36.41596638655462
average success              : 0.39116232047655314
evaluting uses 114371.4657740593 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=24 end_traj_idx=32 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12
success rate: 35942/56560=0.6354667609618104
contact rate: 11897/56560=0.21034299858557284
   drop rate: 4833/56560=0.08544908062234795
timeout rate: 3888/56560=0.06874115983026874
average success done frame   : 4984.214567914974
average success reached frame: 3697.823882922486
average success num steps    : 36.30827444215681
average success              : 0.39187697475791533
evaluting uses 122979.2954556942 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=32 end_traj_idx=40 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12
7 paused

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=40 end_traj_idx=48 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12
6

CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 setup=t0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian show_target_grasp=True demo_dir=tmp/debug_chomp/t0/dense_planning_modify_orientation_loss_load demo_structure=flat record_ego_video=True record_third_person_video=True overwrite_demo=True
success rate: 18/26=0.6923076923076923
contact rate: 6/26=0.23076923076923078
   drop rate: 2/26=0.07692307692307693
timeout rate: 0/26=0.0
average success done frame   : 5230.166666666667
average success reached frame: 3943.3333333333335
average success num steps    : 38.388888888888886
average success              : 0.41383136094674555
evaluting uses 279.34674644470215 seconds
```
### generate smooth 0.12 wo orient loss
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.12_wo_orient_loss
```
### generate smooth 0.08
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.08 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.08
17

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=4 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.use_endpoints=False chomp.foresee_time=0. chomp.replan_period=0.13 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.08 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/dense_planning_smooth_0.08
7
```
## landmark planning
### debug
``` bash
CUDA_VISIBLE_DEVICES=3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=t0 split=train start_object_idx=0 end_object_idx=1000 start_traj_idx=0 end_traj_idx=2 num_runners=24 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 print_failure_ids=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01
success rate: 975/1600=0.609375
contact rate: 417/1600=0.260625
   drop rate: 123/1600=0.076875
timeout rate: 85/1600=0.053125
average done frame        : 5543.263125
average success done frame: 5353.878974358974
average success num steps : 39.18153846153846
average success           : 0.35845879807692305
contact indices: [1000002, 1000013, 1000024, 1000025, 1000027, 1000030, 1000031, 1000034, 1000036, 1000037, 1000040, 1000051, 1000053, 1000062, 1000074, 1000076, 1000084, 1000091, 1000093, 1000100, 1000105, 1000114, 1000127, 1000132, 1000134, 1000151, 1000152, 1000155, 1000156, 1000167, 1000175, 1000181, 1000183, 1000186, 1000193, 1000202, 1000204, 1000207, 1000213, 1000214, 1000224, 1000227, 1000235, 1000240, 1000241, 1000243, 1000244, 1000246, 1000256, 1000262, 1000263, 1000265, 1000272, 1000273, 1000274, 1000275, 1000276, 1000282, 1000304, 1000305, 1000306, 1000307, 1000310, 1000316, 1000317, 1000327, 1000331, 1000334, 1000341, 1000353, 1000356, 1000361, 1000363, 1000367, 1000380, 1000390, 1000396, 1000401, 1000406, 1000412, 1000413, 1000421, 1000431, 1000434, 1000435, 1000436, 1000447, 1000451, 1000452, 1000453, 1000455, 1000456, 1000463, 1000464, 1000471, 1000480, 1000481, 1000482, 1000486, 1000492, 1000496, 1000501, 1000503, 1000506, 1000521, 1000522, 1000523, 1000527, 1000536, 1000537, 1000542, 1000543, 1000545, 1000552, 1000555, 1000556, 1000561, 1000562, 1000567, 1000570, 1000572, 1000574, 1000576, 1000577, 1000580, 1000582, 1000590, 1000601, 1000603, 1000606, 1000611, 1000617, 1000620, 1000621, 1000622, 1000625, 1000634, 1000635, 1000636, 1000637, 1000640, 1000641, 1000643, 1000651, 1000654, 1000657, 1000660, 1000661, 1000662, 1000665, 1000672, 1000674, 1000675, 1000685, 1000687, 1000691, 1000692, 1000693, 1000694, 1000701, 1000716, 1000723, 1000736, 1000740, 1000747, 1000753, 1000762, 1000763, 1000767, 1000770, 1000774, 1000775, 1000784, 1000785, 1000787, 1000792, 1000794, 1000795, 1000802, 1000820, 1000823, 1000824, 1000826, 1000831, 1000834, 1000836, 1000844, 1000846, 1000847, 1000850, 1000852, 1000857, 1000865, 1000866, 1000867, 1000871, 1000872, 1000874, 1000875, 1000882, 1000883, 1000886, 1000887, 1000893, 1000895, 1000896, 1000900, 1000907, 1000917, 1000924, 1000930, 1000932, 1000934, 1000947, 1000964, 1000970, 1000971, 1000974, 1000975, 1000977, 1008851, 1008860, 1008861, 1008869, 1008878, 1008881, 1008883, 1008888, 1008891, 1008893, 1008896, 1008898, 1008906, 1008912, 1008929, 1008936, 1008939, 1008951, 1008958, 1008962, 1008970, 1008977, 1008983, 1008987, 1008991, 1009001, 1009011, 1009020, 1009022, 1009031, 1009036, 1009040, 1009041, 1009052, 1009053, 1009056, 1009058, 1009060, 1009068, 1009071, 1009077, 1009078, 1009079, 1009082, 1009083, 1009087, 1009093, 1009098, 1009100, 1009101, 1009112, 1009118, 1009126, 1009137, 1009139, 1009140, 1009146, 1009150, 1009167, 1009176, 1009188, 1009198, 1009199, 1009201, 1009209, 1009210, 1009212, 1009220, 1009221, 1009227, 1009232, 1009233, 1009240, 1009242, 1009267, 1009270, 1009272, 1009276, 1009279, 1009282, 1009283, 1009287, 1009293, 1009307, 1009311, 1009316, 1009317, 1009318, 1009319, 1009320, 1009322, 1009331, 1009332, 1009336, 1009337, 1009343, 1009346, 1009356, 1009357, 1009362, 1009363, 1009370, 1009378, 1009379, 1009381, 1009386, 1009391, 1009392, 1009397, 1009407, 1009409, 1009411, 1009413, 1009418, 1009426, 1009433, 1009440, 1009446, 1009448, 1009450, 1009451, 1009452, 1009456, 1009458, 1009461, 1009467, 1009468, 1009470, 1009476, 1009478, 1009487, 1009488, 1009500, 1009527, 1009528, 1009529, 1009533, 1009536, 1009542, 1009546, 1009557, 1009559, 1009560, 1009568, 1009569, 1009573, 1009587, 1009590, 1009593, 1009599, 1009610, 1009613, 1009618, 1009620, 1009629, 1009631, 1009633, 1009636, 1009640, 1009642, 1009649, 1009650, 1009651, 1009656, 1009659, 1009668, 1009672, 1009673, 1009677, 1009682, 1009691, 1009696, 1009697, 1009700, 1009708, 1009709, 1009717, 1009720, 1009726, 1009728, 1009731, 1009733, 1009739, 1009740, 1009748, 1009749, 1009750, 1009756, 1009757, 1009758, 1009779, 1009783, 1009798, 1009816, 1009820, 1009822, 1009827]
   drop indices: [1000011, 1000057, 1000087, 1000104, 1000131, 1000140, 1000145, 1000154, 1000161, 1000176, 1000212, 1000252, 1000254, 1000290, 1000295, 1000297, 1000312, 1000314, 1000343, 1000344, 1000373, 1000381, 1000385, 1000387, 1000432, 1000446, 1000450, 1000502, 1000511, 1000553, 1000557, 1000560, 1000566, 1000585, 1000596, 1000600, 1000627, 1000645, 1000646, 1000655, 1000663, 1000676, 1000706, 1000712, 1000717, 1000724, 1000733, 1000750, 1000761, 1000773, 1000791, 1000812, 1000814, 1000821, 1000861, 1000890, 1000897, 1000905, 1000915, 1000921, 1000925, 1000961, 1000981, 1000982, 1000990, 1000995, 1008836, 1008840, 1008847, 1008850, 1008892, 1008907, 1008919, 1008922, 1008938, 1008942, 1008956, 1008957, 1008972, 1009046, 1009059, 1009067, 1009076, 1009092, 1009102, 1009109, 1009123, 1009148, 1009153, 1009173, 1009203, 1009226, 1009230, 1009248, 1009263, 1009299, 1009300, 1009330, 1009398, 1009416, 1009422, 1009480, 1009510, 1009516, 1009548, 1009549, 1009562, 1009571, 1009582, 1009600, 1009606, 1009608, 1009609, 1009630, 1009657, 1009660, 1009669, 1009719, 1009746, 1009763, 1009768, 1009769, 1009812]
timeout indices: [1000190, 1000231, 1000267, 1000296, 1000300, 1000315, 1000320, 1000324, 1000326, 1000330, 1000335, 1000346, 1000351, 1000354, 1000355, 1000370, 1000371, 1000372, 1000397, 1000420, 1000460, 1000465, 1000490, 1000581, 1000666, 1000673, 1000704, 1000755, 1000756, 1000771, 1000781, 1000816, 1000822, 1000931, 1000937, 1000940, 1000965, 1000967, 1000976, 1000992, 1000997, 1008968, 1009026, 1009103, 1009132, 1009136, 1009151, 1009156, 1009162, 1009166, 1009171, 1009179, 1009182, 1009187, 1009190, 1009191, 1009206, 1009207, 1009208, 1009256, 1009296, 1009301, 1009323, 1009326, 1009339, 1009401, 1009412, 1009417, 1009509, 1009540, 1009591, 1009592, 1009597, 1009607, 1009617, 1009652, 1009658, 1009718, 1009767, 1009776, 1009801, 1009803, 1009826, 1009828, 1009833]
evaluting uses 7196.896961212158 seconds
```
``` bash
CUDA_VISIBLE_DEVICES=3,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate setup=t0 split=train start_object_idx=0 end_object_idx=20 start_traj_idx=0 end_traj_idx=2 num_runners=24 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 print_failure_ids=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01 demo_dir=tmp/debug_chomp/t0/landmark_planning demo_structure=flat record_third_person_video=True
```
``` bash
CUDA_VISIBLE_DEVICES=3 python -m evaluate use_ray=False setup=t0 split=train start_object_idx=0 end_object_idx=20 start_traj_idx=0 end_traj_idx=2 num_runners=24 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 print_failure_ids=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01 demo_dir=tmp/debug_chomp/t0/landmark_planning demo_structure=flat record_third_person_video=True scene_ids=[1000002] verbose=True env.verbose=True chomp.show_target_grasp=True chomp.show_filter_results=True

CUDA_VISIBLE_DEVICES=3 python -m evaluate use_ray=False setup=t0 split=train start_object_idx=0 end_object_idx=20 start_traj_idx=0 end_traj_idx=2 num_runners=24 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 print_failure_ids=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01 chomp.use_endpoints=False demo_dir=tmp/debug_chomp/t0/no_endpoints demo_structure=flat record_third_person_video=True scene_ids=[1000002] verbose=True env.verbose=True chomp.show_target_grasp=True chomp.show_filter_results=True
```
``` bash
CUDA_VISIBLE_DEVICES=3 python -m evaluate use_ray=False setup=t0 split=train start_object_idx=0 end_object_idx=20 start_traj_idx=0 end_traj_idx=2 num_runners=24 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 print_failure_ids=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01_use_bbox chomp.use_endpoints=False demo_dir=tmp/debug_chomp/t0/no_endpoints demo_structure=flat record_third_person_video=True scene_ids=[1000002] verbose=True env.verbose=True chomp.show_target_grasp=True chomp.show_filter_results=True
```
``` bash
CUDA_VISIBLE_DEVICES=1,5,6 RAY_DEDUP_LOGS=0 python -m evaluate setup=t0 split=train start_object_idx=0 end_object_idx=1000 start_traj_idx=0 end_traj_idx=2 num_runners=24 env.panda.IK_solver=PyKDL policy=chomp env.stop_moving_time=8 env.stop_moving_dist=0.1 print_failure_ids=True chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0.01_use_bbox
success rate: 988/1600=0.6175
contact rate: 311/1600=0.194375
   drop rate: 145/1600=0.090625
timeout rate: 156/1600=0.0975
average done frame        : 5770.280625
average success done frame: 5176.254048582996
average success num steps : 37.82793522267207
average success           : 0.3716754326923077
contact indices: [1000003, 1000013, 1000021, 1000025, 1000027, 1000030, 1000031, 1000034, 1000036, 1000037, 1000040, 1000051, 1000053, 1000074, 1000094, 1000100, 1000113, 1000114, 1000126, 1000127, 1000134, 1000136, 1000151, 1000152, 1000155, 1000186, 1000192, 1000193, 1000202, 1000204, 1000207, 1000213, 1000222, 1000227, 1000237, 1000240, 1000256, 1000265, 1000272, 1000274, 1000282, 1000305, 1000316, 1000317, 1000327, 1000331, 1000341, 1000353, 1000361, 1000363, 1000385, 1000390, 1000431, 1000434, 1000435, 1000436, 1000451, 1000452, 1000453, 1000455, 1000456, 1000464, 1000466, 1000471, 1000480, 1000483, 1000486, 1000496, 1000501, 1000505, 1000522, 1000523, 1000527, 1000530, 1000536, 1000543, 1000545, 1000552, 1000555, 1000556, 1000561, 1000574, 1000582, 1000590, 1000601, 1000606, 1000611, 1000617, 1000620, 1000621, 1000625, 1000635, 1000636, 1000641, 1000643, 1000645, 1000651, 1000657, 1000660, 1000661, 1000665, 1000666, 1000672, 1000675, 1000691, 1000693, 1000694, 1000716, 1000723, 1000762, 1000763, 1000767, 1000770, 1000775, 1000777, 1000784, 1000787, 1000792, 1000795, 1000797, 1000802, 1000821, 1000824, 1000826, 1000831, 1000834, 1000844, 1000850, 1000852, 1000855, 1000857, 1000861, 1000865, 1000867, 1000871, 1000872, 1000873, 1000875, 1000882, 1000884, 1000886, 1000887, 1000893, 1000895, 1000896, 1000900, 1000917, 1000921, 1000924, 1000930, 1000936, 1000947, 1000964, 1000970, 1000971, 1000975, 1000984, 1008857, 1008869, 1008878, 1008880, 1008883, 1008891, 1008896, 1008906, 1008911, 1008912, 1008936, 1008951, 1008957, 1008962, 1008977, 1008983, 1008987, 1009011, 1009020, 1009031, 1009036, 1009040, 1009052, 1009053, 1009056, 1009058, 1009060, 1009070, 1009077, 1009078, 1009079, 1009083, 1009086, 1009087, 1009093, 1009100, 1009112, 1009126, 1009139, 1009150, 1009152, 1009176, 1009198, 1009199, 1009212, 1009220, 1009221, 1009227, 1009232, 1009233, 1009266, 1009270, 1009272, 1009276, 1009279, 1009281, 1009282, 1009283, 1009287, 1009292, 1009293, 1009301, 1009311, 1009317, 1009318, 1009320, 1009321, 1009322, 1009331, 1009332, 1009336, 1009338, 1009346, 1009356, 1009362, 1009367, 1009373, 1009378, 1009391, 1009392, 1009393, 1009397, 1009402, 1009409, 1009418, 1009440, 1009442, 1009450, 1009452, 1009456, 1009458, 1009461, 1009467, 1009469, 1009470, 1009478, 1009482, 1009487, 1009492, 1009500, 1009523, 1009529, 1009531, 1009536, 1009556, 1009558, 1009559, 1009560, 1009568, 1009569, 1009587, 1009593, 1009599, 1009610, 1009612, 1009613, 1009618, 1009621, 1009629, 1009630, 1009631, 1009633, 1009638, 1009642, 1009647, 1009649, 1009651, 1009653, 1009659, 1009660, 1009672, 1009673, 1009682, 1009691, 1009697, 1009698, 1009700, 1009708, 1009709, 1009717, 1009720, 1009726, 1009728, 1009731, 1009739, 1009749, 1009756, 1009783, 1009786, 1009789, 1009798, 1009820, 1009822, 1009827]
   drop indices: [1000011, 1000042, 1000043, 1000052, 1000057, 1000077, 1000090, 1000096, 1000104, 1000131, 1000135, 1000140, 1000145, 1000154, 1000161, 1000176, 1000181, 1000212, 1000216, 1000254, 1000270, 1000287, 1000290, 1000295, 1000297, 1000310, 1000314, 1000343, 1000344, 1000352, 1000381, 1000387, 1000421, 1000432, 1000446, 1000450, 1000476, 1000494, 1000500, 1000511, 1000541, 1000560, 1000562, 1000566, 1000594, 1000646, 1000662, 1000667, 1000676, 1000680, 1000683, 1000706, 1000712, 1000717, 1000724, 1000733, 1000747, 1000750, 1000753, 1000761, 1000773, 1000791, 1000812, 1000890, 1000897, 1000902, 1000904, 1000915, 1000926, 1000932, 1000934, 1000961, 1000974, 1000982, 1000990, 1008840, 1008847, 1008881, 1008886, 1008887, 1008888, 1008892, 1008893, 1008907, 1008919, 1008922, 1008938, 1008942, 1008956, 1008958, 1009022, 1009048, 1009059, 1009076, 1009092, 1009102, 1009121, 1009123, 1009159, 1009173, 1009183, 1009202, 1009213, 1009230, 1009238, 1009243, 1009248, 1009257, 1009263, 1009268, 1009299, 1009300, 1009307, 1009330, 1009357, 1009358, 1009387, 1009398, 1009407, 1009411, 1009422, 1009431, 1009451, 1009516, 1009527, 1009542, 1009548, 1009549, 1009550, 1009582, 1009600, 1009606, 1009608, 1009609, 1009620, 1009648, 1009657, 1009733, 1009762, 1009769, 1009799, 1009808, 1009809, 1009812, 1009813]
timeout indices: [1000014, 1000024, 1000062, 1000076, 1000093, 1000132, 1000182, 1000190, 1000231, 1000235, 1000243, 1000244, 1000246, 1000261, 1000262, 1000263, 1000267, 1000273, 1000296, 1000300, 1000304, 1000315, 1000320, 1000324, 1000326, 1000330, 1000335, 1000336, 1000346, 1000351, 1000354, 1000355, 1000356, 1000367, 1000370, 1000371, 1000372, 1000380, 1000406, 1000410, 1000412, 1000420, 1000460, 1000465, 1000490, 1000502, 1000503, 1000521, 1000537, 1000542, 1000554, 1000570, 1000576, 1000577, 1000580, 1000581, 1000585, 1000604, 1000640, 1000673, 1000674, 1000692, 1000704, 1000755, 1000756, 1000771, 1000781, 1000785, 1000803, 1000816, 1000820, 1000822, 1000823, 1000846, 1000931, 1000937, 1000940, 1000965, 1000967, 1000976, 1000992, 1000997, 1008860, 1008898, 1008929, 1008968, 1009026, 1009071, 1009080, 1009082, 1009097, 1009098, 1009103, 1009109, 1009131, 1009132, 1009136, 1009137, 1009140, 1009151, 1009156, 1009162, 1009166, 1009171, 1009179, 1009182, 1009187, 1009190, 1009191, 1009192, 1009206, 1009207, 1009208, 1009210, 1009226, 1009242, 1009256, 1009296, 1009323, 1009326, 1009339, 1009341, 1009370, 1009371, 1009379, 1009396, 1009416, 1009417, 1009433, 1009476, 1009506, 1009509, 1009510, 1009528, 1009540, 1009591, 1009592, 1009597, 1009607, 1009617, 1009627, 1009640, 1009652, 1009658, 1009666, 1009718, 1009750, 1009757, 1009767, 1009776, 1009779, 1009801, 1009803, 1009826, 1009828, 1009833]
evaluting uses 2327.4173686504364 seconds
```
### generate
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=10 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning
success rate: 44814/70700=0.6338613861386139
contact rate: 14940/70700=0.21131541725601133
   drop rate: 6171/70700=0.08728429985855729
timeout rate: 4773/70700=0.0675106082036775
average success done frame   : 4763.29457312447
average success reached frame: 3509.0599812558576
average success num steps    : 27.306578301423663
average success              : 0.40165872157545424

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=10 end_traj_idx=15 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning
success rate: 22375/35350=0.632956152758133
contact rate: 7434/35350=0.2102970297029703
   drop rate: 3093/35350=0.0874964639321075
timeout rate: 2448/35350=0.06925035360678924
average success done frame   : 4793.532201117318
average success reached frame: 3537.3303240223463
average success num steps    : 34.85886033519553
average success              : 0.3996128647590034
evaluting uses 20546.867970705032 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=15 end_traj_idx=20 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning
success rate: 22352/35350=0.6323055162659124
contact rate: 7601/35350=0.21502121640735503
   drop rate: 3088/35350=0.08735502121640736
timeout rate: 2308/35350=0.0652899575671853
average success done frame   : 4789.932355046528
average success reached frame: 3525.0632158196136
average success num steps    : 34.83191660701503
average success              : 0.3993771820258949
evaluting uses 22273.29518675804 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=20 end_traj_idx=25 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.endpoints_dir=data/endpoints/s0_6d_delta_130_pred_4 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.1 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning
success rate: 22319/35350=0.6313719943422914
contact rate: 7537/35350=0.21321074964639322
   drop rate: 3047/35350=0.0861951909476662
timeout rate: 2447/35350=0.06922206506364922
average success done frame   : 4774.94448676016
average success reached frame: 3517.647609659931
average success num steps    : 34.71459294771271
average success              : 0.3995154673049722

CUDA_VISIBLE_DEVICES=2,4,5,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 setup=t0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=offline offline.demo_dir=data/demo/t0/train/chomp/landmark_planning show_target_grasp=True demo_dir=tmp/debug_chomp/t0/landmark_planning_load demo_structure=flat record_third_person_video=True
```
### generate smooth 0.12
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12
success rate: 35749/56560=0.6320544554455445
contact rate: 11817/56560=0.20892857142857144
   drop rate: 5397/56560=0.09542079207920792
timeout rate: 3597/56560=0.0635961810466761
average success done frame   : 4159.891829142074
average success reached frame: 2912.528602198663
average success num steps    : 29.976978377017538
average success              : 0.4298509084974431

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=8 end_traj_idx=16 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12
success rate: 35444/56560=0.6266619519094767
contact rate: 11938/56560=0.21106789250353608
   drop rate: 5379/56560=0.0951025459688826
timeout rate: 3798/56560=0.06714992927864215
average success done frame   : 4172.49988714592
average success reached frame: 2931.110427716962
average success num steps    : 30.074032276266788
average success              : 0.42557577793493634
evaluting uses 27948.737160921097 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=16 end_traj_idx=24 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12
success rate: 35587/56560=0.6291902404526167
contact rate: 12015/56560=0.21242927864214992
   drop rate: 5371/56560=0.09496110325318247
timeout rate: 3587/56560=0.06341937765205093
average success done frame   : 4181.0833450417285
average success reached frame: 2926.992609660831
average success num steps    : 30.137550229016213
average success              : 0.4268773446850179

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=24 end_traj_idx=32 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12
26 killed

CUDA_VISIBLE_DEVICES=0,2,3,6 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 setup=t0 split=train start_object_idx=0 end_object_idx=128 start_traj_idx=0 end_traj_idx=1 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=offline offline.demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12 show_target_grasp=True demo_dir=tmp/debug_chomp/t0/landmark_planning_smooth_0.12_load demo_structure=flat record_third_person_video=True record_ego_video=True

CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=32 setup=t0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.planner.optimize_steps=0 chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian show_target_grasp=True chomp.show_target_grasp=True demo_dir=tmp/debug_chomp/t0/landmark_planning_modify_orientation_loss_init demo_structure=flat record_ego_video=True record_third_person_video=True overwrite_demo=True scene_ids=[1000001]
```
### generate smooth 0.12 modified orient loss
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12_modified_orient_loss
success rate: 18027/28280=0.6374469589816124
contact rate: 5825/28280=0.20597595473833097
   drop rate: 2650/28280=0.09370579915134371
timeout rate: 1778/28280=0.06287128712871287
average success done frame   : 4231.031341876075
average success reached frame: 2982.0166971764575
average success num steps    : 27.73001608698064
average success              : 0.4300299885757807
evaluting uses 9423.32917356491 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=4 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12_modified_orient_loss
success rate: 17910/28280=0.6333097595473833
contact rate: 5869/28280=0.20753182461103253
   drop rate: 2696/28280=0.09533239038189534
timeout rate: 1804/28280=0.06379066478076378
average success done frame   : 4240.431211613623
average success reached frame: 2988.5045784477943
average success num steps    : 26.44824120603015
average success              : 0.42678105483625284
evaluting uses 6896.031273841858 seconds
```
### generate smooth 0.12 wo orient loss
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12_wo_orient_loss
success rate: 17415/28280=0.6158062234794908
contact rate: 6241/28280=0.22068599717114568
   drop rate: 2898/28280=0.10247524752475247
timeout rate: 1725/28280=0.060997171145685994
average success done frame   : 3971.425724949756
average success reached frame: 2718.0859603789836
average success num steps    : 28.52058570198105
average success              : 0.42772831030355785
evaluting uses 13368.145305395126 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=4 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.12_wo_orient_loss
success rate: 17288/28280=0.6113154172560114
contact rate: 6218/28280=0.21987270155586988
   drop rate: 3026/28280=0.10700141442715701
timeout rate: 1748/28280=0.06181046676096181
average success done frame   : 3976.2586765386395
average success reached frame: 2716.073345673299
average success num steps    : 28.547316057380844
average success              : 0.42438180829071914
evaluting uses 15215.36039352417 seconds
```
### generate smooth 0.12 period 3
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.foresee_time=0.39 chomp.replan_period=0.39 chomp.planner.ee_orient_loss_coef=1. chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_3_smooth_0.12
success rate: 18213/28280=0.644024045261669
contact rate: 5816/28280=0.20565770862800567
   drop rate: 2441/28280=0.08631541725601131
timeout rate: 1809/28280=0.06396746817538897
average success done frame   : 4330.021687805413
average success reached frame: 3072.126942293966
average success num steps    : 31.28929885246802
average success              : 0.4295629637689044
evaluting uses 20486.26772427559 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=4 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.foresee_time=0.39 chomp.replan_period=0.39 chomp.planner.ee_orient_loss_coef=1. chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.12 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_3_smooth_0.12
success rate: 17947/28280=0.6346181046676096
contact rate: 5911/28280=0.20901697312588402
   drop rate: 2606/28280=0.09214992927864216
timeout rate: 1814/28280=0.06414427157001415
average success done frame   : 4327.424750654705
average success reached frame: 3070.3961107706023
average success num steps    : 31.25865047083078
average success              : 0.4234159911870308
evaluting uses 24666.035316467285 seconds
```
### generate smooth 0.08
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.08 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.08_modified_orient_loss
success rate: 18055/28280=0.6384370579915134
contact rate: 5961/28280=0.2107850070721358
   drop rate: 2221/28280=0.07853606789250353
timeout rate: 2043/28280=0.07224186704384725
average success done frame   : 5639.607809471061
average success reached frame: 4370.994959844918
average success num steps    : 41.3812240376627
average success              : 0.3615219671417691
evaluting uses 22193.059579133987 seconds

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=4 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.08 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.08_modified_orient_loss
success rate: 17722/28280=0.6266619519094767
contact rate: 6140/28280=0.21711456859971712
   drop rate: 2298/28280=0.08125884016973126
timeout rate: 2120/28280=0.07496463932107496
average success done frame   : 5620.831396004965
average success reached frame: 4348.280385960952
average success num steps    : 41.224241056314185
average success              : 0.355759297138505
evaluting uses 19854.578979969025 seconds
```
### generate smooth 0.09
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=0 end_traj_idx=4 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.09 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.09_modified_orient_loss
26

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=t0 split=train start_traj_idx=4 end_traj_idx=8 env.panda.IK_solver=PyKDL env.stop_moving_time=8 env.stop_moving_dist=0.1 policy=chomp chomp.planner.ee_orient_loss_coef=1. chomp.foresee_time=0.65 chomp.replan_period=0.65 chomp.hand_collision_free_mask_dir=env/data/hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.trajectory_smoothing_step_size=0.09 chomp.action_type=ego_cartesian demo_dir=data/demo/t0/train/chomp/landmark_planning_smooth_0.09_modified_orient_loss
13
```
## landmark planning without endpoints
