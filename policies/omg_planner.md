conda activate genh2r
cd /share1/haoran/HRI/GenH2R

# sequential
``` bash
OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name OMG_original policy.wait_time 3. env.visualize True env.verbose True
CUDA_VISIBLE_DEVICES=0,1,2,3 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name OMG_original policy.wait_time 3.
```

# sequential, filter hand collision
``` bash
OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name OMG_original policy.wait_time 3. policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True env.visualize True env.verbose True
OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name OMG_original policy.wait_time 3. policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.SHOW_FILTERING True
CUDA_VISIBLE_DEVICES=0,1,2,3 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name OMG_original policy.wait_time 3. policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True
```
```
success rate: 603/720=0.8375
contact rate: 45/720=0.0625
   drop rate: 35/720=0.04861111111111111
timeout rate: 37/720=0.05138888888888889
```

# simultaneous
## replan 1
``` bash
OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray False policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 env.visualize True

CUDA_VISIBLE_DEVICES=4,5,6,7 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name OMG_original policy.OMG_ORIGINAL.FILTER_HAND_COLLISION_GRASPS True policy.OMG_ORIGINAL.ONE_TRIAL True policy.OMG_ORIGINAL.ACTION_TYPE cartesian policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING True policy.OMG_ORIGINAL.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08 policy.OMG_ORIGINAL.SIMULTANEOUS True policy.OMG_ORIGINAL.REPLAN_PERIOD 1 policy.DEMO_DIR data/demo/omg_original/s0/sequential/train
success rate: 577/720=0.8013888888888889
contact rate: 43/720=0.059722222222222225
   drop rate: 24/720=0.03333333333333333
timeout rate: 76/720=0.10555555555555556
not deterministic???
CUDA_VISIBLE_DEVICES=4,5,6,7 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split train evaluate.use_ray True evaluate.num_runners 32 policy.name offline policy.OFFLINE.DEMO_DIR omg_original/s0/sequential/train
```

``` bash
OMG.ONE_TRIAL True OMG.ONLY_NEAREST_GRASP True OMG.TRAJECTORY_SMOOTHING True OMG.SAVE_GRASPS_NO_CONTACT_MASK True OMG.GRASP_DIR /share/haoran/HRI/handover-sim/data/acronym OMG.ONLY_FILTER_ONCE True OMG.LOAD_GRASPS_NO_CONTACT_MASK True GEN_DEMO.SIMULTANEOUS True OMG.SIMULTANEOUS True OMG.REPLAN_PERIOD 1 OMG.TRAJECTORY_SMOOTHING_STEP_SIZE 0.08
```

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy RAY_DEDUP_LOGS=0 python -m evaluate num_runners=64 setup=s0 split=train env.panda.IK_solver=PyKDL policy=omg_planner omg_planner.FILTER_HAND_COLLISION_GRASPS=True omg_planner.ONLY_NEAREST_GRASP=True omg_planner.ONE_TRIAL=True omg_planner.ACTION_TYPE=cartesian omg_planner.TRAJECTORY_SMOOTHING=True omg_planner.TRAJECTORY_SMOOTHING_STEP_SIZE=0.08 omg_planner.SIMULTANEOUS=True omg_planner.REPLAN_PERIOD=1 omg_planner.USE_REAL_POSE_WHEN_CLOSE=True omg_planner.OBJ_POSE_CHANGE_THRES=0.001 omg_planner.REPLAN_WHEN_OBJ_STOP=True dart=True demo_dir=data/demo/s0/train/omg/dense_planning_smooth_0.08_dart start_seed=0 end_seed=20000 step_seed=1000

CUDA_VISIBLE_DEVICES=0 OMG_PLANNER_DIR=/share/haoran/HRI/OMG-Planner-copy RAY_DEDUP_LOGS=0 python -m evaluate use_ray=False setup=s0 split=train env.panda.IK_solver=PyKDL policy=omg_planner omg_planner.FILTER_HAND_COLLISION_GRASPS=True omg_planner.ONLY_NEAREST_GRASP=True omg_planner.ONE_TRIAL=True omg_planner.ACTION_TYPE=cartesian omg_planner.TRAJECTORY_SMOOTHING=True omg_planner.TRAJECTORY_SMOOTHING_STEP_SIZE=0.08 omg_planner.SIMULTANEOUS=True omg_planner.REPLAN_PERIOD=1 omg_planner.USE_REAL_POSE_WHEN_CLOSE=True omg_planner.OBJ_POSE_CHANGE_THRES=0.001 omg_planner.REPLAN_WHEN_OBJ_STOP=True dart=True demo_dir=debug/omg/
```