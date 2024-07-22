##  demonstration generation
CUDA_VISIBLE_DEVICES=0 python -m evaluate setup=m0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=0.001 policy=chomp chomp.wait_time=0. chomp.hand_collision_free_mask_dir=env/data/mobile_hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.use_endpoints=False chomp.trajectory_smoothing=False demo_dir=log/m0/static record_third_person_video=True overwrite_demo=True env.status_checker.verbose=True  num_runners=1 env.visualize=True record_ego_video=True use_ray=False chomp.replan_period=0.01 chomp.know_destination=False env.set_human_hand_obj_last_frame=True chomp.planner.fix_end=True show_target_grasp=True env.show_trajectory=True scene_ids=[10014] chomp.action_type=ego_cartesian_robot_body

CUDA_VISIBLE_DEVICES=3,5 python -m evaluate setup=m0 split=train start_object_idx=10000 end_object_idx=12149 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=0.001 policy=chomp chomp.wait_time=0. chomp.hand_collision_free_mask_dir=env/data/mobile_hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.use_endpoints=False chomp.trajectory_smoothing=False demo_dir=log/m0/static record_third_person_video=True overwrite_demo=True env.status_checker.verbose=False  num_runners=20 env.visualize=False record_ego_video=True use_ray=True chomp.replan_period=0.01 chomp.know_destination=False env.set_human_hand_obj_last_frame=True chomp.planner.fix_end=True show_target_grasp=True env.show_trajectory=False demo_failure_time=3 chomp.action_type=ego_cartesian_robot_body 

CUDA_VISIBLE_DEVICES=0 python -m evaluate setup=m0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=0.001 policy=chomp chomp.wait_time=0. chomp.hand_collision_free_mask_dir=env/data/mobile_hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.use_endpoints=False chomp.trajectory_smoothing=False demo_dir=log/m0/static record_third_person_video=True overwrite_demo=True num_runners=1 env.visualize=True record_ego_video=True use_ray=False chomp.replan_period=0.01 chomp.know_destination=False env.set_human_hand_obj_last_frame=True chomp.planner.fix_end=True show_target_grasp=True


python -m  env.galbot step_time=0.001


code.interact(local=dict(globals(), **locals()))


CUDA_VISIBLE_DEVICES=0 python -m evaluate setup=m0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=0.001 policy=chomp chomp.wait_time=0. chomp.hand_collision_free_mask_dir=env/data/mobile_hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.use_endpoints=False chomp.trajectory_smoothing=False demo_dir=log/m0/static record_third_person_video=True overwrite_demo=True env.status_checker.verbose=True  num_runners=1 env.visualize=True record_ego_video=True use_ray=False chomp.replan_period=0.01 chomp.know_destination=False env.set_human_hand_obj_last_frame=True chomp.planner.fix_end=True show_target_grasp=True env.show_trajectory=True scene_ids=[20000]



### for new data:
python -m script.add_new_scene_data
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m env.tools.mobile_generate_hand_collision_free_mask




## policy learning

# train
CUDA_VISIBLE_DEVICES=5 python -m train.train_imitation train.data.demo_dir log/m0/static train.data.seed 0 train.run_dir data/models/tmp
# evaluate
CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate evaluate.setup s0 evaluate.split test evaluate.use_ray True evaluate.num_runners 32 policy.name pointnet2 policy.pointnet2.pretrained_dir data/models/s0/cartesian_simultaneous_dart/0 policy.pointnet2.pretrained_suffix iter_80000

CUDA_VISIBLE_DEVICES=4 python -m train.train_imitation data.demo_dir=log/m0/static seed=0 run_dir=data/models/m0/static/0 verbose=True data.processor.pc.flow_frame_num=3 model.obj_pose_pred_frame_num=0 model.obj_pose_pred_coff=0.5

CUDA_VISIBLE_DEVICES=4 python -m train.train_imitation data.demo_dir=log/m0/static seed=0 run_dir=data/models/m0/static/0 verbose=True data.processor.pc.flow_frame_num=3 model.obj_pose_pred_frame_num=0 model.obj_pose_pred_coff=0.5 wandb=True data.cache_all_data=True
