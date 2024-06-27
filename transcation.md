CUDA_VISIBLE_DEVICES=0 python -m evaluate setup=m0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=0.001 policy=chomp chomp.wait_time=0. chomp.hand_collision_free_mask_dir=env/data/mobile_hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.use_endpoints=False chomp.trajectory_smoothing=False demo_dir=log/m0/static record_third_person_video=True overwrite_demo=True env.status_checker.verbose=True  num_runners=1 env.visualize=True record_ego_video=True use_ray=False chomp.replan_period=0.01 chomp.know_destination=False env.set_human_hand_obj_last_frame=True chomp.planner.fix_end=True show_target_grasp=True scene_ids=[10014]



CUDA_VISIBLE_DEVICES=0 python -m evaluate setup=m0 split=train start_object_idx=0 end_object_idx=32 start_traj_idx=0 end_traj_idx=1 env.stop_moving_time=0.001 policy=chomp chomp.wait_time=0. chomp.hand_collision_free_mask_dir=env/data/mobile_hand_collision_free_mask/augment_True_threshold_0_use_bbox chomp.use_endpoints=False chomp.trajectory_smoothing=False demo_dir=log/m0/static record_third_person_video=True overwrite_demo=True num_runners=1 env.visualize=True record_ego_video=True use_ray=False chomp.replan_period=0.01 chomp.know_destination=False env.set_human_hand_obj_last_frame=True chomp.planner.fix_end=True show_target_grasp=True


python -m  env.galbot step_time=0.001
1