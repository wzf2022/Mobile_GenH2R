import numpy as np
from scipy.spatial.transform import Rotation as R


joint_name_to_angle = {
    "leg_joint1": 0.497,
    "leg_joint2": 1.571,
    "leg_joint3": 1.178,
    "leg_joint4": 0.0, 
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "right_arm_joint1": -1.571,
    "right_arm_joint2": -1.571,
    "right_arm_joint3": 0.0,
    "right_arm_joint4": 0.0,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.0,
    "right_arm_joint7": 0.0,
}

joint_init_rpy = {
    "leg_joint1": [-3.141592653589793, 0, 2.756573021],
    "leg_joint2": [-3.141592653589793, 0, 2.756573021],
    "leg_joint3": [-3.141592653589793, 0, -3.141592653589793],
    "leg_joint4": [1.5707963267948966, 0, -3.141592653589793],
    "head_joint1": [0, 0, 0],
    "head_joint2": [-1.5707963267948966, 0, 0],
    "right_arm_joint1": [0, 0, 0],
    "right_arm_joint2": [-1.5707963267948966, 0, 3.141592653589793],
    "right_arm_joint3": [1.5707963267948966, 1.5707963267948966, 0],
    "right_arm_joint4": [-1.5707963267948966, 0, 3.141592653589793],
    "right_arm_joint5": [1.5707963267948966, 0, 0],
    "right_arm_joint6": [-1.5707963267948966, 0, 3.141592653589793],
    "right_arm_joint7": [-1.5707963267948966, 3.141592653589793, 3.141592653589793]
}


for joint_name, DoF_angle in joint_name_to_angle.items():
    initial_rpy = np.array(joint_init_rpy[joint_name])


    # the rotation around fixed axis: first roll around x, then pitch around y and finally yaw around z. 
    initial_rotation = R.from_euler('xyz', initial_rpy).as_matrix()

    # rotation around self z axis
    joint_rotation = R.from_euler('z', DoF_angle).as_matrix()

    combined_rotation =  initial_rotation @ joint_rotation 

    final_rpy = R.from_matrix(combined_rotation).as_euler('xyz')

    print(joint_name, final_rpy[0], final_rpy[1], final_rpy[2])

# leg_joint1 -3.141592653589793 2.220446049250313e-16 2.259573021
# leg_joint2 3.141592653589793 2.220446049250313e-16 1.185573021
# leg_joint3 -3.141592653589793 2.220446049250313e-16 1.9635926535897932
# leg_joint4 1.5707963267948963 0.0 3.141592653589793
# head_joint1 0.0 0.0 0.0
# head_joint2 -1.5707963267948963 0.0 0.0
# right_arm_joint1 0.0 0.0 -1.571
# right_arm_joint2 1.5707963267935339 -1.5705926535881751 1.3627504802821158e-12

# # 初始rpy值
# initial_rpy = np.array([-3.141592653589793, 0, 2.756573021])

# initial_rpy = np.random.rand(3)

# # 关节旋转角度
# joint_angle = 0.497

# joint_angle = np.random.random_sample()

# # 初始旋转矩阵
# initial_rotation = R.from_euler('xyz', initial_rpy).as_matrix()

# # 关节旋转矩阵（绕z轴旋转）
# joint_rotation = R.from_euler('z', joint_angle).as_matrix()

# # 组合旋转矩阵
# combined_rotation =  initial_rotation @ joint_rotation 

# # 将组合后的旋转矩阵转换回rpy
# final_rpy = R.from_matrix(combined_rotation).as_euler('xyz')

# print("Initial rpy ", initial_rpy, "Initial angle", joint_angle)
# print("Final rpy:", final_rpy)
# print("error", initial_rpy[2] - final_rpy[2] + joint_angle)