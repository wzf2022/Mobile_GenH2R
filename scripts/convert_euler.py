from scipy.spatial.transform import Rotation as R

# 给定的四元数
qx = -0.06840926606432406
qy = 0.11045547394688866
qz = 0.8455017178449404
qw = -0.5179252897020606
r = R.from_quat([qx, qy, qz, qw])

r = R.from_quat([0.7022425551467179, -0.15135464549324493, 0.6785433643278053, 0.1533820972451689])

# 创建旋转对象
# r = R.from_quat([qx, qy, qz, qw])

# 将四元数转换为欧拉角（单位为弧度）
euler_angles = r.as_euler('xyz', degrees=False)

# 打印欧拉角
print("Roll (x):", euler_angles[0])
print("Pitch (y):", euler_angles[1])
print("Yaw (z):", euler_angles[2])