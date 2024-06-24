import os
import numpy as np
import PyKDL
import pickle
import ipdb
import code

from .transform import mat_to_pos_ros_quat
from .robotPose.urdf_parser_py.urdf import URDF
from .robotPose.kdl_parser import kdl_tree_from_urdf_model

def joint_list_to_kdl(q):
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = PyKDL.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl

def deg2rad(deg):
    if type(deg) is list:
        return [x / 180.0 * np.pi for x in deg]
    return deg / 180.0 * np.pi

def rad2deg(rad):
    if type(rad) is list:
        return [x/np.pi*180 for x in rad]
    return rad/np.pi*180

def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX

def rotz(joints):
    M = np.tile(np.eye(4), [joints.shape[0], 1, 1])
    M[..., 0, 0] = np.cos(joints)
    M[..., 0, 1] = -np.sin(joints)
    M[..., 1, 0] = np.sin(joints)
    M[..., 1, 1] = np.cos(joints)
    return M

def DH(pose, joints, offset=0):
    rotx = rotX(offset)
    M = rotz(joints)
    pose = np.matmul(pose, np.matmul(M, rotx))
    return pose

def wrap_value(value):
    if value.shape[0] <= 7:
        return rad2deg(value)
    value_new = np.zeros(value.shape[0] + 1)
    value_new[:7] = rad2deg(value[:7])
    value_new[8:] = rad2deg(value[7:])
    return value_new

class RobotKinematics:
    def __init__(self, IK_solver_max_iter: int = 100, IK_solver_eps: float = 1e-6):
        self.IK_solver_max_iter = IK_solver_max_iter
        self.IK_solver_eps = IK_solver_eps
        
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "assets", "panda_kdl", "robot_p3.pkl"), "rb") as f:
            robot_info = pickle.load(f)

        self._pose_0 = robot_info["_pose_0"]
        self.finger_pose = self._pose_0[-2].copy()

        self._joint_origin = robot_info["_joint_axis"]
        self._tip2joint = robot_info["_tip2joint"]
        self._joint_axis = robot_info["_joint_axis"]
        self._joint_limits = robot_info["_joint_limits"]
        self._joint2tips = robot_info["_joint2tips"]
        self._joint_name = robot_info["_joint_name"]
        self.center_offset = np.array(robot_info["center_offset"])
        self._link_names = robot_info["_link_names"]

        self._base_link, self._tip_link = "panda_link0", "panda_hand"
        self._num_jnts = 7

        self.soft_joint_limit_padding = 0.2

        mins_kdl = joint_list_to_kdl(np.array([self._joint_limits[n][0] + self.soft_joint_limit_padding for n in self._joint_name[:-3]]))
        maxs_kdl = joint_list_to_kdl(np.array([self._joint_limits[n][1] - self.soft_joint_limit_padding for n in self._joint_name[:-3]]))

        self._robot = URDF.from_xml_string(open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "assets", "panda_kdl", "panda_arm_hand.urdf"),"r").read())
        self._kdl_tree, _ = kdl_tree_from_urdf_model(self._robot)
        self._arm_chain = self._kdl_tree.getChain(self._base_link, self._tip_link)
        # ipdb.set_trace()
        self._fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)
        self._ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self._arm_chain)
        self._ik_p_kdl = PyKDL.ChainIkSolverPos_NR_JL(self._arm_chain, mins_kdl, maxs_kdl, self._fk_p_kdl, self._ik_v_kdl, maxiter=self.IK_solver_max_iter, eps=self.IK_solver_eps)

    def forward_kinematics_parallel(
        self,
        joint_values=None,
        base_link="right_arm_mount",
        base_pose=None,
        offset=True,
        return_joint_info=False,
    ):
        """
        Input joint angles in degrees, output poses list in robot coordinates
        For a batch of joint requests
        """

        n, q = joint_values.shape
        initial_pose = np.array(self._pose_0)
        joints = deg2rad(joint_values)
        pose = np.tile(initial_pose, [n, 1, 1, 1])
        output_pose = np.zeros_like(pose)
        offsets = [0, -np.pi, np.pi, np.pi, -np.pi, np.pi, np.pi]

        cur_pose = base_pose
        if cur_pose is None:
            cur_pose = np.eye(4)
        cur_pose = cur_pose[None, ...]

        for i in range(7):
            b = DH(pose[:, i], joints[:, i], offsets[i])
            if i > 0:
                b[..., [1, 2]] *= -1

            cur_pose = np.matmul(cur_pose, b)
            output_pose[:, i] = cur_pose.copy()

        left_finger_pose = np.tile(initial_pose[8], [n, 1, 1])
        left_finger_pose[:, 1, 3] += joints[:, -2]
        right_finger_pose = np.tile(initial_pose[9], [n, 1, 1])
        right_finger_pose[:, 1, 3] -= joints[:, -1]

        output_pose[:, 7] = np.matmul(output_pose[:, 6], initial_pose[7])
        output_pose[:, 8] = np.matmul(output_pose[:, 7], left_finger_pose)
        output_pose[:, 9] = np.matmul(output_pose[:, 7], right_finger_pose)

        if return_joint_info:
            pose2axis = np.array(self._joint_axis)
            pose2origin = np.array(self._joint_origin)
            joint_pose = np.matmul(
                output_pose, self._tip2joint
            )  # pose_joint.dot(poses[idx])
            joint_axis = np.matmul(joint_pose[..., :3, :3], pose2axis[..., None])[
                ..., 0
            ]
            joint_origin = (
                np.matmul(joint_pose[..., :3, :3], pose2origin[..., None])[..., 0]
                + joint_pose[..., :3, 3]
            )

        if offset:  # for on
            output_pose = np.matmul(output_pose, self.center_offset)

        if return_joint_info:
            return output_pose, joint_origin, joint_axis
        else:
            return output_pose

    def inverse_kinematics(self, position, orientation, seed=None):
        """
        Inverse kinematics in radians
        """

        pos = PyKDL.Vector(position[0], position[1], position[2])
        rot = PyKDL.Rotation.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        # Populate seed with current angles if not provided
        seed_array = PyKDL.JntArray(self._num_jnts)
        if seed is None:
            seed = np.zeros(7)
            seed = seed[: self._num_jnts]
            seed = deg2rad(seed)
        seed_array.resize(len(seed))
        for idx in range(seed.shape[0]):
            seed_array[idx] = seed[idx]

        # Make IK Call
        goal_pose = PyKDL.Frame(rot, pos)
        result_angles = PyKDL.JntArray(self._num_jnts)

        # ipdb.set_trace()
        info = self._ik_p_kdl.CartToJnt(seed_array, goal_pose, result_angles) # E_NOERROR = 0, E_NOT_UP_TO_DATE = -3, E_SIZE_MISMATCH = -4, E_MAX_ITERATIONS_EXCEEDED = -5, E_IKSOLVERVEL_FAILED = -100, E_FKSOLVERPOS_FAILED = -101 
        result = np.array(list(result_angles))
        return result, info
    
    def joint_to_cartesian(self, joints):
        """ Convert joint space position to task space position by fk """
        return self.forward_kinematics_parallel(wrap_value(joints)[None], offset=False)[0][-3]

    def cartesian_to_joint(self, base_to_ee, seed):
        pos, orn = mat_to_pos_ros_quat(base_to_ee)
        target_joints = self.inverse_kinematics(pos, orn, seed=seed)
        return target_joints

def debug():
    np.set_printoptions(suppress=True) # no scientific notation
    robot = RobotKinematics(IK_solver_max_iter=100)
    pos = np.array([ 0.1439894 , -0.00910749,  0.71072687])
    ros_quat = np.array([ 0.96438653,  0.03465594,  0.2612568 , -0.02241564])
    seed = np.array([ 0.   , -1.285,  0.   , -2.356,  0.   ,  1.571,  0.785])
    ik_joint_values, info = robot.inverse_kinematics(pos, ros_quat, seed)
    pose = robot.joint_to_cartesian(ik_joint_values)
    home_joint_values = np.zeros(7)
    home_pose = robot.joint_to_cartesian(home_joint_values)
    joint_values = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [.1, 0., 0., 0., 0., 0., 0.],
        [0., .1, 0., 0., 0., 0., 0.],
        [0., 0., .1, 0., 0., 0., 0.],
        [0., 0., 0., .1, 0., 0., 0.],
        [0., 0., 0., 0., .1, 0., 0.],
        [0., 0., 0., 0., 0., .1, 0.],
        [0., 0., 0., 0., 0., 0., .1],
        [.1, .2, .3, .4, .5, .6, .7],
    ])
    # for joint_value in joint_values:
    #     print(f"joint_value {joint_value}, base_to_ee\n{robot.joint_to_cartesian(joint_value)}")
    joint_value = np.append(joint_values[-1], [0.04, 0.04])
    output_pose = robot.forward_kinematics_parallel(wrap_value(joint_value)[None], offset=False)[0] # (10, 4, 4)
    code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    debug()

"""
python -m env.utils.robot_kinematics_old
"""