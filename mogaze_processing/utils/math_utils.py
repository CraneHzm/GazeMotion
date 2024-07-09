import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def quaternion_matrix(quaternion):
    """Return rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    # We assum the ROS convention (x, y, z, w)
    quaternion_tmp = np.array([0.0] * 4)
    quaternion_tmp[1] = quaternion[0]  # x
    quaternion_tmp[2] = quaternion[1]  # y
    quaternion_tmp[3] = quaternion[2]  # z
    quaternion_tmp[0] = quaternion[3]  # w
    q = np.array(quaternion_tmp, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])
        
        
def euler2xyz(pose_euler):
    """
    Convert the human pose in MoGaze dataset from euler representation to xyz representation.
    """     
    # names of all the 21 joints
    joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'linnerShoulder',  
                         'lShoulder', 'lElbow', 'lWrist', 'rinnerShoulder', 'rShoulder', 
                         'rElbow', 'rWrist', 'lHip', 'lKnee', 'lAnkle', 
                         'lToe', 'rHip', 'rKnee', 'rAnkle', 'rToe']                                                                                   
    joint_ids = {name: idx for idx, name in enumerate(joint_names)}
                         
    # translation of the 20 joints (excluding base), obtained from the mogaze dataset
    joint_trans = np.array([[0, 0, 0.074],
                            [0, 0, 0.201],
                            [0, 0, 0.234], 
                            [0, -0.018, 0.140],
                            [0.036, 0, 0.183], 
                            [0.153, 0, 0],
                            [0.243, 0, 0], 
                            [0.267, -0.002, 0],
                            [-0.036, 0, 0.183], 
                            [-0.153, 0, 0],
                            [-0.243, 0, 0], 
                            [-0.267, -0.002, 0],
                            [0.090, 0, 0], 
                            [0, 0, -0.383],
                            [0, 0, -0.354], 
                            [0, -0.135, -0.059],
                            [-0.090, 0, 0], 
                            [0, 0, -0.383],
                            [0, 0, -0.354], 
                            [0, -0.135, -0.059]])
         

    # parent of every joint
    joint_parent_names = {
                                  # root
                                  'base':           'base',
                                  'pelvis':         'base',                               
                                  'torso':          'pelvis', 
                                  'neck':           'torso', 
                                  'head':           'neck', 
                                  'linnerShoulder': 'torso',
                                  'lShoulder':      'linnerShoulder', 
                                  'lElbow':         'lShoulder', 
                                  'lWrist':         'lElbow', 
                                  'rinnerShoulder': 'torso', 
                                  'rShoulder':      'rinnerShoulder', 
                                  'rElbow':         'rShoulder', 
                                  'rWrist':         'rElbow', 
                                  'lHip':           'base', 
                                  'lKnee':          'lHip', 
                                  'lAnkle':         'lKnee', 
                                  'lToe':           'lAnkle', 
                                  'rHip':           'base', 
                                  'rKnee':          'rHip', 
                                  'rAnkle':         'rKnee', 
                                  'rToe':           'rAnkle'}                               
    # id of joint parent
    joint_parent_ids = [joint_ids[joint_parent_names[child_name]] for child_name in joint_names]
        
    # forward kinematics
    joint_number = len(joint_names)
    pose_xyz = np.zeros((pose_euler.shape[0], joint_number*3))
    for i in range(pose_euler.shape[0]):        
        # xyz position in the world coordinate system
        pose_xyz_tmp = np.zeros((joint_number, 3))
        pose_xyz_tmp[0] = [pose_euler[i][0], pose_euler[i][1], pose_euler[i][2]]                        
        pose_rot_mat = np.zeros((joint_number, 3, 3))
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            pose_rot_mat[j] = R.from_euler('XYZ', rot).as_matrix()
                          
        for j in range(1, joint_number):
            pose_rot_mat_parent = pose_rot_mat[joint_parent_ids[j]]
            pose_xyz_tmp[j] = np.matmul(pose_rot_mat_parent, joint_trans[j-1]) + pose_xyz_tmp[joint_parent_ids[j]]
            pose_rot_mat[j] = np.matmul(pose_rot_mat_parent, pose_rot_mat[j])
        
        pose_xyz[i] = pose_xyz_tmp.reshape(joint_number*3)
    return pose_xyz


def euler2xyz_head(pose_euler):
    """
    Calculate head direction from human pose
    """     
    # names of the joints
    joint_names = ['base', 'pelvis', 'torso', 'neck', 'head']                                                                                   
    # forward kinematics
    joint_number = len(joint_names)        
    data_size = pose_euler.shape[0]
    head_direction = np.zeros((data_size, 3))
    
    for i in range(data_size):
        pose_rot = R.identity().as_matrix()    
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            rot_mat = R.from_euler('XYZ', rot).as_matrix()
            pose_rot = np.matmul(pose_rot, rot_mat)          
            
        head_direction[i] = [0, -1, 0]
        head_direction[i] = np.matmul(pose_rot, head_direction[i])
        
    return head_direction