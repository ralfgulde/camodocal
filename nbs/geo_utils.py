from math3d import Transform
import numpy as np
import quaternion

def pose7d2homTF(pose7d):
    homTF = np.eye(4)
    
    pos = pose7d[0:3]
    q = arr2quat(pose7d[3:7])
    R = quaternion.as_rotation_matrix(q)

    homTF[0:3, 0:3] = R
    homTF[0:3, 3] = np.transpose(pos)
    return homTF

def rotm2quat(rotm):

    q = quaternion.from_rotation_matrix(rotm)
    return quaternion.as_float_array(q)

def urx_trans_to_pose(trans:Transform)->np.ndarray:
    position = trans.pos.array
    rotm = trans.orient.array
    quat = rotm2quat(rotm)
    return np.concatenate((position, quat))

def pose_to_urx_trans(pose:np.ndarray)->Transform:
    homtf = pose7d2homTF(pose)
    return Transform(homtf) 


# Helper functions
def pose7d2rotm(pose):
    R = quaternion.as_rotation_matrix(quaternion.from_float_array(pose[3:7]))
    rotm = np.concatenate((R, np.transpose([pose[0:3]])), axis=1)
    rotm = np.concatenate((rotm, np.array([[0,0,0,1]])), axis=0)
    return rotm

def pose7d2homTF(pose):
    R = quaternion.as_rotation_matrix(quaternion.from_float_array(pose[3:7]))
    rotm = np.concatenate((R, np.transpose([pose[0:3]])), axis=1)
    rotm = np.concatenate((rotm, np.array([[0,0,0,1]])), axis=0)
    return rotm

def homTF2pose7d(homTF):
    pose7d = np.zeros(7)
    pose7d[0:3] = np.transpose(homTF[0:3, 3])

    R = homTF[0:3, 0:3]
    q_ = quaternion.from_rotation_matrix(R)
    q = quaternion.as_float_array(q_)

    pose7d[3:7] = q
    return pose7d


def urx2rotm(urx_i):
    R = urx_i.get_orient()[:]
    t = urx_i.get_pos()[:]
    rotm = np.concatenate((R, np.transpose([t])), axis=1)
    rotm = np.concatenate((rotm, np.array([[0,0,0,1]])), axis=0)
    return rotm

def create_transform(R=np.eye(3),t=np.zeros((3))):
    transform = np.array([[R[0,0], R[0,1], R[0,2], t[0]],
                     [R[1,0], R[1,1], R[1,2], t[1]],
                      [R[2,0], R[2,1], R[2,2], t[2]],
                       [0,0,0,1]])

    #return transform
    return transform
