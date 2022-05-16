import numpy as np
from math import *

def euler2quaternion(roll,pitch,yaw):
    """
        roll: radians
        pitch: radians
        yaw: radians
    """
    r = roll/2
    p = pitch/2
    y = yaw/2

    qw = cos(r)*cos(p)*cos(y)+sin(r)*sin(p)*sin(y)
    qx = sin(r)*cos(p)*cos(y)-cos(r)*sin(p)*sin(y)
    qy = cos(r)*sin(p)*cos(y)+sin(r)*cos(p)*sin(y)
    qz = cos(r)*cos(p)*sin(y)-sin(r)*sin(p)*cos(y)

    quaternion = np.array([[qw],
                           [qx],
                           [qy],
                           [qz]])

    return quaternion,qw,qx,qy,qz


def quaternion2euler(qw,qx,qy,qz):
    roll = atan2((2*(qw*qx+qy*qz)),(1-2*qx*qx-2*qy*qy))
    pitch = asin(2*(qw*qy-qx-qz))
    yaw = atan2((2*(qw*qz+qx*qy)),(1-2*qy*qy-qz*qz))

    return roll,pitch,yaw


def CTMquaternion(qw,qx,qy,qz):

    C = np.array([[(qw*qw+qx*qx-qy*qy-qz*qz),(2*(qx*qy+qw*qz)),(2*(qx*qz-qy*qw))],
                  [(2*(qx*qy-qw*qz)),(qw*qw-qx*qx+qy*qy-qz*qz),(2*(qy*qz+qw*qx))],
                  [(2*(qx*qz+qy*qw)),(2*(qy*qz+qw*qx)),(qw*qw-qx*qx-qy*qy+qz*qz)]])

    Ct = C.transpose()

    return C


    
