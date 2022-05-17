import numpy as np
from math import *
from quaternion import *

sampleRate = 100 #hertz


def stateEstimation(correctPreState,F,sampleRate = 100):

    identity = np.eye(7)
    stateTransitionMatrix = identity + F*(1/sampleRate)
    estimatedState = np.dot(stateTransitionMatrix,correctPreState)

    return estimatedState

def covarianceEstimation(correctPreCovariance,F,processNoiseCoovarianceMatrix,sampleRate = 100):

    identity = np.eye(7)
    stateTransitionMatrix = identity + F*(1/sampleRate)
    TRstateTransitionMatrix = np.transpose(stateTransitionMatrix)
    estimatedCovariance = np.dot(stateTransitionMatrix,correctPreCovariance)
    estimatedCovariance = np.dot(estimatedCovariance,TRstateTransitionMatrix)
    estimatedCovariance = estimatedCovariance + processNoiseCoovarianceMatrix
    return estimatedCovariance

def JacobianF(gyromeas,biasgyro,corr_quaternion):
    wx = gyromeas[0]
    wy = gyromeas[1]
    wz = gyromeas[2]
    bx = biasgyro[0]
    by = biasgyro[1]
    bz = biasgyro[2]
    qw = corr_quaternion[0][0]
    qx = corr_quaternion[0][1]
    qy = corr_quaternion[0][2]
    qz = corr_quaternion[0][3]

    jf = np.array([[0,(-(wx-bx)),(-(wy-by)),(-(wz-bz)),qx,qy,qz],
                   [(wx-bx),0,(wz-bz),(-(wy-by)),-qw,qz,-qy],
                   [(wy-by),(-(wz-bz)),0,(wx-bx),-qz,-qw,qx],
                   [(wz-bz),(wy-by),(-(wx-bx)),0,qy,-qx,-qw],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0]])*0.5

    return jf

def JacobianH(gravityvector,magneticfielvector,corr_quaternion,isAcc = True,isMag = True):
    g = gravityvector[2]
    mx = magneticfielvector[0]
    my = magneticfielvector[1]
    mz = magneticfielvector[2]
    qw = corr_quaternion[0][0]
    qx = corr_quaternion[0][1]
    qy = corr_quaternion[0][2]
    qz = corr_quaternion[0][3]

    ha1 = -2*qy*g
    ha2 = 2*qz*g
    ha3 = -2*qw*g
    ha4 = 2*qx*g
    hb1 = 2*(mx*qw+my*qz-mz*qy)
    hb2 = 2*(mx*qx+my*qy+mz*qz)
    hb3 = 2*(-mx*qy+my*qx-mz*qw)
    hb4 = 2*(-mx*qz+my*qw+mz*qx)

    if isAcc and isMag:

        jh = np.array([[ha1,ha2,ha3,ha4,0,0,0],
                       [ha4,-ha3,ha2,-ha1,0,0,0],
                       [-ha3,-ha4,ha1,ha2,0,0,0],
                       [hb1,hb2,hb3,hb4,0,0,0],
                       [hb4,-hb3,hb2,-hb1,0,0,0],
                       [-hb3,-hb4,hb1,hb2,0,0,0]])

    elif isAcc == True and isMag == False:

        jh = np.array([[ha1,ha2,ha3,ha4,0,0,0],
                       [ha4,-ha3,ha2,-ha1,0,0,0],
                       [-ha3,-ha4,ha1,ha2,0,0,0]])
    
    elif isAcc == False and isMag == True:

        jh = np.array([[hb1,hb2,hb3,hb4,0,0,0],
                       [hb4,-hb3,hb2,-hb1,0,0,0],
                       [-hb3,-hb4,hb1,hb2,0,0,0]])

    return jh

def measmodel(acc,mag,Cnb,isAcc = True,isMag = True):

    both = np.array([[Cnb[0][0],Cnb[0][1],Cnb[0][2],0,0,0],
                     [Cnb[1][0],Cnb[1][1],Cnb[1][2],0,0,0],
                     [Cnb[2][0],Cnb[2][1],Cnb[2][2],0,0,0],
                     [0,0,0,Cnb[0][0],Cnb[0][1],Cnb[0][2]],
                     [0,0,0,Cnb[1][0],Cnb[1][1],Cnb[1][2]],
                     [0,0,0,Cnb[2][0],Cnb[2][1],Cnb[2][2]]])

    
    onlyacc = np.array([[Cnb[0][0],Cnb[0][1],Cnb[0][2],0,0,0],
                        [Cnb[1][0],Cnb[1][1],Cnb[1][2],0,0,0],
                        [Cnb[2][0],Cnb[2][1],Cnb[2][2],0,0,0]])

    onlymag = np.array([0,0,0,Cnb[0][0],Cnb[0][1],Cnb[0][2]],
                       [0,0,0,Cnb[1][0],Cnb[1][1],Cnb[1][2]],
                       [0,0,0,Cnb[2][0],Cnb[2][1],Cnb[2][2]])

    if isAcc and isMag :

        measvector = np.array([[acc[0]],
                               [acc[1]],
                               [acc[2]],
                               [mag[0]],
                               [mag[1]],
                               [mag[2]]])

        h = np.dot(both,measvector)

    elif isAcc == True and isMag == False:

        measvector = np.array([[acc[0]],
                               [acc[1]],
                               [acc[2]]])

        h = np.dot(onlyacc,measvector)

    elif isAcc == False and isMag == True:

        measvector = np.array([[mag[0]],
                               [mag[1]],
                               [mag[2]]])

        h = np.dot(onlymag,measvector)

    return h

def processNoiseCovariance(corr_quaternion,gyro_noisevector,sampleRate):

    qw = corr_quaternion[0][0]
    qx = corr_quaternion[0][1]
    qy = corr_quaternion[0][2]
    qz = corr_quaternion[0][3]

    W = np.array([[-qx,-qy,-qz],
                  [qw,-qz,qy],
                  [qz,qw,-qx],
                  [-qy,qx,qw],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0]])*((1/sampleRate)/2)

    Wt = np.transpose(W)

    Q = W*gyro_noisevector
    Q = np.dot(Q,Wt)

    return Q

def calculateKalmanGain(estCovariance,observationMatrix,measNoiseVector,isAcc = True,isMag = True):

    TRobservationMatrix = np.transpose(observationMatrix)
    S = np.dot(observationMatrix,estCovariance)
    S = np.dot(S,TRobservationMatrix)
    noisevector = np.array([[measNoiseVector[0]],
                            [measNoiseVector[0]],
                            [measNoiseVector[0]],
                            [measNoiseVector[1]],
                            [measNoiseVector[1]],
                            [measNoiseVector[1]]])
    
    ident = np.eye(6)

    R = np.dot(noisevector,ident)

    S = S+R
    S = np.linalg.inv(S)

    K = np.dot(estCovariance,TRobservationMatrix)
    K = np.dot(K,S)

    return K

def observationVector(measVector,measModel):
    v = measVector - measModel
    return v

def correctState(estimatedState,kalmanGain,observVector):
    corr_State = estimatedState + np.dot(kalmanGain,observVector)

    return corr_State

def correctCovariance(estimatedCovarianceMatrix,kalmanGain,observationMatrix):
    ident = np.eye(7)
    corr_Cov = ident - np.dot(kalmanGain,observationMatrix)
    corr_Cov = np.dot(corr_Cov,estimatedCovarianceMatrix)

    return correctCovariance


