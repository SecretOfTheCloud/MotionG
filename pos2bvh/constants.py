# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import numpy as np
from pos2bvh.utils import calculate_diatance_3d

# parameter
ORIGIN = point = np.array([0, 0, 0, 1])
DEBUG = 0
IK_DEBUG = 0
FK_DEBUG = 0
TOLERANCE = 1.0
MAX_ITERATION = 200
FILTER_WINDOW = 5

# data
FRAME_PER_SECOND = 60.0
DEFAULT_ROTATION_ORDER = ['Zrotation', 'Xrotation', 'Yrotation']
INVERSE_ROTATION_ORDER = [DEFAULT_ROTATION_ORDER[2], DEFAULT_ROTATION_ORDER[1], DEFAULT_ROTATION_ORDER[0]]

SPINE_INDEX = 11
LEFT_HIP_INDEX = 1
RIGHT_HIP_INDEX = 6

U = np.array([0, 1, 0])
D = np.array([0, -1, 0])
L = np.array([1, 0, 0])
R = np.array([-1, 0, 0])
INITIAL_DIRECTION = [
    np.array([0, 0, 0]), # 'Hips',
    L, D, D, L, L,       # 'LHipjoint',	    'LeftUpLeg',    	'LeftLeg',	        'LeftFoot',	    'LeftToeBase',
    R, D, D, R, R,       # 'RHipjoint',	    'RightUpLeg',   	'RightLeg',	        'RightFoot',	'RightToeBase',
    U, U, U, U, U, U,    # 'LowerBack',	    'Spine',            'Spine1',	        'Neck',         'Neck1',            'Head',
    L, L, L, L, L, L, L, # 'LeftShoulder',	 'LeftArm',	        'LeftForeArm',	    'LeftHand',	    'LeftFingerBase',	'LeftHandIndex1',   'LThumb',
    R, R, R, R, R, R, R  # 'RightShoulder',	'RightArm',	        'RightForeArm',	    'RightHand',	'RightFingerBase',	'RightHandIndex1',  'RThumb'
]

JOINT_NAME = [
    # 0
    'Hips',
    # 1                 2                   3                   4               5
    'LHipjoint',	    'LeftUpLeg',    	'LeftLeg',	        'LeftFoot',	    'LeftToeBase',
    # 6                 7                   8                   9               10
    'RHipjoint',	    'RightUpLeg',   	'RightLeg',	        'RightFoot',	'RightToeBase',
    # 11                12                  *13                 14              15              16
    'LowerBack',	    'Spine',            'Spine1',	        'Neck',         'Neck1',        'Head',
    # 17                18                  19                  *20             21                  22
    'LeftShoulder',	    'LeftArm',	        'LeftForeArm',	    'LeftHand',	    'LeftFingerBase',	'LeftHandIndex1',
                                                                                # 23
                                                                                'LThumb',
    # 24                25             +     26                  *27             28                  29
    'RightShoulder',	'RightArm',	        'RightForeArm',	    'RightHand',	'RightFingerBase',	'RightHandIndex1',
                                                                                # 30
                                                                                'RThumb'
]
PARENT_INDEX = [
    -1,
    0,			    1,		    2,		    3,		    4,
    0,			    6,		    7,		    8,		    9,
    0,			    11,			12,		    13,		    14,         15,
    13,			    17,			18,			19,		    20,		    21,			20,
    13,			    24,			25,			26,		    27,		    28,			27
]


SKELETON_OFFSET = np.array([
    # root
    [0., 0., 0.],
    # ldown
    [0.000000, 0.000000, 0.000000],
    [1.363060, -1.794630, 0.839290],
    [2.448110, -6.726130, 0.000000],
    [2.562200, -7.039590, 0.000000],
    [0.157640, -0.433110, 2.322550],
    # rdown
    [0.000000, 0.000000, 0.000000],
    [-1.305520, -1.794630, 0.839290],
    [-2.542530, -6.985550, 0.000000],
    [-2.568260, -7.056230, 0.000000],
    [-0.164730, -0.452590, 2.363150],
    # spine
    [0.000000, 0.000000, 0.000000],
    [0.028270, 2.035590, -0.193380],
    [0.056720, 2.048850, -0.042750],
    [0.000000, 0.000000, 0.000000],
    [-0.054170, 1.746240, 0.172020],
    [0.104070, 1.761360, -0.123970],
    # lup
    [0.000000, 0.000000, 0.000000],
    [3.362410, 1.200890, -0.311210],
    [4.983000, -0.000000, -0.000000],
    [3.483560, -0.000000, -0.000000],
    [0.000000, 0.000000, 0.000000],
    [0.715260, -0.000000, -0.000000],
    [0.000000, 0.000000, 0.000000],
    # rup
    [0.000000, 0.000000, 0.000000],
    [-3.136600, 1.374050, -0.404650],
    [-5.241900, -0.000000, -0.000000],
    [-3.444170, -0.000000, -0.000000],
    [0.000000, 0.000000, 0.000000],
    [-0.622530, -0.000000, -0.000000],
    [0.000000, 0.000000, 0.000000]
])
INITIAL_OFFSET = dict()
for ind in range(len(JOINT_NAME)):
    INITIAL_OFFSET[JOINT_NAME[ind]] = SKELETON_OFFSET[ind]


def set_initial_offset(position):
    for i, joint in enumerate(JOINT_NAME):
        if i == 0:
            INITIAL_OFFSET[joint] = np.array([0, 0, 0])
        else:
            direction = INITIAL_DIRECTION[i]
            parent = PARENT_INDEX[i]
            dis = calculate_diatance_3d(position[i], position[parent])
            direction_dis = np.linalg.norm(direction, ord=2, axis=0)
            INITIAL_OFFSET[joint] = direction / direction_dis * dis

'''
# the information of Cyprus bvh file
SPINE_INDEX = 9
LEFT_HIP_INDEX = 1
RIGHT_HIP_INDEX = 5
INITIAL_OFFSET = {
    'Hips': np.array([0, 0, 0]),
    'LeftUpLeg': np.array([8.64259, 1.95398, -2.38835]),  # 1
    'LeftLeg': np.array([43.7702, 0, 0]),
    'LeftFoot': np.array([40.6238, 0, 0]),
    'LeftToeBase': np.array([12.7242, 0, 0]),
    'RightUpLeg': np.array([-8.2249, 1.95398, -2.59336]),  # 5
    'RightLeg': np.array([43.5653, 0, 0]),
    'RightFoot': np.array([40.6238, 0, 0]),
    'RightToeBase': np.array([12.7242, 0, 0]),
    'Spine': np.array([0.165834, 0.87714, 6.61349]),  # 9
    'Spine1': np.array([7.38963, 1.71661e-005, 0]),
    'Spine2': np.array([11.9445, 1.14441e-005, 0]),
    'Spine3': np.array([14.7254, 4.19617e-005, 0]),
    'Spine4': np.array([10.9679, 0, 0]),
    'RightShoulder': np.array([2.46616, -1.6653, 3.69651]),  # 14
    'RightArm': np.array([13.0012, 0, 0]),
    'RightForeArm': np.array([28.7848, 1.52588e-005, 0]),
    'RightHand': np.array([26.5688, 1.52588e-005, 0]),
    'RightFinger': np.array([8.03675, -0.886581, 0.727868]),
    'LeftShoulder': np.array([2.38034, -1.29275, -3.7523]),  # 19
    'LeftArm': np.array([16.3182, 6.10352e-005, 0]),
    'LeftForeArm': np.array([27.0901, 0, 0]),
    'LeftHand': np.array([24.5451, 1.52588e-005, 0]),
    'LeftFinger': np.array([9.21802, 0.665817, 0.545532]),
    'Neck': np.array([5.0226, -0.0266342, 0]),  # 24
    'Head': np.array([6.69364, 2.86102e-005, 0])  # 25
}
'''
