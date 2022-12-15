# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import argparse
import numpy as np
from pos2bvh.forward_kinematics import ForwardKinematics
from pos2bvh.constants import DEBUG, PARENT_INDEX, JOINT_NAME, FILTER_WINDOW, set_initial_offset
from pos2bvh.inverse_kinematics import InverseKinematics
from pos2bvh.filter_rotations import DataFilter
from pos2bvh.bvh_writer import BVHWriter
import os
import re


def GaussianFilter_1D(array, K_size, times):
    if times == 0:
        return array
    w, c = array.shape
    # 高斯滤波
    sigma = 5

    # 零填充
    pad = K_size // 2
    out = np.zeros((w + 2 * pad, c), dtype=float)
    out[pad:pad + w] = array.copy().astype(float)
    # out[:pad] = array[0]
    # out[pad+w:] = array[-1]

    # 定义滤波核
    K = np.zeros((K_size), dtype=float)

    for x in range(-pad, -pad + K_size):
        K[x + pad] = np.exp(-(x ** 2) / (2 * (sigma ** 2)))
    K /= (sigma * np.sqrt(2 * np.pi))
    K /= K.sum()

    # 卷积的过程
    tmp = out.copy()
    for x in range(w):
        for ci in range(c):
            out[pad + x, ci] = np.sum(K * tmp[x:x + K_size, ci])

    # out = out[pad:pad+w].astype(np.uint8)
    out = out[pad:pad + w]

    return GaussianFilter_1D(out, K_size, times - 1)


def read_data(position_file, joint_file):
    read_position = open(position_file, "r")
    position_data = []
    for line in read_position.readlines():
        data = line.replace('\n', '').replace('\r', '')
        strs = data.split(' ')
        line_data = []
        for i in range(int(len(strs) / 3)):
            line_data.append(np.array([float(strs[i * 3]), float(strs[i * 3 + 1]), float(strs[i * 3 + 2])]))
        position_data.append(line_data)
    read_position.close()
    parent_index = PARENT_INDEX
    joint_data = JOINT_NAME
    # set_initial_offset(position_data[0]) # set to a fixed skeleton in constants.py
    return position_data, joint_data, parent_index


def test_FK(joints, parent_index, bvh_data=None):
    if bvh_data is None:
        f = open("data/rotation_6_test.txt", "r")
        bvh_data = f.readlines()[0].replace('\n', '').split(' ')
    rotation_data = []
    # channel_6 = True
    channel_6 = False
    if channel_6:
        for i in range(int(len(bvh_data) / 6)):
            if (38 <= i <= 51) or (19 <= i <= 32):
                continue
            if i == 0:
                rotation_data.append(
                    np.array([float(bvh_data[i * 6]), float(bvh_data[i * 6 + 1]), float(bvh_data[i * 6 + 2])]))
            rotation_data.append(
                np.array([float(bvh_data[i * 6 + 3]), float(bvh_data[i * 6 + 4]), float(bvh_data[i * 6 + 5])]))
    else:
        for i in range(int((len(bvh_data) - 3) / 3)):
            if i == 0:
                rotation_data.append(
                    np.array([float(bvh_data[i * 3]), float(bvh_data[i * 3 + 1]), float(bvh_data[i * 3 + 2])]))
            rotation_data.append(
                np.array([float(bvh_data[i * 3 + 3]), float(bvh_data[i * 3 + 4]), float(bvh_data[i * 3 + 5])]))
    print(rotation_data)

    fk = ForwardKinematics(joints)
    fk.set_root_matrix(rotation_data[0], rotation_data[1], joints[0])

    for i, name in enumerate(joints):
        if i == 0:
            p = fk.get_node_position(i, rotation_data[i + 1])
        else:
            if i == 1:
                p = fk.get_node_position(i, rotation_data[i + 1], parent_index[i])
            else:
                p = fk.get_node_position(i, rotation_data[i + 1], parent_index[i])
        if DEBUG:
            print("node:", name, i)
            # print("parent:", parent_index[i], joints[parent_index[i]])
            print("pos:", p)


def check_data(position, joints, parent_index, children_index):
    print("position length:", len(position))
    print("pos:", position)
    print("joints length:", len(joints))
    print("parent_index:", len(parent_index))
    for i in range(len(joints)):
        print(i, "joint:", joints[i], "parent_index:", parent_index[i], "position:", position[0][i])

    print(children_index)
    for index, children in children_index.items():
        print(index, ":", children)


def adjust_height(rotations, joints, parent_index, lowj, times=0):
    if times==0:
        return rotations
    ''' build joint posotions '''
    fk = ForwardKinematics(joints)
    contact_y = np.zeros([len(rotations), 4, 3])
    for t in range(contact_y.shape[0]):
        for i, name in enumerate(joints):
            rotation_data = rotations[t]
            fk.set_root_matrix(rotation_data[0], rotation_data[1], joints[0])
            if i == 0:
                p = fk.get_node_position(i, rotation_data[i + 1])
            else:
                if i == 1:
                    p = fk.get_node_position(i, rotation_data[i + 1], parent_index[i])
                else:
                    p = fk.get_node_position(i, rotation_data[i + 1], parent_index[i])
            if name == 'LeftFoot':
                contact_y[t, 0] = p[:3]
            elif name == 'LeftToeBase':
                contact_y[t, 1] = p[:3]
            elif name == 'RightFoot':
                contact_y[t, 2] = p[:3]
            elif name == 'RightToeBase':
                contact_y[t, 3] = p[:3]
        ''' adjust root height to guarantee foot contact '''
        # leftfoot, lefttoe, rightfoot, righttoe
        #        5,       6,        10,       11
        # ind = np.array([4 * 3, 5 * 3, 9 * 3, 10 * 3])
    ''' set lowest joints as contact '''
    rx = np.zeros([contact_y.shape[0]])
    rz = np.zeros([contact_y.shape[0]])
    for i in range(contact_y.shape[0]):
        jy_i = contact_y[i, :, 1]
        dh = np.min(jy_i)
        ii = np.argmin(jy_i)
        rotations[i][0][1] -= dh
        if i > 0:
            rv = rotations[i][0] - rotations[i-1][0]
            # ii = np.argmin(np.linalg.norm((contact_y[i] - contact_y[i-1] - rv), axis=1))
            if ii > 1:
                ii = 3
            else:
                ii = 1
            dj = 0.5 * (contact_y[i, ii] - contact_y[i-1, ii] +
                        contact_y[i, lowj[i]] - contact_y[i-1, lowj[i]])
            # rotations[i][0][0] -= dj[0]
            # rotations[i][0][2] -= dj[2]
            rx[i] = 0.5 * (rx[i-1] - dj[0] + rotations[i][0][0])
            rz[i] = 0.5 * (rz[i-1] - dj[2] + rotations[i][0][2])
    # for i in range(contact_y.shape[0]):
    #     jy_i = contact_y[i]
    #     if lowj[i] < 2:
    #         low_i = 1
    #     else:
    #         low_i = 3
    #     dh = jy_i[low_i]
    #     rotations[i][0][1] -= dh
    ''' filt data '''
    h = np.array(rotations)[:, 0, 1:2]
    k_size = 2
    h[k_size:-k_size] = 0.5*(h + GaussianFilter_1D(h, k_size, 2))[k_size:-k_size]
    for i in range(h.shape[0]):
        rotations[i][0][1] = h[i][0]
        # rotations[i][0][0] = rx[i]
        # rotations[i][0][2] = rz[i]
    return adjust_height(rotations, joints, parent_index, lowj, times-1)


def test_IK(positions, rot, joints, parent_index, children_list, out_file, lowj):
    ik = InverseKinematics(positions, joints, parent_index, children_list)
    rotations = ik.calculate_all_rotation()
    filter = DataFilter(rotations, FILTER_WINDOW)
    # ''' add root rotations here '''
    # for i in range(len(rotations)):
    #     rotations[i][1][2] = rot[i]
    # updated_rotations = filter.smooth_data()
    rotations = adjust_height(rotations, joints, parent_index, lowj, times=2)
    writer = BVHWriter(rotations, out_file, joints, children_list)
    writer.write_to_bvh()
    print("Saved motion to", out_file)


def test_bvh(positions, joints, children_list):
    writer = BVHWriter(positions, "output.bvh", joints, children_list)
    writer.get_bvh_info_of_3d_pose_estimation()


def main(args):
    position, joints, parent_index = read_data(args.position, args.joint_name)
    children_index = dict()
    for i in range(len(joints)):
        children = []
        for j in range(len(parent_index)):
            if (i == parent_index[j]):
                children.append(j)
        children_index[i] = children

    # test_bvh(position, joints, children_index)
    # check_data(position, joints, parent_index, children_index)
    # test_FK(joints, parent_index)
    ''' add root rotation '''
    rot = np.loadtxt(args.rotation)
    rot = rot / np.pi * 180.0
    test_IK(position, rot, joints, parent_index, children_index, args.output, args.lowj)


def pos2bvh(file_names, file_dir, lj=None):
    print('pos2bvh', file_names['n'])
    parser = argparse.ArgumentParser(description='Position data to BVH file.')
    parser.add_argument('--joint_name', nargs='?', default='./pamc_joint_pfnn.txt', help='The names of joints.')
    parser.add_argument('--position', nargs='?', default=os.path.join(file_dir, '{}'.format(file_names['m'])),
                        help='File name to store position.')
    parser.add_argument('--rotation', nargs='?', default=os.path.join(file_dir, '{}'.format(file_names['rm'])),
                        help='File name to store root rotation.')
    parser.add_argument('--output', nargs='?', default=os.path.join(file_dir, '{}.bvh'.format(file_names['n'])),
                        help='The output file.')
    parser.add_argument('--lowj', nargs='?', default=lj,
                        help='Lowest index among 4 contact joints.')

    args = parser.parse_args()
    main(args)


# --position data/position.txt --joint_name data\joint.txt --output data/result.bvh
if __name__ == '__main__':
    '''
    --joint_name
    data/pamc_joint.txt
    --position
    data/gt_test_4K_95_cat_gt_filtRoot.txt
    --output
    data/gt_test_4K_95_cat_gt_filtRoot.bvh
    '''
    file_names = [
        'm_2604_60_120_t2718.txt'
    ]
    file_names = os.listdir(r'./data/')
    for file_name in file_names:
        # if len(re.findall(r'\[\d+, \d+, \d+]', file_name)) > 0 and file_name[-4:]=='.txt':
        if re.match(r'm_\d+_\d+_t\d+_\d+.txt', file_name):
            file_name = file_name[:-4]
            # file_name = r'turn/16_11_position'
            print('processing', file_name)
            parser = argparse.ArgumentParser(description='Position data to BVH file.')
            parser.add_argument('--joint_name', nargs='?', default='./data/bk.pfnn/pamc_joint.txt', help='The names of joints.')
            parser.add_argument('--position', nargs='?', default='./data/{}.txt'.format(file_name), help='File name to store position.')
            parser.add_argument('--rotation', nargs='?', default='./data/{}.txt'.format('r'+file_name),
                                help='File name to store root rotation.')
            parser.add_argument('--output', nargs='?', default='./data/{}.bvh'.format(file_name), help='The output file.')

            args = parser.parse_args()
            main(args)

    # comment out line 82 in calculate_rotation_for_frame in ik.py