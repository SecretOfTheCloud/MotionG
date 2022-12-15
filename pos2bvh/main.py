# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import argparse
import numpy as np
from forward_kinematics import ForwardKinematics
from constants import DEBUG, PARENT_INDEX, JOINT_NAME, FILTER_WINDOW, set_initial_offset
from inverse_kinematics import InverseKinematics
from filter_rotations import DataFilter
from bvh_writer import BVHWriter


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
    set_initial_offset(position_data[0])
    return position_data, joint_data, parent_index


def test_FK(joints, parent_index):
    f = open("data/rotation_6_test.txt", "r")
    bvh_data = f.readlines()[0].replace('\n', '').split(' ')
    rotation_data = []
    channel_6 = True
    # channel_6 = False
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


def test_IK(positions, joints, parent_list, children_list, out_file):
    ik = InverseKinematics(positions, joints, parent_list, children_list)
    rotations = ik.calculate_all_rotation()
    # filter = DataFilter(rotations, FILTER_WINDOW)
    # updated_rotations = filter.smooth_data()

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
    test_IK(position, joints, parent_index, children_index, args.output)


# --position data/position.txt --joint_name data\joint.txt --output data/result.bvh
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Position data to BVH file.')
    parser.add_argument('--position', nargs='?', default='position.txt', help='File name to store position.')
    parser.add_argument('--joint_name', nargs='?', default='joint.txt', help='The names of joints.')
    parser.add_argument('--output', nargs='?', default='result.bvh', help='The output file.')

    args = parser.parse_args()
    main(args)
