import sys
import os
import numpy as np
# import shutil
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R

import process

def main():
    root_path = '../datasets/'
    dataset_name = '100Style'
    dataset_dir = os.path.join(root_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    print("Create folfer ", dataset_dir)
    #此处只用了测试的三个风格
    bvh_dir = '../database/1STYLE'
    bvh_dir_sub = get_folders(bvh_dir)
    bvh_files = np.concatenate([get_files(i) for i in bvh_dir_sub], axis=0)

    total_len = len(bvh_files)
    test_idx = np.random.choice(total_len, total_len//10, replace=False)
    train_inx = np.setdiff1d(np.arange(0,total_len), test_idx)
    test_bvh_files = bvh_files[test_idx]
    train_bvh_files = bvh_files[train_inx]
    
    # 这里划分测试和训练文件是为了之后测试模型泛化能力
    # for train and test clips
    train_input, train_output, train_clip = np.array([]).reshape((0,316)), np.array([]).reshape((0,306)), np.array([]).reshape((0,2))
    test_input, test_output, test_clip = [], [], []
    for i, item in enumerate(bvh_files):
        print('Processing %i of %i (%s)' % (i+1, total_len, item))

        input_frame, output_frame, input_clip = process_data(item)
        print(input_frame.shape)
        print(output_frame.shape)

        if i in test_idx:
            test_input = np.append(test_input, input_frame, axis=0)
            test_output = np.append(test_output, output_frame, axis=0)
            test_clip = np.append(test_clip, input_clip)
        else:
            train_input = np.append(train_input, input_frame, axis=0)
            train_output = np.append(train_output, output_frame, axis=0)
            train_clip = np.append(train_clip, input_clip)


    np.savez_compressed('../database1116_1.npz', X=train_input, Y=train_output, Z=train_clip)


def get_folders(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isdir(os.path.join(directory,f))]

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh')]

def process_data(bvh_file, start=None, end=None, ordre=None):

    x = np.array([]).reshape((0,304))
    y = np.array([]).reshape((0,294))

    joint_name, joint_parent, joint_offset = process.load(bvh_file)
    motion_data = process.load_motion_data(bvh_file)

    joint_positions_G_list = []
    joint_orientations_G_list = []

    for motion_frame in motion_data:
        joint_positions_G, joint_orientations_G = process.part2_forward_kinematics(joint_name, joint_parent,
                                                                 joint_offset, motion_frame)
        joint_positions_G_list.append(joint_positions_G[0])
        joint_orientations_G_list.append(R.inv(R.from_quat(joint_orientations_G[0])).as_matrix())

    len_data = len(joint_positions_G_list)

    joint_positions_L_list = []
    joint_orientations_L_list = []

    joint_position_50, joint_orientation_50 = process.part2_forward_kinematics_local(joint_name, joint_parent,
                                                                 joint_offset, motion_data[49])

    for i in range(50, len_data-71):
        joint_positions_L, joint_orientations_L = process.part2_forward_kinematics_local(joint_name, joint_parent,
                                                                 joint_offset, motion_data[i])

        feature_one_frame = []
        for change in range(-50,70,10):
            vector_distance = joint_positions_G_list[i+change]-joint_positions_G_list[i]
            vector_distance_L = joint_orientations_G_list[i]@vector_distance
            feature_one_frame += [vector_distance_L[0], vector_distance_L[2]]

        #这里有24维的根轨迹

        feature_one_frame += list(joint_positions_L.reshape(-1))
        #28*3=84

        if i == 50:
            feature_one_frame += list(joint_positions_L.reshape(-1)-joint_position_50.reshape(-1))
            last = joint_positions_L.reshape(-1)
        else:
            feature_one_frame += list(joint_positions_L.reshape(-1)-last)
            last = joint_positions_L.reshape(-1)
        #84

        feature_one_frame += list(joint_orientations_L.reshape(-1))
        #28*4=112

        x = np.append(x, np.array(feature_one_frame).reshape(1,-1), axis=0)


    vector_distance = joint_positions_G_list[50]-joint_positions_G_list[49]
    vector_distance_L = joint_orientations_G_list[49]@vector_distance
    joint_position_51, joint_orientation_51 = process.part2_forward_kinematics_local_future(joint_name, joint_parent,
                                                                 joint_offset, motion_data[50],
                                                                 R.inv(R.from_matrix(joint_orientations_G_list[50]))*R.from_matrix(joint_orientations_G_list[49]),
                                                                 vector_distance_L)

    for i in range(51, len_data-70):
        vector_distance = joint_positions_G_list[i]-joint_positions_G_list[i-1]
        vector_distance_L = joint_orientations_G_list[i-1]@vector_distance

        joint_positions_L, joint_orientations_L = process.part2_forward_kinematics_local_future(joint_name, joint_parent,
                                                                 joint_offset, motion_data[i], 
                                                                 R.inv(R.from_matrix(joint_orientations_G_list[i]))*R.from_matrix(joint_orientations_G_list[i-1]),
                                                                 vector_distance_L)

        feature_one_frame = []
        for change in range(0,70,10):
            vector_distance = joint_positions_G_list[i+change]-joint_positions_G_list[i-1]
            vector_distance_L = joint_orientations_G_list[i-1]@vector_distance
            feature_one_frame += [vector_distance_L[0], vector_distance_L[2]]
        
        #14

        feature_one_frame += list(joint_positions_L.reshape(-1))
            #28*3=84

        if i == 51:
            feature_one_frame += list(joint_positions_L.reshape(-1)-joint_position_51.reshape(-1))
            last = joint_positions_L.reshape(-1)
        else:
            feature_one_frame += list(joint_positions_L.reshape(-1)-last)
            last = joint_positions_L.reshape(-1)
        #84

        feature_one_frame += list(joint_orientations_L.reshape(-1))
        #28*4=112

        y = np.append(y, np.array(feature_one_frame).reshape(1,-1), axis=0)

    network = None
    if network == None:
        len_frames = x.shape[0]
        x = np.concatenate((x, np.zeros([len_frames, 12])), axis=1)
        y = np.concatenate((y, np.zeros([len_frames, 12])), axis=1)
    else:
        x, y = network(x, y)

    z = np.zeros([len_frames, 2])

    
    return x, y, z


if __name__ == '__main__':
    main()

