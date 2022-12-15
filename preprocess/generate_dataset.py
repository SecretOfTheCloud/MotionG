import sys
import os
import numpy as np
# import shutil
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R
import pandas as pd
import torch

import process
sys.path.append('../PyTorch/PAE')
import Network

window = 2.0 #time duration of the time window
frames = 121 #sample count of the time window (60FPS)
keys = 13 #optional, used to rescale the FT window to resolution for motion controller training afterwards
joints = 28

input_channels = 3*joints #number of channels along time in the input data (here 3*J as XYZ-velocity component of each joint)
phase_channels = 6 #desired number of latent phase channels (usually between 2-10)


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
    start_frame, stop_frame = frame_utile()

    #导入DeepPhase模型
    network = ToDevice(Network.Model(
        input_channels=input_channels,
        embedding_channels=phase_channels,
        time_range=frames,
        key_range=keys,
        window=window
    ))

    network.load_state_dict(torch.load("../model/DeepPhase.pth"))
    network.eval()


    # 这里划分测试和训练文件是为了之后测试模型泛化能力
    # for train and test clips
    train_input, train_output, train_clip = np.array([]).reshape((0,316)), np.array([]).reshape((0,306)), np.array([]).reshape((0,2))
    test_input, test_output, test_clip = [], [], []
    for i, item in enumerate(bvh_files):
        print('Processing %i of %i (%s)' % (i+1, total_len, item))

        nom = item.split('\\')[-1].split('.')[-2]
        start = start_frame[nom]
        stop = stop_frame[nom]

        input_frame, output_frame, input_clip = process_data(item, start=start, end=stop, network=network)
        print(input_frame.shape)
        print(output_frame.shape)

        if i in test_idx:
            test_input = np.append(test_input, input_frame, axis=0)
            test_output = np.append(test_output, output_frame, axis=0)
            test_clip = np.append(test_clip, input_clip, axis=0)
        else:
            train_input = np.append(train_input, input_frame, axis=0)
            train_output = np.append(train_output, output_frame, axis=0)
            train_clip = np.append(train_clip, input_clip, axis=0)


    np.savez_compressed('../datasets/database1210_1.npz', X=train_input, Y=train_output, Z=train_clip)


def get_folders(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isdir(os.path.join(directory,f))]

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh')]

def process_data(bvh_file, start=None, end=None, ordre=None, network = None):

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

    joint_positions_L_list = []
    joint_orientations_L_list = []

    if start==None:
        start=0
    if end == None:
        end=len(joint_positions_G_list)

    joint_position_50, joint_orientation_50 = process.part2_forward_kinematics_local(joint_name, joint_parent,
                                                                 joint_offset, motion_data[49])

    for i in range(50+start, end-71):
        joint_positions_L, joint_orientations_L = process.part2_forward_kinematics_local(joint_name, joint_parent,
                                                                 joint_offset, motion_data[i])
        # if i<100+start:
        #     print(joint_positions_L[0:10])
        feature_one_frame = []
        for change in range(-50,70,10):
            vector_distance = joint_positions_G_list[i+change]-joint_positions_G_list[i]
            vector_distance_L = joint_orientations_G_list[i]@vector_distance
            feature_one_frame += [vector_distance_L[0], vector_distance_L[2]]

        #这里有24维的根轨迹

        feature_one_frame += list(joint_positions_L.reshape(-1))
        #28*3=84

        if i == 50+start:
            feature_one_frame += list(joint_positions_L.reshape(-1)-joint_position_50.reshape(-1))
            
            last = joint_positions_L.reshape(-1)
        else:
            feature_one_frame += list(joint_positions_L.reshape(-1)-last)
            bb=joint_positions_L.reshape(-1)-last
            # print(bb)
            # aa=0
            # for ww in bb:
            #     if ww == 0: aa+=1
            # print(aa)
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

    for i in range(start+51, end-70):
        vector_distance = joint_positions_G_list[i]-joint_positions_G_list[i-1]
        vector_distance_L = joint_orientations_G_list[i-1]@vector_distance

        joint_positions_L, joint_orientations_L = process.part2_forward_kinematics_local_future(joint_name, joint_parent,
                                                                 joint_offset, motion_data[i], 
                                                                 R.inv(R.from_matrix(joint_orientations_G_list[i]))*R.from_matrix(joint_orientations_G_list[i-1]),
                                                                 vector_distance_L)

        feature_one_frame = []
        for change in range(0,70,10):
            vector_distance = joint_positions_G_list[i+change]-joint_positions_G_list[i-1]
            # vector_distance_L = joint_orientations_G_list[i-1]@vector_distance
            vector_distance_L = joint_orientations_G_list[i]@vector_distance
            feature_one_frame += [vector_distance_L[0], vector_distance_L[2]]
        
        #14

        feature_one_frame += list(joint_positions_L.reshape(-1))
            #28*3=84

        if i == 51+start:
            feature_one_frame += list(joint_positions_L.reshape(-1)-joint_position_51.reshape(-1))
            last = joint_positions_L.reshape(-1)
        else:
            feature_one_frame += list(joint_positions_L.reshape(-1)-last)
            last = joint_positions_L.reshape(-1)
        #84

        feature_one_frame += list(joint_orientations_L.reshape(-1))
        #28*4=112

        y = np.append(y, np.array(feature_one_frame).reshape(1,-1), axis=0)

    
    if network == None:
        x = np.concatenate((x, np.zeros([x.shape[0], 12])), axis=1)
        y = np.concatenate((y, np.zeros([x.shape[0], 12])), axis=1)
    else:
        longeur_x = len(x)

        v_x=x[:,108:192]
        v_p_x = np.array([]).reshape((0, 12))
        for index in range(60, longeur_x-60):
            features = []
            for joint in range(84):
                for modification in range(-60,61):
                    features.append(v_x[index+modification,joint])
            vector=ToDevice(torch.from_numpy(np.array(features).reshape(1,-1)).type(torch.FloatTensor))
            # aa=0
            # for ww in vector[0]:
            #     if ww == 0: aa+=1
            # print(aa)
            # print(vector.shape)
            # print(vector[:,0:10])
            # print(vector.dtype)
            yPred, latent, signal, params = network(vector)
            # p=np.array(params[0].cpu()).reshape(1,6)
            # f=np.array(params[1].cpu()).reshape(1,6)
            p=params[0].cpu().detach().numpy().reshape(1,6)
            f=params[1].cpu().detach().numpy().reshape(1,6)
            p_f=np.append(p,f,axis=1)
            v_p_x = np.append(v_p_x, p_f, axis=0)
        x=x[60:longeur_x-60]
        print("vvvvvvvvvvvvvvvvvvvv")
        print(v_p_x)
        x = np.concatenate((x, v_p_x), axis=1)

        v_y=y[:,98:182]
        v_p_y = np.array([]).reshape((0, 12))
        for index in range(60, longeur_x-60):
            features = []
            for joint in range(84):
                for modification in range(-60,61):
                    features.append(v_y[index+modification,joint])
            vector=ToDevice(torch.from_numpy(np.array(features).reshape(1,-1)).type(torch.FloatTensor))
            yPred, latent, signal, params = network(vector)
            # print(params[0].cpu())
            p=params[0].cpu().detach().numpy().reshape(1,6)
            f=params[1].cpu().detach().numpy().reshape(1,6)
            p_f=np.append(p,f,axis=1)
            v_p_y = np.append(v_p_y, p_f, axis=0)
        y=y[60:longeur_x-60]
        y = np.concatenate((y, v_p_y), axis=1)

    len_frames = x.shape[0]
    z = np.zeros([len_frames, 2])

    
    return x, y, z

def frame_utile():
    df = pd.read_csv("../database/Frame_Cuts.csv")
    df_list = df.values.tolist()

    start_Frame = {}
    stop_Frame = {}

    for i in range(100):
        style_name = df_list[i][0]

        courrant = df_list[i][1]
        start_Frame[style_name+'_BR'] = courrant
        courrant = df_list[i][2]
        stop_Frame[style_name+'_BR'] = courrant

        courrant = df_list[i][3]
        start_Frame[style_name+'_BW'] = courrant
        courrant = df_list[i][4]
        stop_Frame[style_name+'_BW'] = courrant

        courrant = df_list[i][5]
        start_Frame[style_name+'_FR'] = courrant
        courrant = df_list[i][6]
        stop_Frame[style_name+'_FR'] = courrant

        courrant = df_list[i][7]
        start_Frame[style_name+'_FW'] = courrant
        courrant = df_list[i][8]
        stop_Frame[style_name+'_FW'] = courrant

        courrant = df_list[i][9]
        start_Frame[style_name+'_ID'] = courrant
        courrant = df_list[i][10]
        stop_Frame[style_name+'_ID'] = courrant

        courrant = df_list[i][11]
        start_Frame[style_name+'_SR'] = courrant
        courrant = df_list[i][12]
        stop_Frame[style_name+'_SR'] = courrant

        courrant = df_list[i][13]
        start_Frame[style_name+'_SW'] = courrant
        courrant = df_list[i][14]
        stop_Frame[style_name+'_SW'] = courrant

        courrant = df_list[i][15]
        start_Frame[style_name+'_TR1'] = courrant
        courrant = df_list[i][16]
        stop_Frame[style_name+'_TR1'] = courrant

        courrant = df_list[i][17]
        start_Frame[style_name+'_TR2'] = courrant
        courrant = df_list[i][18]
        stop_Frame[style_name+'_TR2'] = courrant

        courrant = df_list[i][19]
        start_Frame[style_name+'_TR3'] = courrant
        courrant = df_list[i][20]
        stop_Frame[style_name+'_TR3'] = courrant
    
    return start_Frame, stop_Frame

def ToDevice(x):
    return x.cuda() if torch.cuda.is_available() else x

if __name__ == '__main__':
    main()

