import numpy as np
from scipy.spatial.transform import Rotation as R


def load(filename):
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    offsets = np.array([]).reshape((0,3))
    orients = []
    parents = np.array([], dtype=int)

    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        if line.lstrip().startswith('ROOT'):
            names.append(line.split()[1])
            offsets = np.append(offsets, np.array([[0,0,0]]), axis=0)
            orients.append(R.from_rotvec([0,0,0]))
            parents = np.append(parents, active)
            active = (len(parents)-1)

        if "{" in line: continue

        if "}" in line:
            active = parents[active]
            continue

        if line.lstrip().startswith('OFFSET'):
            parties = line.split()
            offsets[active] = np.array([float(parties[1]), float(parties[2]), float(parties[3])])
            continue

        if line.lstrip().startswith('JOINT'):
            parties = line.split()
            names.append(line.split()[1])
            offsets = np.append(offsets, np.array([[0,0,0]]), axis=0)
            orients.append(R.from_rotvec([0,0,0]))
            parents = np.append(parents, active)
            active = (len(parents)-1)
            continue

        if line.lstrip().startswith('End'):
            parties = line.split()
            names.append(names[active]+'_end')
            offsets = np.append(offsets, np.array([[0,0,0]]), axis=0)
            orients.append(R.from_rotvec([0,0,0]))
            parents = np.append(parents, active)
            active = (len(parents)-1)
            continue

    f.close()    

    return names, list(parents), offsets

def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_frame):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_frame:需要读取的帧
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = np.array([]).reshape((0,3))
    joint_orientations = np.array([]).reshape((0,4))

    root_offset = motion_frame[0:3]
    rotations = motion_frame[3:].reshape([-1,3])

    activate = 0
    for i, joint in enumerate(joint_name):
        if "_end" in joint:
            rotation_quat = joint_orientations[joint_parent[i]]
            joint_orientations = np.append(joint_orientations, rotation_quat.reshape([1,4]), axis=0)

            offset_parent = joint_positions[joint_parent[i]]
            offset_add = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix() @ joint_offset[i]   
            offset_current = offset_parent + offset_add
            joint_positions = np.append(joint_positions, offset_current.reshape([1,3]), axis=0)

        else:
            rotation_euler = rotations[activate]
            activate += 1

            if i == 0:
                rotation_quat = R.from_euler('YXZ', [rotation_euler[0], 
                                            rotation_euler[1], rotation_euler[2]], degrees=True).as_quat()
            else:
                rotation_local = R.from_euler('YXZ', [rotation_euler[0], 
                                            rotation_euler[1], rotation_euler[2]], degrees=True)
                
                rotation_parent = R.from_quat(joint_orientations[joint_parent[i]])
                rotation_quat = (rotation_parent*rotation_local).as_quat()
                
            joint_orientations = np.append(joint_orientations, rotation_quat.reshape([1,4]), axis=0)

            if i == 0:
                offset_current = root_offset
            else:
                offset_parent = joint_positions[joint_parent[i]]
                offset_add = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix() @ joint_offset[i]   
                offset_current = offset_parent + offset_add
            joint_positions = np.append(joint_positions, offset_current.reshape([1,3]), axis=0)

    return joint_positions, joint_orientations

def part2_forward_kinematics_local(joint_name, joint_parent, joint_offset, motion_frame):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_frame:需要读取的帧
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的相对根关节的位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的相对根关节旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = np.array([]).reshape((0,3))
    joint_orientations = np.array([]).reshape((0,4))

    root_offset = motion_frame[0:3]
    rotations = motion_frame[3:].reshape([-1,3])

    activate = 0
    for i, joint in enumerate(joint_name):
        if "_end" in joint:
            rotation_quat = joint_orientations[joint_parent[i]]
            joint_orientations = np.append(joint_orientations, rotation_quat.reshape([1,4]), axis=0)

            offset_parent = joint_positions[joint_parent[i]]
            offset_add = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix() @ joint_offset[i]   
            offset_current = offset_parent + offset_add
            joint_positions = np.append(joint_positions, offset_current.reshape([1,3]), axis=0)

        else:
            rotation_euler = rotations[activate]
            activate += 1

            if i == 0:
                rotation_quat = R.from_euler('YXZ', [0,0,0], degrees=True).as_quat()
            else:
                rotation_local = R.from_euler('YXZ', [rotation_euler[0], 
                                            rotation_euler[1], rotation_euler[2]], degrees=True)
                
                rotation_parent = R.from_quat(joint_orientations[joint_parent[i]])
                rotation_quat = (rotation_parent*rotation_local).as_quat()
                
            joint_orientations = np.append(joint_orientations, rotation_quat.reshape([1,4]), axis=0)

            if i == 0:
                offset_current = np.array([0,0,0])
            else:
                offset_parent = joint_positions[joint_parent[i]]
                offset_add = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix() @ joint_offset[i]   
                offset_current = offset_parent + offset_add
            joint_positions = np.append(joint_positions, offset_current.reshape([1,3]), axis=0)

    return joint_positions, joint_orientations


def part2_forward_kinematics_local_future(joint_name, joint_parent, joint_offset, motion_frame, Ro, offset):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_frame:需要读取的帧
    输出:
        此处的表示均在前一帧的根节点坐标系
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的相对根关节的位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的相对根关节旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = np.array([]).reshape((0,3))
    joint_orientations = np.array([]).reshape((0,4))

    root_offset = motion_frame[0:3]
    rotations = motion_frame[3:].reshape([-1,3])

    activate = 0
    for i, joint in enumerate(joint_name):
        if "_end" in joint:
            rotation_quat = joint_orientations[joint_parent[i]]
            joint_orientations = np.append(joint_orientations, rotation_quat.reshape([1,4]), axis=0)

            offset_parent = joint_positions[joint_parent[i]]
            offset_add = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix() @ joint_offset[i]   
            offset_current = offset_parent + offset_add
            joint_positions = np.append(joint_positions, offset_current.reshape([1,3]), axis=0)

        else:
            rotation_euler = rotations[activate]
            activate += 1

            if i == 0:
                rotation_quat = Ro.as_quat()
            else:
                rotation_local = R.from_euler('YXZ', [rotation_euler[0], 
                                            rotation_euler[1], rotation_euler[2]], degrees=True)
                
                rotation_parent = R.from_quat(joint_orientations[joint_parent[i]])
                rotation_quat = (rotation_parent*rotation_local).as_quat()
                
            joint_orientations = np.append(joint_orientations, rotation_quat.reshape([1,4]), axis=0)

            if i == 0:
                offset_current = offset
            else:
                offset_parent = joint_positions[joint_parent[i]]
                offset_add = R.from_quat(joint_orientations[joint_parent[i]]).as_matrix() @ joint_offset[i]   
                offset_current = offset_parent + offset_add
            joint_positions = np.append(joint_positions, offset_current.reshape([1,3]), axis=0)

    return joint_positions, joint_orientations


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data