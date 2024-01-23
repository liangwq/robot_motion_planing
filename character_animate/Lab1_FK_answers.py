import numpy as np
from scipy.spatial.transform import Rotation as R

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



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    def parse_hierarchy(pred):
        nonlocal joint_offset
        nonlocal line

        parts = str.split(bvh_file[line].strip(), ' ')
        if parts[0].startswith('End'):
            parts[1] = joint_name[pred] + '_end'

        offset = np.array([float(num) for num in str.split(bvh_file[line + 2].strip())[1:]]).reshape(1,3)
        joint_name.append(parts[1])
        joint_parent.append(pred)
        joint_offset = np.append(joint_offset, offset, axis=0)
        line += 4

        if parts[0].startswith('End'):
            return

        pred = len(joint_name) - 1
        while bvh_file[line].strip().startswith('JOINT') or bvh_file[line].strip().startswith('End'):
            parse_hierarchy(pred)
        line += 1
     

    joint_name = []
    joint_parent = []
    joint_offset = np.empty((0, 3))

    with open(bvh_file_path, 'r') as f:
        bvh_file = f.readlines()
        for i in range(len(bvh_file)):
            if bvh_file[i].startswith('ROOT'):
                line = i
                parse_hierarchy(-1)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    bone_num = len(joint_name)

    joint_positions = np.empty((bone_num,3))
    joint_orientations = np.empty((bone_num,4))

    joint_positions[0] = motion_data[frame_id][:3]
    #joint_orientations[0] = R.from_euler('XYZ', motion_data[frame_id][3:6], degrees=True).as_quat()
    joint_orientations[0] = R.from_euler('ZYX', motion_data[frame_id][3:6], degrees=True).as_quat()
    
    frame_count = 1
    for i in range(1, bone_num):
        p = joint_parent[i]
        cur_rotate = None
        if joint_name[i].endswith('_end'):
            #cur_rotate = R.from_euler('XYZ', [0., 0., 0.], degrees=True)
            cur_rotate = R.from_euler('ZYX', [0., 0., 0.], degrees=True)
        else:
            #print(motion_data[frame_id][3+frame_count*3 : 6+frame_count*3])
            #cur_rotate = R.from_euler('XYZ', motion_data[frame_id][3+frame_count*3 : 6+frame_count*3], degrees=True)
            cur_rotate = R.from_euler('ZYX', motion_data[frame_id][3+frame_count*3 : 6+frame_count*3], degrees=True)
            frame_count += 1

        p_orient =  R.from_quat(joint_orientations[p])
        joint_orientations[i] = (p_orient * cur_rotate).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offset[i])
        #print(joint_name[i])
        #print(joint_positions[i])
        #print(joint_orientations[i])
    #print(joint_positions)
    #print(joint_orientations)

    return joint_positions, joint_orientations



def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    def index_bone_to_channel(index, flag):
        if flag == 't':
            end_bone_index = end_bone_index_t
        else:
            end_bone_index = end_bone_index_a
        for i in range(len(end_bone_index)):
            if end_bone_index[i] > index:
                return index - i
        return index - len(end_bone_index)
    
    def get_t2a_offset(bone_name):
        l_bone = ['lShoulder', 'lElbow', 'lWrist'] 
        r_bone = ['rShoulder', 'rElbow', 'rWrist']
        if bone_name in l_bone:
            return R.from_euler('XYZ', [0.,0.,45.], degrees=True)
        if bone_name in r_bone:
            return R.from_euler('XYZ', [0.,0.,-45.], degrees=True)
        return R.from_euler('XYZ', [0.,0.,0.], degrees=True)


    motion_data = load_motion_data(A_pose_bvh_path)

    t_name, t_parent, t_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_name, a_parent, a_offset = part1_calculate_T_pose(A_pose_bvh_path)

    end_bone_index_t = []
    for i in range(len(t_name)):
        if t_name[i].endswith('_end'):
            end_bone_index_t.append(i)

    end_bone_index_a = []
    for i in range(len(a_name)):
        if a_name[i].endswith('_end'):
            end_bone_index_a.append(i)

    for m_i in range(len(motion_data)):
        frame = motion_data[m_i]
        cur_frame = np.empty(frame.shape[0])
        cur_frame[:3] = frame[:3]
        for t_i in range(len(t_name)):
            cur_bone = t_name[t_i]
            a_i = a_name.index(t_name[t_i])
            if cur_bone.endswith('_end'):
                continue
            channel_t_i = index_bone_to_channel(t_i, 't')
            channel_a_i = index_bone_to_channel(a_i, 'a')
            
            # retarget
            local_rotation = frame[3+channel_a_i*3 : 6+channel_a_i*3]
            if cur_bone in ['lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow', 'rWrist']:
                p_bone_name = t_name[t_parent[t_i]]
                Q_pi = get_t2a_offset(p_bone_name)
                Q_i = get_t2a_offset(cur_bone)
                local_rotation = (Q_pi * R.from_euler('XYZ', local_rotation, degrees=True) * Q_i.inv()).as_euler('XYZ', degrees=True)
            cur_frame[3+channel_t_i*3 : 6+channel_t_i*3] = local_rotation

        motion_data[m_i] = cur_frame

    return motion_data
