import re


class BvhNode:

    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return str(' '.join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:

    def __init__(self, data):
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.tokenize()

    def tokenize(self):
        first_round = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                    first_round.append(re.split('\\s+', accumulator.strip()))
                    accumulator = ''
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)
        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)
        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter('JOINT'):
                iterate_joints(child)
        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]
        raise LookupError('joint not found')

    def joint_offset(self, name):
        joint = self.get_joint(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        joint = self.get_joint(name)
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index
        
    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(
                        self.frames[frame_index][joint_index + channel_index]
                    )
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')
      
    
import numpy as np
with open('data/lafan1/run1_subject5.bvh') as f:
    mocap = Bvh(f.read())

def iterate_joints(joint):
    bones.append(str(joint))
    for child in joint.filter('JOINT'):
        iterate_joints(child)
        
rotations = np.zeros((len(mocap.get_joints_names()), 3), dtype=np.float64)
joint_parent=[]
joint_offset = []
i=0
for join_name in mocap.get_joints_names():
    name = join_name
    offset = mocap.joint_offset(join_name)
    parent_name = mocap.joint_parent(join_name)
    parent_index = mocap.joint_parent_index(join_name)
    #rotations.append(list(offset))
    for j in range (3):
        rotations[i][j] = offset[j]
    i=i+1
    joint_parent.append(parent_index)
    joint_offset.append(offset)
    print(f'join name:{name} value:{list(offset)} parent name:{parent_name} parent index {parent_index}')
    
import plotly.graph_objects as go
import numpy as np

# 节点信息
nodes_info = joint_positions


# 父节点关系
parent_indices = joint_parent#[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

# 创建可交互的3D图
fig = go.Figure()

# 绘制节点
for i in range(len(nodes_info)):
    x, y, z = nodes_info[i]
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(size=5), name=f'Node {i}'))

# 绘制连线
for i in range(1, len(parent_indices)):
    parent_index = parent_indices[i]
    x_parent, y_parent, z_parent = nodes_info[parent_index]
    x_child, y_child, z_child = nodes_info[i]
    fig.add_trace(go.Scatter3d(x=[x_parent, x_child], y=[y_parent, y_child], z=[z_parent, z_child], mode='lines', line=dict(color='gray')))

# 设置坐标轴标签
fig.update_layout(scene=dict(xaxis_title='X轴', yaxis_title='Y轴', zaxis_title='Z轴'))

# 显示图形
fig.show()