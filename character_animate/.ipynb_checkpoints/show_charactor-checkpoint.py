from direct.showbase.ShowBase import ShowBase
import numpy as np
from panda3d.core import ClockObject
from panda3d.core import LineSegs, NodePath
from panda3d.core import LineSegs, Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter, Vec4
from panda3d.core import Point3, Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomTrifans
from math import cos, sin

import panda3d.core as pc
import math
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *

import numpy as np
from Lab1_FK_answers import *
import imageio

class CameraCtrl(DirectObject):
    def __init__(self, base, camera):
        super(CameraCtrl).__init__()
        self.accept('mouse1',self.onMouse1Down)
        self.accept('mouse1-up',self.onMouse1Up)
        self.accept('mouse2',self.onMouse2Down)
        self.accept('mouse2-up',self.onMouse2Up)
        self.accept('mouse3',self.onMouse3Down)
        self.accept('mouse3-up',self.onMouse3Up)
        self.accept('wheel_down',self.onMouseWheelDown)
        self.accept('wheel_up',self.onMouseWheelUp)

        self.accept('control-mouse1',self.onMouse1Down)
        self.accept('control-mouse1-up',self.onMouse1Up)
        self.accept('control-mouse2',self.onMouse2Down)
        self.accept('control-mouse2-up',self.onMouse2Up)
        self.accept('control-mouse3',self.onMouse3Down)
        self.accept('control-mouse3-up',self.onMouse3Up)
        self.accept('control-wheel_down',self.onMouseWheelDown)
        self.accept('control-wheel_up',self.onMouseWheelUp)

        self.position = pc.LVector3(4,4,4)
        self.center = pc.LVector3(0,1,0)
        self.up = pc.LVector3(0,1,0)

        self.base = base
        base.taskMgr.add(self.onUpdate, 'updateCamera')
        self.camera = camera
        
        self._locked_info = None
        self._locked_mouse_pos = None
        self._mouse_id = -1

        self.look()

    def look(self):    
        self.camera.setPos(self.position)
        self.camera.lookAt(self.center, self.up)

    @property
    def _mousePos(self):
        return pc.LVector2(self.base.mouseWatcherNode.getMouseX(), self.base.mouseWatcherNode.getMouseY())

    def _lockMouseInfo(self):
        self._locked_info = (pc.LVector3(self.position), pc.LVector3(self.center), pc.LVector3(self.up))
        self._locked_mouse_pos = self._mousePos

    def onMouse1Down(self):
        self._lockMouseInfo()
        self._mouse_id = 1

    def onMouse1Up(self):
        self._mouse_id = -1

    def onMouse2Down(self):
        self._lockMouseInfo()
        self._mouse_id = 2

    def onMouse2Up(self):
        self._mouse_id = -1

    def onMouse3Down(self):
        self._lockMouseInfo()
        self._mouse_id = 3

    def onMouse3Up(self):
        self._mouse_id = -1

    def onMouseWheelDown(self):
        z =  self.position - self.center 
        
        scale = 1.1

        if scale < 0.05:
            scale = 0.05

        self.position = self.center + z * scale
        self.look()

    def onMouseWheelUp(self):
        z =  self.position - self.center 
        
        scale = 0.9

        if scale < 0.05:
            scale = 0.05

        self.position = self.center + z * scale
        self.look()

    def onUpdate(self, task):
        if self._mouse_id < 0:
            return task.cont
        
        mousePosOff0 = self._mousePos - self._locked_mouse_pos
        mousePosOff = self._mousePos - self._locked_mouse_pos

        if self._mouse_id == 1:
            z = self._locked_info[0] - self._locked_info[1]

            zDotUp = self._locked_info[2].dot(z)
            zMap = z - self._locked_info[2] * zDotUp
            angX = math.acos(zMap.length() / z.length()) / math.pi * 180.0

            if zDotUp < 0:
                angX = -angX

            angleScale = 200.0

            x = self._locked_info[2].cross(z)
            x.normalize()
            y = z.cross(x)
            y.normalize()

            rot_x_angle = -mousePosOff.getY() * angleScale
            rot_x_angle += angX
            if rot_x_angle > 85:
                rot_x_angle = 85
            if rot_x_angle < -85:
                rot_x_angle = -85
            rot_x_angle -= angX

            rot_y = pc.LMatrix3()
            rot_y.setRotateMat(-mousePosOff.getX() * angleScale, y, pc.CS_yup_right)
            
            rot_x = pc.LMatrix3()
            rot_x.setRotateMat(-rot_x_angle, x, pc.CS_yup_right)

            self.position = self._locked_info[1] + (rot_x * rot_y).xform(z)

        elif self._mouse_id == 2:
            z = self._locked_info[0] - self._locked_info[1]

            shiftScale = 0.5 * z.length()

            x = self._locked_info[2].cross(z)
            z.normalize()
            x.normalize()
            y = z.cross(x)

            shift = x * -mousePosOff.getX() + y* -mousePosOff.getY()
            shift *= shiftScale
            self.position = self._locked_info[0] + shift
            self.center = self._locked_info[1] + shift

        elif self._mouse_id == 3:
            z = self._locked_info[0] - self._locked_info[1]
            
            scale = 1
            scale = 1.0 + scale * mousePosOff0.getY()

            if scale < 0.05:
                scale = 0.05

            self.position = self._locked_info[1] + z * scale

        self.look()

        return task.cont
    
class SimpleViewer(ShowBase):
    def __init__(self, fStartDirect=True, windowType=None):
        '''
        this is only used for my project... lots of assumptions...
        '''
        super().__init__(fStartDirect, windowType)
        self.disableMouse()        
        
        self.camera.lookAt(0,1,0)
        self.setupCameraLight()
        self.camera.setHpr(0,0,0)
        
        self.setFrameRateMeter(True)
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(20)
        
        self.load_ground()
        
        xSize = self.pipe.getDisplayWidth()
        ySize = self.pipe.getDisplayHeight()
        props = pc.WindowProperties()
        props.setSize(min(xSize-300, 500), min(ySize-300, 500))
        self.win.requestProperties(props)
        
        # color for links
        color = [131/255,175/255,155/255,1]
        self.tex = self.create_texture(color, 'link_tex')
        
        self.load_character()
        self.update_func = None
        self.add_task(self.update, 'update')
        self.update_flag = True
        self.accept('space', self.receive_space)
        pass
    def get_charactor_model_data(self):
        bvh_file_path = "data/lafan1/run1_subject5.bvh"#"example.bvh"#
        joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
        motion_data = load_motion_data(bvh_file_path)
        joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)
        #joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 1)
        return joint_name, joint_parent,joint_positions, joint_orientations

        

    def load_character(self):
        joint_name,parent_index,joint_pos,joint_orientations = self.get_charactor_model_data()
        joint_pos = joint_pos/80  #除以80，让模型可以适配模拟场景中活动范围

        
        body_pos = []
        for i in range(len(joint_name)):
            if 'end' in joint_name[i]:
                pass
            body_pos.append(list(joint_pos[i]))
        body_pos = np.array(body_pos)
        


        body_rot = None
        
        joint, body = [], []
        
        thickness = 0.03
        name_idx_map = {joint_name[i]:i for i in range(len(joint_name))}

        body_pos[name_idx_map['Hips']] = joint_pos[name_idx_map['Hips']]
        body_pos[name_idx_map['Head']] = joint_pos[name_idx_map['Head']]
        body_pos[name_idx_map['Neck']] = joint_pos[name_idx_map['Neck']]
        body_pos[name_idx_map['Spine']] = joint_pos[name_idx_map['Spine']]
        body_pos[name_idx_map['Spine1']] = joint_pos[name_idx_map['Spine1']]  
        body_pos[name_idx_map['Spine2']]  = joint_pos[name_idx_map['Spine2']]  
        body_pos[name_idx_map['LeftUpLeg']] = [(joint_pos[name_idx_map['LeftUpLeg']][0] + joint_pos[name_idx_map['LeftLeg']][0])/2+0.19,joint_pos[name_idx_map['LeftUpLeg']][1],joint_pos[name_idx_map['LeftUpLeg']][2]]
        body_pos[name_idx_map['RightUpLeg']] = [(joint_pos[name_idx_map['RightUpLeg']][0] + joint_pos[name_idx_map['RightLeg']][0])/2+0.19,joint_pos[name_idx_map['RightUpLeg']][1],joint_pos[name_idx_map['RightUpLeg']][2]]
        body_pos[name_idx_map['LeftLeg']] = [(joint_pos[name_idx_map['LeftLeg']][0]+joint_pos[name_idx_map['LeftFoot']][0])/2 +0.18,joint_pos[name_idx_map['LeftLeg']][1],joint_pos[name_idx_map['LeftLeg']][2]]
        body_pos[name_idx_map['RightLeg']] =[(joint_pos[name_idx_map['RightLeg']][0]+joint_pos[name_idx_map['RightFoot']][0])/2 +0.18,joint_pos[name_idx_map['RightLeg']][1],joint_pos[name_idx_map['RightLeg']][2]]
        body_pos[name_idx_map['LeftFoot']] = joint_pos[name_idx_map['LeftFoot']] 
        body_pos[name_idx_map['RightFoot']] = joint_pos[name_idx_map['RightFoot']]
        body_pos[name_idx_map['LeftToe']] = joint_pos[name_idx_map['LeftToe']] 
        body_pos[name_idx_map['RightToe']] = joint_pos[name_idx_map['RightToe']]
        body_pos[name_idx_map['LeftShoulder']] = joint_pos[name_idx_map['LeftShoulder']]
        body_pos[name_idx_map['RightShoulder']] = joint_pos[name_idx_map['RightShoulder']]
        body_pos[name_idx_map['LeftArm']] = [(joint_pos[name_idx_map['LeftArm']][0]+joint_pos[name_idx_map['LeftForeArm']][0])/2+0.09, joint_pos[name_idx_map['LeftArm']][1],joint_pos[name_idx_map['LeftArm']][2]]
        body_pos[name_idx_map['RightArm']] = [joint_pos[name_idx_map['RightArm']][0]+0.19,joint_pos[name_idx_map['RightArm']][1],joint_pos[name_idx_map['RightArm']][2]]
        body_pos[name_idx_map['LeftForeArm']] = [joint_pos[name_idx_map['LeftForeArm']][0]+0.12, joint_pos[name_idx_map['LeftForeArm']][1],joint_pos[name_idx_map['LeftForeArm']][2]]
        body_pos[name_idx_map['RightForeArm']] = [joint_pos[name_idx_map['RightForeArm']][0]+0.12,joint_pos[name_idx_map['RightForeArm']][1],joint_pos[name_idx_map['RightForeArm']][2]]
        body_pos[name_idx_map['LeftHand']] = joint_pos[name_idx_map['LeftHand']] 
        body_pos[name_idx_map['RightHand']] = joint_pos[name_idx_map['RightHand']] 

        scale = [ [thickness]*3 for i in range(len(body_pos))]
        
        scale[name_idx_map['Hips']] = [0.06,0.03,thickness*3]
        scale[name_idx_map['Head']] =  [0.05,0.05,thickness*2]
        scale[name_idx_map['Neck']] = [thickness*2,0.03,thickness*1.5]
        scale[name_idx_map['Spine']] = scale[name_idx_map['Spine1']] = scale[name_idx_map['Spine2']]  = [thickness,0.006,thickness*3]
        
        scale[name_idx_map['LeftUpLeg']] = scale[name_idx_map['RightUpLeg']] = [thickness*8,0.05,thickness*2]
        scale[name_idx_map['LeftLeg']] = scale[name_idx_map['RightLeg']] = [thickness*8,0.03,thickness]
        scale[name_idx_map['LeftFoot']] = scale[name_idx_map['RightFoot']] = [thickness*1.5,thickness*3,0.06]
        scale[name_idx_map['LeftToe']] = scale[name_idx_map['RightToe']] = [thickness*1.5,thickness*0.6,0.02]

        scale[name_idx_map['LeftShoulder']] = scale[name_idx_map['RightShoulder']] = [0.05,thickness,thickness]
        scale[name_idx_map['LeftArm']] = scale[name_idx_map['RightArm']] = [0.21,thickness*1.5,thickness*1.5]
        scale[name_idx_map['LeftForeArm']] = scale[name_idx_map['RightForeArm']] = [0.19,thickness,thickness]
        scale[name_idx_map['LeftHand']] = scale[name_idx_map['RightHand']] = [0.03,thickness*0.6,thickness*1.2]

        # 每个节点对应的父节点
        parents = parent_index
        local_pos = []
        #for nodes in joint_pos:
        for i in range(len(parents)):
            if parents[i] == -1:
                local_pos.append([joint_pos[parents[1]],joint_pos[0]])
            local_pos.append([joint_pos[parents[i]],joint_pos[i]])
 

        for i in range(len(joint_pos)):
            joint.append(self.create_joint(i, joint_pos[i], 'end' in joint_name[i]))
            '''
            if i < len(parents):
                body.append(self.create_link_free(i,joint_pos[parents[i]],joint_pos[i], rot = body_rot[i] if body_rot is not None else None))
                
                #print(body[-1])
                #print(body[0])
                body[0].wrtReparentTo(joint[0])
                print(body[0])
                print(joint[0])
            '''
            if i < body_pos.shape[0]:
                body.append(self.create_link(i, body_pos[i], scale[i], rot = body_rot[i] if body_rot is not None else None))
                body[-1].wrtReparentTo(joint[-1])

        self.joints = joint
        self.joint_name = joint_name
        self.name2idx = name_idx_map
        self.parent_index = parent_index
        self.init_joint_pos = self.get_joint_positions()

    def create_link_free(self,link_id, joint_1, joint_2, rot):
        
        # 假设已有两个坐标点
        point1 = joint_1
        point2 = joint_2

        # 创建一个 LineSegs 对象
        line_segs = LineSegs()

        # 添加线段的两个端点
        line_segs.moveTo(point1[0], point1[1], point1[2])
        line_segs.drawTo(point2[0], point2[1], point2[2])

        # 创建线段的 NodePath
        line_nodepath = NodePath(line_segs.create())

        # 设置线段颜色为红色
        line_nodepath.setColor(Vec4(1, 0, 0, 1))

        # 两个点的坐标
        point1 = Point3(point1[0], point1[1], point1[2])
        point2 = Point3(point2[0], point2[1], point2[2])


        # 计算圆柱体的高度和半径
        height = (point2 - point1).length()
        radius = 0.5

        # 创建圆柱体
        #self.create_cylinder(link_id,point1, point2, radius)


        

        # create a link
        node = self.render.attachNewNode(f"link{link_id}")
        # 将线段添加到场景中
        line_nodepath.reparentTo(node)
        
        
        # add texture
        line_nodepath.setTextureOff(1)
        line_nodepath.setTexture(self.tex,1)
        #line_nodepath.setScale(*scale)
        
        
        node.setPos(self.render, *joint_1)
        if rot is not None:
            node.setQuat(self.render, pc.Quat(*rot[[3,0,1,2]].tolist()))
        return node




     

    def create_cylinder(self, point1, point2, radius):
        # 创建顶点数据格式
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData('cylinder_data', format, Geom.UHStatic)

        # 创建顶点写入器
        vertex_writer = GeomVertexWriter(vdata, 'vertex')

        # 创建圆柱体的顶点
        for angle in range(0, 360, 10):
            radian = angle * (3.1415926535 / 180.0)
            x = radius * 1.0 * cos(radian)
            y = radius * 1.0 * sin(radian)
            vertex_writer.addData3(x, y, 0)

        # 创建几何体
        tris = GeomTrifans(Geom.UHStatic)
        tris.addConsecutiveVertices(0, 36)
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        # 创建节点
        node = GeomNode('cylinder_node')
        node.addGeom(geom)

        # 创建NodePath对象并添加到场景中
        cylinder = self.render.attachNewNode(node)
        cylinder.lookAt(point2)
        cylinder.setPos(point1 + (point2 - point1) * 0.5)

        
    
    def receive_space(self):
        self.update_flag = not self.update_flag
        
    def create_texture(self, color, name):
        img = pc.PNMImage(32,32)
        img.fill(*color[:3])
        img.alphaFill(color[3])
        tex = pc.Texture(name)
        tex.load(img)
        return tex
        
    def load_ground(self):
        self.ground = self.loader.loadModel("material/GroundScene.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 1, 100)
        self.ground.setTexScale(pc.TextureStage.getDefault(), 50, 50)
        self.ground.setPos(0, -1, 0)
        
    def setupCameraLight(self):
        # create a orbiting camera
        self.cameractrl = CameraCtrl(self, self.cam)
        self.cameraRefNode = self.camera # pc.NodePath('camera holder')
        self.cameraRefNode.setPos(0,0,0)
        self.cameraRefNode.setHpr(0,0,0)
        self.cameraRefNode.reparentTo(self.render)
        
        self.accept("v", self.bufferViewer.toggleEnable)

        self.d_lights = []
        # Create Ambient Light
        ambientLight = pc.AmbientLight('ambientLight')
        ambientLight.setColor((0.3, 0.3, 0.3, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)
        
        # Directional light 01
        directionalLight = pc.DirectionalLight('directionalLight1')
        directionalLight.setColor((0.4, 0.4, 0.4, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setPos(10, 10, 10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)
        
        # Directional light 02
        directionalLight = pc.DirectionalLight('directionalLight2')
        directionalLight.setColor((0.4, 0.4, 0.4, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setPos(-10, 10, 10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)
        
        
        # Directional light 03
        directionalLight = pc.DirectionalLight('directionalLight3')
        directionalLight.setColorTemperature(6500)        
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setPos(0, 20, -10)
        directionalLightNP.lookAt((0, 0, 0), (0, 1, 0))

        directionalLightNP.wrtReparentTo(self.cameraRefNode)
        directionalLight.setShadowCaster(True, 2048, 2048)
        directionalLight.getLens().setFilmSize((10,10))
        directionalLight.getLens().setNearFar(0.1,300)
        
        self.render.setLight(directionalLightNP)
        self.d_lights.append(directionalLightNP)

        self.render.setShaderAuto(True)

    
    def create_joint(self, link_id, position, end_effector=False):
        # create a joint
        box = self.loader.loadModel("material/GroundScene.egg")
        node = self.render.attachNewNode(f"joint{link_id}")
        
        box.reparentTo(node)
        
        #print(position)
        # add texture
        box.setTextureOff(1)
        if end_effector:
            tex = self.create_texture([0,1,0,1], f"joint{link_id}_tex")
            box.setTexture(tex, 1)
        box.setScale(0.10,0.08,0.08)

        node.setPos(self.render, *position)

        return node
    
    def create_link(self, link_id, position, scale, rot):
        # create a link
        box = self.loader.loadModel("material/GroundScene.egg")
        node = self.render.attachNewNode(f"link{link_id}")
        box.reparentTo(node)
        
        # add texture
        box.setTextureOff(1)
        box.setTexture(self.tex,1)
        box.setScale(*scale)
        
        node.setPos(self.render, *position)
        if rot is not None:
            node.setQuat(self.render, pc.Quat(*rot[[3,0,1,2]].tolist()))
        return node
    
    def show_axis_frame(self):
        pose = [ [1,0,0], [0,1,0], [0,0,1] ]
        color = [ [1,0,0,1], [0,1,0,1], [0,0,1,1] ]
        for i in range(3):
            box = self.loader.loadModel("material/GroundScene.egg")
            box.setScale(0.1, 0.1, 0.1)
            box.setPos(*pose[i])
            tex = self.create_texture(color[i], f"frame{i}")
            box.setTextureOff(1)
            box.setTexture(tex,1)
            box.reparentTo(self.render)
    
    
    def update(self, task):
        if self.update_func and self.update_flag:
            self.update_func(self)
        return task.cont
    
    def get_joint_positions(self):
        pos = [joint.getPos(self.render) for joint in self.joints]
        return np.concatenate([pos], axis=0)
    
    def get_joint_orientations(self):
        quat = [joint.getQuat(self.render) for joint in self.joints]
        return np.concatenate([quat], axis=0)[..., [1,2,3,0]]
    
    def get_joint_position_by_name(self, name):
        pos = self.joints[self.name2idx[name]].getPos(self.render)
        return np.array(pos)
    
    def get_joint_orientation_by_name(self, name):
        quat = self.joints[self.name2idx[name]].getQuat(self.render)
        return np.array(quat)[..., [1,2,3,0]]
    
    def set_joint_position_by_name(self, name, pos):
        self.joints[self.name2idx[name]].setPos(self.render, *pos)
    
    def set_joint_orientation_by_name(self, name, quat):
        self.joints[self.name2idx[name]].setQuat(self.render, pc.Quat(*quat[...,[3,0,1,2]].tolist()))
    
    def set_joint_position_orientation(self, link_name, pos, quat):
        if not link_name in self.name2idx:
            return
        self.joints[self.name2idx[link_name]].setPos(self.render, *pos.tolist())
        self.joints[self.name2idx[link_name]].setQuat(self.render, pc.Quat(*quat[...,[3,0,1,2]].tolist()))
    
    def show_pose(self, joint_name_list, joint_positions, joint_orientations):
        length = len(joint_name_list)
        assert joint_positions.shape == (length, 3)
        assert joint_orientations.shape == (length, 4)
        
        for i in range(length):
            self.set_joint_position_orientation(joint_name_list[i], joint_positions[i], joint_orientations[i])
    def show_rest_pose(self, joint_name, joint_parent, joint_offset):
        length = len(joint_name)
        joint_positions = np.zeros((length, 3), dtype=np.float64)
        joint_orientations = np.zeros((length, 4), dtype=np.float64)
        for i in range(length):
            if joint_parent[i] == -1:
                joint_positions[i] = joint_offset[i]
            else:
                joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i]
            joint_orientations[i, 3] = 1.0
            self.set_joint_position_orientation(joint_name[i], joint_positions[i], joint_orientations[i])

    def get_meta_data(self):
        return self.joint_name, self.parent_index, self.init_joint_pos
    
    def move_marker(self, marker, x, y):
        
        if not self.update_marker_func:
            return
        
        y_axis = self.cameractrl._locked_info[2]
        z_axis = self.cameractrl.position - self.cameractrl.center
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        pos = np.array(marker.getPos(self.render))
        pos += x_axis * x + y_axis * y
        marker.setPos(self.render, *pos.tolist())
        self.update_marker_func(self)
    
    def create_marker(self, pos, color):
        self.update_marker_func = None
        marker = self.loader.loadModel("material/GroundScene.egg")
        marker.setScale(0.05,0.05,0.05)
        marker.setPos(*pos)
        tex = self.create_texture(color, "marker")
        marker.setTextureOff(1)
        marker.setTexture(tex,1)
        
        marker.wrtReparentTo(self.render)
        
        self.accept('w', self.move_marker, [marker, 0, 0.05])
        self.accept('s', self.move_marker, [marker, 0, -0.05])
        self.accept('a', self.move_marker, [marker, -0.05, 0])
        self.accept('d', self.move_marker, [marker, 0.05, 0])
        
        self.accept('w-repeat', self.move_marker, [marker, 0, 0.05])
        self.accept('s-repeat', self.move_marker, [marker, 0, -0.05])
        self.accept('a-repeat', self.move_marker, [marker, -0.05, 0])
        self.accept('d-repeat', self.move_marker, [marker, 0.05, 0])
        return marker
    
    def create_marker2(self, pos, color):
        self.update_marker_func = None
        marker = self.loader.loadModel("material/GroundScene.egg")
        marker.setScale(0.05,0.05,0.05)
        marker.setPos(*pos)
        tex = self.create_texture(color, "marker")
        marker.setTextureOff(1)
        marker.setTexture(tex,1)
        
        marker.wrtReparentTo(self.render)
        
        self.accept('arrow_up', self.move_marker, [marker, 0, 0.05])
        self.accept('arrow_down', self.move_marker, [marker, 0, -0.05])
        self.accept('arrow_left', self.move_marker, [marker, -0.05, 0])
        self.accept('arrow_right', self.move_marker, [marker, 0.05, 0])
        
        self.accept('arrow_up-repeat', self.move_marker, [marker, 0, 0.05])
        self.accept('arrow_down-repeat', self.move_marker, [marker, 0, -0.05])
        self.accept('arrow_left-repeat', self.move_marker, [marker, -0.05, 0])
        self.accept('arrow_right-repeat', self.move_marker, [marker, 0.05, 0])
        return marker
