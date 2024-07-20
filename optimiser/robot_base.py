# Copyright (C) 2024 University of Leeds
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import mujoco
import rospy
import numpy as np
from copy import deepcopy
from std_msgs.msg import Float64MultiArray
from controller_manager import ControllerManager
from sensor_msgs.msg import JointState


class RobotBase:
    def __init__(self, model, data, robot_name, joint_names, end_effector_name):
        self.model = model
        self.data = data
        self.name = robot_name
        self.joint_names = joint_names
        self.end_effector_name = end_effector_name
        self.real_robot_set_up = False
        self.sub = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        
    def joint_values_callback(self, message):
        self.joint_angles = message.position[:7]
        
    def set_arm_configuration(self, config):
        for i, joint_name in enumerate(self.joint_names):
            self.data.joint(joint_name).qpos = config[i]
        mujoco.mj_forward(self.model, self.data)

    def setup_trajectory_controller(self):
        self.controller_manager = ControllerManager()
        self.controller_manager.switch_to('effort_group_position_controller')
        self.joint_positions_publisher = rospy.Publisher('/effort_group_position_controller/command', Float64MultiArray, queue_size = 1)
        self.real_robot_set_up = True

    def execute_position_trajectory(self, trajectory):
        for control in trajectory:
            self.execute_control(control)

    def execute_control(self, control):
        rate = rospy.Rate(1000)
        positions = Float64MultiArray()
        positions.data = control
        for i in range(30):
            self.joint_positions_publisher.publish(positions)
            rate.sleep()

        # while True:
        #     distance = np.linalg.norm(np.array(control) - np.array(self.joint_angles))
        #     print(distance)
        #     if distance < 0.05:
        #         break
            
        #     self.joint_positions_publisher.publish(positions)
        #     rate.sleep()

    @property
    def arm_configuration(self):
        arm_configuration = []

        for joint_name in self.joint_names:
            joint_position = self.data.joint(joint_name).qpos[0]
            arm_configuration.append(joint_position)

        return np.array(arm_configuration)

    @property
    def joint_velocities(self):
        joint_velocities = []

        for joint_name in self.joint_names:
            joint_velocity = self.data.joint(joint_name).qvel[0]
            joint_velocities.append(joint_velocity)

        return np.array(joint_velocities)

    @property
    def end_effector_position(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_point')
        return deepcopy(np.array(self.data.site_xpos[site_id]))

    @property
    def end_effector_orientation(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_point')
        matrix = self.data.site_xmat[site_id]
        quat = np.array([0.0] * 4)
        mujoco.mju_mat2Quat(quat, matrix)
        return quat

    @property
    def rotational_jacobian(self):
        jacobian_shape = (3, self.model.nv)
        rotational_jacobian = np.zeros(jacobian_shape, dtype=np.float64)
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.end_effector_name)
        mujoco.mj_jacBody(self.model, self.data, None, rotational_jacobian, hand_id)
        return deepcopy([part[:7] for part in rotational_jacobian])

    @property
    def translational_jacobian(self):
        jacobian_shape = (3, self.model.nv)
        translational_jacobian = np.zeros(jacobian_shape, dtype=np.float64)
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.end_effector_name)
        mujoco.mj_jacBody(self.model, self.data, translational_jacobian, None, hand_id)
        return deepcopy([part[:7] for part in translational_jacobian])

    def set_arm_controls(self, controls):
        for i, joint_name in enumerate(self.joint_names):
            gravity_compensation = self.data.joint(joint_name).qfrc_bias
            self.data.actuator(joint_name).ctrl = gravity_compensation + controls[i]
