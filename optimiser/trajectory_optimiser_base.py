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


import rospy
import multiprocessing
import math
from copy import copy, deepcopy
import time
import numpy as np
from pyquaternion import Quaternion
import utils
from cost_sequence import CostSequence
from results import OptimisationOutcome, OptimisationResult
from rollout import RollOut
from states import StateSequence, ObjectState, State
from trajectory import Trajectory
from scipy.spatial.transform import Rotation


class TrajectoryOptimiserBase:
    def __init__(self, main_simulator):
        self.random_seed = time.time()
        self.optimisation_parameters = utils.get_optimisation_parameters(main_simulator.config_file_path)

        # Create independent simulators for each rollout
        self.simulators = {f'rollout_{i}': copy(main_simulator) for i in range(self.num_of_rollouts)}

        # Also have a main simulator
        self.simulators['main'] = copy(main_simulator)
        self.simulators['real'] = copy(main_simulator)
        self.initial_traj_state_sequence = None

        # All object names except the goal object name.
        self.all_obstacle_names = self.other_obstacle_names

        # All objects including the goal object
        self.all_objects_names = deepcopy(set(self.all_obstacle_names))
        self.all_objects_names.add(self.goal_object_name)

    @property
    def static_obstacle_names(self):
        return self.optimisation_parameters['static_obstacle_names']

    @property
    def other_obstacle_names(self):
        return self.optimisation_parameters['other_obstacle_names']

    @property
    def goal_object_name(self):
        return self.optimisation_parameters['goal_object_name']

    @property
    def num_of_rollouts(self):
        return self.optimisation_parameters['number_of_rollouts']

    @property
    def initial_trajectory(self):
        def get_desired_position_waypoints():
            hand_initial_x, hand_initial_y, _ = self.simulators['real'].robot.end_effector_position

            goal_object_position = self.simulators['real'].get_object_position(self.goal_object_name)
            hand_final_x = goal_object_position[0]
            hand_final_y = goal_object_position[1]

            desired_xs_1 = np.linspace(hand_initial_x, hand_final_x, num=300)[1:]
            desired_ys_1 = np.linspace(hand_initial_y, hand_final_y, num=300)[1:]

            hand_initial_x, hand_initial_y = hand_final_x, hand_final_y

            goal_region_position = self.simulators['real'].get_object_position('goal_region')
            hand_final_x = goal_region_position[0]
            hand_final_y = goal_region_position[1]

            desired_xs_2 = np.linspace(hand_initial_x, hand_final_x, num=700)[1:]
            desired_ys_2 = np.linspace(hand_initial_y, hand_final_y, num=700)[1:]

            desired_xs = np.concatenate([desired_xs_1, desired_xs_2])
            desired_ys = np.concatenate([desired_ys_1, desired_ys_2])

            return desired_xs, desired_ys

        def calculate_linear_joint_velocities():
            hand_current_position = self.simulators['real'].robot.end_effector_position
            delta_hand_position = desired_hand_position - hand_current_position
            hand_linear_velocity = delta_hand_position / 0.01
            return translational_jacobian_transpose @ hand_linear_velocity

        def calculate_angular_joint_velocities():
            current_hand_orientation = Quaternion(self.simulators['real'].robot.end_effector_orientation)

            desired_hand_orientation = initial_hand_orientation
            if step % 10 == 0:
                # Calculate a line along x-axis from hand position
                hand_current_position = self.simulators['real'].robot.end_effector_position
                x1, y1 = hand_current_position[:2]
                x2, y2 = (x1 + 0.5), y1
                slope_line_hand_straight = (y2 - y1) / (x2 - x1)

                # Calculate a second line from hand position to goal object
                current_goal_object_position = self.simulators['real'].get_object_position(self.goal_object_name)
                x2, y2 = current_goal_object_position[:2]
                slope_line_hand_to_object = (y2 - y1) / (x2 - x1)

                # Find the angle between the two lines, eventually giving a
                # rotaiton the hand needs to rotate towards the goal object.
                angle = abs((slope_line_hand_to_object - slope_line_hand_straight) / (
                        1 + slope_line_hand_straight * slope_line_hand_to_object))

                # Find the angle in radians, and scale it down (scaling is applied
                # to neglect small angle errors as the hand aligns with the
                # goal object and avoid overshooting).
                rad = math.atan(angle) * 0.5

                axis_of_rotation = (1.0, 0.0, 0.0)
                if y2 > y1:
                    axis_of_rotation = (-1.0, 0.0, 0.0)

                hand_rotation_towards_goal_object = Quaternion(axis=axis_of_rotation, radians=rad)
                desired_hand_orientation *= hand_rotation_towards_goal_object

            delta_quaternion = desired_hand_orientation * current_hand_orientation.inverse
            delta_rotation_radians = delta_quaternion.axis * delta_quaternion.angle
            hand_angular_velocity = delta_rotation_radians / 0.06
            return rotational_jacobian_transpose @ hand_angular_velocity

        def compute_gripper_controls():
            return np.array([0.0, 0.0])

        self.simulators['real'].reset()

        desired_hand_position = self.simulators['real'].robot.end_effector_position
        initial_hand_orientation = Quaternion(self.simulators['real'].robot.end_effector_orientation)

        initial_trajectory = Trajectory()
        xs, ys = get_desired_position_waypoints()
        for step, (desired_x, desired_y) in enumerate(zip(xs, ys)):
            translational_jacobian_transpose = np.transpose(self.simulators['real'].robot.translational_jacobian)
            rotational_jacobian_transpose = np.transpose(self.simulators['real'].robot.rotational_jacobian)

            desired_hand_position[0] = desired_x
            desired_hand_position[1] = desired_y

            joints_linear_velocities = calculate_linear_joint_velocities()
            joints_angular_velocities = calculate_angular_joint_velocities()
            arm_controls = joints_linear_velocities + joints_angular_velocities
            gripper_controls = compute_gripper_controls()

            initial_trajectory.append([arm_controls, gripper_controls])

            self.simulators['real'].robot.set_arm_controls(arm_controls)
            self.simulators['real'].robot.set_gripper_controls(gripper_controls)
            self.simulators['real'].step()

        self.simulators['real'].reset()
        return deepcopy(initial_trajectory)

    def optimise_traj(self, initial_trajectory):
        for name in self.simulators.keys():
            self.simulators[name].reset()

        rospy.loginfo('Starting optimisation...')
        rospy.loginfo(f'Parallel rollouts: {self.num_of_rollouts}')

        optimisation_outcome = OptimisationOutcome.SUCCESS
        iteration = 0
        consecutive_non_improving_iterations = 0

        optimisation_start_time = time.time()

        best_trajectory = deepcopy(initial_trajectory)
        self.initial_traj_state_sequence = self.get_state_sequence(best_trajectory, 'rollout_0')
        best_rollout = self.rollout(best_trajectory, with_simulator_name='rollout_0')
        initial_cost = best_rollout.cost

        rospy.loginfo(f'Initial cost: {initial_cost:.2f}')

        rollout_times = []
        rospy.loginfo(f'Initial check of traj took roughly: {time.time() - optimisation_start_time:.2f} seconds.')
        previous_best_rollout = deepcopy(best_rollout)

        time_limit = self.optimisation_parameters['time_limit']
        max_iterations = self.optimisation_parameters['max_iterations']
        local_minima_iterations = self.optimisation_parameters['local_minima_iterations']

        while best_rollout.distance_to_goal > self.optimisation_parameters['distance_to_goal']:
            iteration += 1
            rollout_start_time = time.time()
            noisy_rollouts = self.sample_noisy_trajectories(from_trajectory=deepcopy(best_trajectory))
            rollout_end_time = time.time()
            rollout_times.append(rollout_end_time - rollout_start_time)

            best_rollout = min(noisy_rollouts + [best_rollout], key=lambda rollout: rollout.cost)
            best_trajectory = best_rollout.trajectory

            if abs(previous_best_rollout.cost - best_rollout.cost) <= 0.01:
                consecutive_non_improving_iterations += 1
            else:
                consecutive_non_improving_iterations = 0

            previous_best_rollout = deepcopy(best_rollout)

            rospy.loginfo(f'{iteration}: Current cost: {best_rollout.cost:,.2f} (initial: {initial_cost:,.2f}, distance to goal: {best_rollout.distance_to_goal:,.2f})')
            planning_time = time.time() - optimisation_start_time

            if consecutive_non_improving_iterations >= local_minima_iterations:
                optimisation_outcome = OptimisationOutcome.FAILED_LOCAL_MINIMA
                break
            elif planning_time >= time_limit or iteration >= max_iterations:
                optimisation_outcome = OptimisationOutcome.FAILED_TIMEOUT
                break

        planning_time = time.time() - optimisation_start_time

        if len(rollout_times) == 0:
            rollout_times.append(0.0)
        
        return OptimisationResult(
            iterations=iteration,
            outcome=optimisation_outcome,
            best_trajectory=deepcopy(best_trajectory),
            rollout_times=rollout_times,
            planning_time=planning_time
        )

    def sample_noisy_trajectories(self, from_trajectory):
        if self.num_of_rollouts == 1:
            return [self.sample_noisy_trajectory('rollout_0', from_trajectory)]
        else:
            rollout_results = []
            processes = []

            parallel = multiprocessing.Process
            for i in range(self.num_of_rollouts):
                pipe_parent, pipe_child = multiprocessing.Pipe()
                process = parallel(target=self.sample_noisy_trajectory, args=(f'rollout_{i}', deepcopy(from_trajectory), pipe_child))
                process.start()
                processes.append((process, pipe_parent))

            for process, pipe in processes:
                process.join()
                rollout_results.append(pipe.recv())

            return rollout_results

    def sample_noisy_trajectory(self, sim_name, base_trajectory, pipe=None):
        # This is needed due to how multiprocessing works which will fork the
        # main process, including the seed for numpy random library and as a
        # result, if the seed is not re-generated in each process, each
        # process will  generate the same noisy trajectory.
        np.random.seed()

        for i in range(len(base_trajectory)):
            for j, joint_noise_sigma in enumerate(self.optimisation_parameters['joint_noise_sigmas']):
                joint_noise = np.random.normal(0.0, joint_noise_sigma, 1)[0]
                base_trajectory[i][0][j] += joint_noise

        rollout = self.rollout(base_trajectory, with_simulator_name=sim_name)
        if pipe:  # Multi-processing
            pipe.send(rollout)
        else:  # Single-processing
            return rollout

    def rollout(self, trajectory, with_simulator_name):
        self.simulators[with_simulator_name].reset()

        state_sequence = self.get_state_sequence(trajectory, with_simulator_name)
        last_state = state_sequence[-1]

        goal_region_position = self.simulators['real'].get_object_position('goal_region')
        goal_region_position = np.array([goal_region_position[0], goal_region_position[1]])
        
        goal_object_position = last_state.objects[self.goal_object_name].position
        goal_object_position = np.array([goal_object_position[0], goal_object_position[1]])
        distance_to_goal = np.linalg.norm(goal_object_position - goal_region_position)

        cost_sequence = self.cost_of(state_sequence)

        return RollOut(trajectory, cost_sequence, distance_to_goal)

    def get_state_sequence(self, trajectory, with_simulator_name):
        state_sequence = StateSequence()
        for arm_controls, gripper_controls in trajectory:
            self.simulators[with_simulator_name].execute_control(arm_controls, gripper_controls)
            state = self.get_state(with_simulator_name)
            state_sequence.append(state)

        self.simulators[with_simulator_name].reset()
        return state_sequence

    def cost_of(self, state_sequence):
        cost_sequence = CostSequence()

        force_threshold = self.optimisation_parameters['force_threshold']

        initial_state = state_sequence[0]
        initial_hand_z_position = initial_state.hand_position[2]

        for i, state in enumerate(state_sequence):
            cost = 0.0

            if state.robot_in_collision_with_static_obstacles:
                cost += 10_000

            cost += abs(sum(state.hand_velocity)) * 3_000

            goal_region_position = self.simulators['real'].get_object_position('goal_region')
            goal_region_position = np.array([goal_region_position[0], goal_region_position[1]])
            
            goal_object_position = state.objects[self.goal_object_name].position
            goal_object_position = np.array([goal_object_position[0], goal_object_position[1]])

            euclidean_distance_object_to_region = np.linalg.norm(goal_object_position - goal_region_position)
            cost += euclidean_distance_object_to_region * 60_000

            hand_position = state.hand_position
            hand_position = np.array([hand_position[0], hand_position[1]])
            euclidean_distance_hand_to_goal_object = np.linalg.norm(goal_object_position - hand_position)
            cost += euclidean_distance_hand_to_goal_object * 20_000

            hand_z_position = state.hand_position[2]
            cost += (hand_z_position - initial_hand_z_position) * 45_000

            #for object_name in state.objects.keys():
                #quat0 = initial_state.objects[object_name].orientation
                #q0 = Quaternion(w=quat0[0],x=quat0[1],y=quat0[2],z=quat0[3])
                #r0 = Rotation.from_quat([q0.x, q0.y, q0.z, q0.w])
                #euler0 = r0.as_euler('xyz', degrees=False)

                #quati = state.objects[object_name].orientation
                #qi = Quaternion(w=quati[0],x=quati[1],y=quati[2],z=quati[3])
                #ri = Rotation.from_quat([qi.x, qi.y, qi.z, qi.w])
                #euleri = ri.as_euler('xyz', degrees=False)

                #cost += np.sum((euler0[:2] - euleri[:2]) ** 2) * 5000

            cost_sequence.append(cost)

        return cost_sequence

    def get_state(self, sim_name):
        objects = {}
        for object_name in self.all_objects_names:
            if object_name in self.simulators[sim_name].removed_objects.keys():
                continue
            position, orientation = self.simulators[sim_name].get_object_pose(object_name)
            objects[object_name] = ObjectState(position, orientation)

        objects_forces = None
        if self.optimisation_parameters['check_forces']:
            objects_forces = self.simulators[sim_name].get_object_forces(self.all_objects_names, self.static_obstacle_names)

        robot_in_collision = self.simulators[sim_name].in_collision(self.simulators[sim_name].robot.name, self.static_obstacle_names)
        ee_velocity = self.simulators[sim_name].get_object_velocity('panda_hand')

        return State(self.simulators[sim_name].robot.arm_configuration,
                     self.simulators[sim_name].robot.joint_velocities,
                     self.simulators[sim_name].robot.end_effector_position,
                     Quaternion(self.simulators[sim_name].robot.end_effector_orientation),
                     ee_velocity,
                     objects,
                     objects_forces,
                     robot_in_collision)

