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

from discovered_optimisers import register_optimiser
from copy import deepcopy
from trajectory_optimiser_base import TrajectoryOptimiserBase
from results import Result
from cost_sequence import CostSequence
from rollout import RollOut
import random
import numpy as np
import multiprocessing


class PF_Base(TrajectoryOptimiserBase):
    def __init__(self, main_simulator):
        # self.sub = rospy.Subscriber('/rob_par_list', particle_list, self.particle_filtering_callback)

        # Fanout is the number of particles that will be considered for every
        # noisy trajectory. If we have k noisy trajectories, then each noisy
        # trajectory will be evaluated over fanout_number particles.
        self.fanout_number = 5

        super().__init__(main_simulator)

    def particle_filtering_callback(self, message):
        self.particles = message.particles
        random.shuffle(self.particles)

    def sample_noisy_trajectories(self, from_trajectory):
        rollout_results = []
        processes = []

        parallel = multiprocessing.Process
        for i in range(0, self.num_of_rollouts, self.fanout_number):
            pipe_parent, pipe_child = multiprocessing.Pipe()
            process = parallel(target=self.sample_noisy_trajectory, args=(f'rollout_{i}', deepcopy(from_trajectory), pipe_child))
            process.start()
            processes.append((process, pipe_parent))

        for process, pipe in processes:
            process.join()
            rollout = pipe.recv()
            rollout_results.append(rollout)

        return rollout_results

    def rollout_fanout(self, simulator_index, trajectory, pipe_child):
        # This is needed due to how multiprocessing works which will fork the
        # main process, including the seed for numpy random library and as a
        # result, if the seed is not re-generated in each process, each
        # process will  generate the same noisy trajectory.
        np.random.seed()

        def relative_to_absolute(robot_position, robot_orientation, object_relative_pose):
            object_relative_position = (object_relative_pose.position.x, object_relative_pose.position.y, object_relative_pose.position.z)
            object_relative_orientation = (object_relative_pose.orientation.x, object_relative_pose.orientation.y, object_relative_pose.orientation.z, object_relative_pose.orientation.w)

            robot_transform = tf.transformations.concatenate_matrices(
                tf.transformations.translation_matrix(robot_position),
                tf.transformations.quaternion_matrix(robot_orientation)
            )

            object_relative_transform = tf.transformations.concatenate_matrices(
                tf.transformations.translation_matrix(object_relative_position),
                tf.transformations.quaternion_matrix(object_relative_orientation)
            )

            absolute_transform = tf.transformations.concatenate_matrices(robot_transform, object_relative_transform)
            absolute_position = tf.transformations.translation_from_matrix(absolute_transform)
            absolute_orientation = tf.transformations.quaternion_from_matrix(absolute_transform)

            return absolute_position, absolute_orientation

        simulator_name = f'rollout_{simulator_index}'

        # particle = self.particles[simulator_index]
        # robot_position, robot_orientation = self.simulators[simulator_name].get_object_pose('panda')
        # for obj in particle.objects:
        #     object_name = obj.name
        #     object_relative_pose = obj.pose
        #     absolute_position, absolute_orientation = relative_to_absolute(robot_position, robot_orientation, object_relative_pose)
        #     x = absolute_position[0]
        #     y = absolute_position[1]
        #     z = absolute_position[2]
        #     self.simulators[simulator_name].set_object_pose(object_name=object_name, x=x, y=y, z=z, quaternion=absolute_orientation)
        # self.simulators[simulator_name].save_state()

        state_sequence = self.get_state_sequence(trajectory, simulator_name)
        last_state = state_sequence[-1]

        goal_region_position = self.simulators['main'].get_object_position('goal_region')
        goal_region_position = np.array([goal_region_position[0], goal_region_position[1]])
        
        goal_object_position = last_state.objects[self.goal_object_name].position
        goal_object_position = np.array([goal_object_position[0], goal_object_position[1]])
        distance_to_goal = np.linalg.norm(goal_object_position - goal_region_position)

        cost_sequence = self.cost_of(state_sequence)
        rollout = RollOut(trajectory, cost_sequence, distance_to_goal)
        pipe_child.send(rollout)

    def rollout(self, trajectory, with_simulator_name):
        # This is needed due to how multiprocessing works which will fork the
        # main process, including the seed for numpy random library and as a
        # result, if the seed is not re-generated in each process, each
        # process will  generate the same noisy trajectory.
        np.random.seed()

        start_simulator_index = int(with_simulator_name.split('_')[1])

        processes = []
        parallel = multiprocessing.Process
        for i in range(self.fanout_number):
            pipe_parent, pipe_child = multiprocessing.Pipe()
            process = parallel(target=self.rollout_fanout, args=(start_simulator_index + i, trajectory, pipe_child))
            process.start()
            processes.append((process, pipe_parent))

        rollouts = []
        for process, pipe in processes:
            process.join()
            rollout = pipe.recv()
            rollouts.append(rollout)

        rollout = self.aggregate(rollouts)
        return rollout

    def optimise(self, trajectory):
        start_arm_configuration = self.simulators['main'].robot.arm_configuration
        start_hand_configuration = self.simulators['main'].robot.end_effector_configuration

        result = self.optimise_traj(trajectory)

        return Result(self.__class__.__name__, 
                      self.simulators['main'].model_filename,
                      start_arm_configuration, 
                      start_hand_configuration,
                      result)

@register_optimiser
class PF_WorstAggregation(PF_Base):
    def aggregate(self, rollouts):
        worst_rollout = rollouts[0]
        for i in range(1, self.fanout_number):
            current_rollout = rollouts[i]
            if current_rollout.cost > worst_rollout.cost:
                worst_rollout = current_rollout

        return worst_rollout

@register_optimiser
class PF_AverageAggregation(PF_Base):
    def aggregate(self, rollouts):
        total_cost = 0.0
        for i in range(self.fanout_number):
            current_rollout = rollouts[i]
            total_cost += current_rollout.cost_sequence.total

        best_rollout = rollouts[0]
        for i in range(1, self.fanout_number):
            current_rollout = rollouts[i]
            if current_rollout.cost < best_rollout.cost:
                best_rollout = current_rollout

        average_cost = total_cost / self.fanout_number
        # Here we just "faking" the cost sequence to the average cost.
        best_rollout.cost_sequence = CostSequence([average_cost])
        return best_rollout

