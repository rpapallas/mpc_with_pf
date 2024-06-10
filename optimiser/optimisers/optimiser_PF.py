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
from trajectory_optimiser_base import TrajectoryOptimiserBase
from results import Result
import logging
import random
import rospy
import multiprocessing


@register_optimiser
class PF(TrajectoryOptimiserBase):
    def __init__(self, main_simulator):
        self.particles_received = False
        self.sub = rospy.Subscriber('/joint_states', particle_list, self.particle_filtering_callback)

        # Fanout is the number of particles that will be considered for every
        # noisy trajectory. If we have k noisy trajectories, then each noisy
        # trajectory will be evaluated over fanout_number particles.
        self.fanout_number = 5

        super().__init__(main_simulator)

    def particle_filtering_callback(self, message):
        self.particles_received = True
        self.particles = message.particles
        random.shuffle(self.particles)

    def sample_noisy_trajectories(self, from_trajectory):
        if self.num_of_rollouts == 1:
            return [self.sample_noisy_trajectory('rollout_0', from_trajectory)]
        else:
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
                rollout_results.append(pipe.recv())

            return rollout_results

    def rollout(self, trajectory, with_simulator_name):
        rollouts = []

        def rollout(simulator_index):
            simulator_name = f'rollout_{simulator_index}'
            logging.info(simulator_name)
            self.simulators[simulator_name].reset()

            particle = self.particles[simulator_index]
            for obj in particle.objects:
                object_name = obj.name
                object_pose = obj.pose
                quat = [object_pose.orientation.x, object_pose.orientation.y, object_pose.orientation.z, object_pose.orientation.w]
                x, y, z = object_pose.position.x, object_pose.position.y, object_pose.position.z
                self.simulators[simulator_name].set_object_pose(object_name=object_name, x=x, y=y, z=z, quaternion=quat)
            self.simulators[simulator_name].save_state()

            state_sequence = StateSequence()
            state = None
            for arm_controls, gripper_controls in trajectory:
                self.simulators[simulator_name].execute_control(arm_controls, gripper_controls)
                state = self.get_state(simulator_name)
                state_sequence.append(state)

            goal_region_position = self.simulators['main'].get_object_position('goal_region')
            goal_region_position = np.array([goal_region_position[0], goal_region_position[1]])
            
            goal_object_position = state.objects[self.goal_object_name].position
            goal_object_position = np.array([goal_object_position[0], goal_object_position[1]])
            distance_to_goal = np.linalg.norm(goal_object_position - goal_region_position)

            cost_sequence = self.cost_of(state_sequence)
            rollout = RollOut(trajectory, cost_sequence, distance_to_goal), state_sequence
            rollouts.append((rollout, state_sequence))
            self.simulators[simulator_name].reset()

        start_simulator_index = int(with_simulator_name.split('_')[1])

        for i in range(self.fanout_number):
            rollout(start_simulator_index + i)

        while len(rollouts) < self.fanout_number:
            pass

        worst_rollout, worst_state_seq = rollouts[0]
        for i in range(1, self.fanout_number):
            current_rollout, current_state_seq = rollouts[i]
            current_cost = current_rollout.cost_sequence.total
            best_cost = worst_rollout.cost_sequence.total
            if current_cost > best_cost:
                worst_rollout = current_rollout
                worst_state_seq = current_state_seq

        return worst_rollout, worst_state_seq

    def optimise(self, trajectory):
        start_arm_configuration = self.simulators['main'].robot.arm_configuration
        start_hand_configuration = self.simulators['main'].robot.end_effector_configuration

        result = super().optimise(trajectory)

        return Result(self.__class__.__name__, 
                      self.simulators['main'].model_filename,
                      start_arm_configuration, 
                      start_hand_configuration,
                      result)


