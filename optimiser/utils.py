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


import csv
import time
import rospy
import time
import sys
from pathlib import Path
import os
import mujoco
import yaml
from discovered_optimisers import all_optimisers
from threading import Thread
from simulator import Simulator
from mujoco_viewer import MujocoViewer
from sensor_msgs.msg import JointState
import numpy
import tf
from pyquaternion import Quaternion

def relative_to_absolute(robot_position, robot_orientation, object_relative_position, object_relative_orientation):
    robot_in_world_matrix = robot_orientation.transformation_matrix
    robot_in_world_matrix[0, 3] = robot_position[0]
    robot_in_world_matrix[1, 3] = robot_position[1]
    robot_in_world_matrix[2, 3] = robot_position[2]

    object_in_robot_matrix = object_relative_orientation.transformation_matrix
    object_in_robot_matrix[0, 3] = object_relative_position[0]
    object_in_robot_matrix[1, 3] = object_relative_position[1]
    object_in_robot_matrix[2, 3] = object_relative_position[2]

    object_in_world_matrix = robot_in_world_matrix @ object_in_robot_matrix
    absolute_orientation = Quaternion(matrix=object_in_world_matrix)
    absolute_position = numpy.array([
        object_in_world_matrix[0, 3],
        object_in_world_matrix[1, 3],
        object_in_world_matrix[2, 3],
    ])

    return absolute_position, absolute_orientation

def get_real_robot_joint_angles():
    joint_angles = None
    def joint_values_callback(message):
        nonlocal joint_angles
        joint_angles = message.position

    rospy.Subscriber('/joint_states', JointState, joint_values_callback)
    while joint_angles is None and not rospy.is_shutdown():
        rospy.sleep(0.05)

    return joint_angles


def optimiser_factory(optimiser_name, simulator):
    all_optimisers_upper = {key.upper(): value for key, value in all_optimisers.items()}

    if optimiser_name.upper() in all_optimisers_upper.keys():
        return all_optimisers_upper[optimiser_name.upper()](simulator)
    else:
        available_planners = ", ".join(list(all_optimisers.keys()))
        sys.exit(f'Unrecognised optimiser name. Possible planners: {available_planners}')


def create_simulator(model_filename, Robot):
    model = load_model(model_filename)
    return Simulator(model, model_filename, Robot)


def load_model(model_filename):
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    return mujoco.MjModel.from_xml_path(f'{root_path}/models/{model_filename}')


def get_optimisation_parameters(yaml_name):
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    with open(f'{root_path}/config/{yaml_name}') as file:
        optimiser_parameters = yaml.safe_load(file)
        return optimiser_parameters


def print_optimisation_result(experiment_result):
    optimisation_result = experiment_result.optimisation_result
    print(f'Outcome: {optimisation_result.outcome.name}')
    average_rollout_time = sum(optimisation_result.rollout_times) / optimisation_result.iterations if optimisation_result.iterations > 0 else 0.0
    print(f'Average rollout time: {average_rollout_time:.3f} (in {optimisation_result.iterations} iterations)')
    print(f'Planning time: {optimisation_result.planning_time:.3f}')


def get_next_experiment_id():
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    path = root_dir / 'results' / 'next_experiment_id.txt'
    experiment_id = int(path.read_text())
    return experiment_id


def update_next_experiment_id_file():
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    path = root_dir / 'results' / 'next_experiment_id.txt'
    experiment_id = int(path.read_text())
    path.write_text(str(experiment_id + 1))


def save_data_to_file(world_name, result):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()

    experiment_id = get_next_experiment_id()

    path = root_dir / 'results' / 'results.csv'
    with open(path, 'a') as results_file:
        csv_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_ALL)
        csv_writer.writerow([experiment_id,
                             world_name,
                             result.optimiser_name,
                             result.models[-1].outcome.name,
                             [opt_result.rollout_times for opt_result in result.models],
                             sum([opt_result.planning_time for opt_result in result.models]),
                             [opt_result.iterations for opt_result in result.models],
                             ])

        update_next_experiment_id_file()


def reset_simulation(sim):
    sim.reset()
    mujoco_viewer.data = sim.data
    mujoco_viewer.model = sim.model


def infinitely_execute_trajectory_in_simulation(sim, trajectory):
    while not keyboard_interrupted:
        reset_simulation(sim)
        for arm_controls, gripper_controls in trajectory:
            while is_paused and not keyboard_interrupted:
                if keyboard_interrupted:
                    return
                continue
            sim.execute_control(arm_controls, gripper_controls)
            time.sleep(sim.timestep)
        time.sleep(1)


def update_simulator_from_real_world_state_pbpf(pbpf, simulator, trajectory_optimiser, joints=True):
    optimisation_parameters = get_optimisation_parameters(simulator.config_file_path)
    goal_object_name = optimisation_parameters['goal_object_name']

    robot_position, robot_orientation = simulator.get_object_pose('panda')
    robot_orientation = Quaternion(robot_orientation)

    other_object_names = optimisation_parameters['other_obstacle_names']
    all_object_names = [goal_object_name] + other_object_names
    closest_particle = pbpf.closest_particle

    if closest_particle is None:
        sys.exit('Error, no particle information')

    for object_name in all_object_names:
        orientation = particle['quat']
        position = particle['position']
        position, orientation = relative_to_absolute(robot_position, robot_orientation, position, orientation)
        simulator.set_object_pose(object_name, x=position[0], y=position[1], z=position[2], quaternion=orientation)
        for simulator_name in trajectory_optimiser.simulators.keys():
            trajectory_optimiser.simulators[simulator_name].set_object_pose(object_name, x=position[0], y=position[1], z=position[2], quaternion=orientation)

    if joints:
        config = get_real_robot_joint_angles()
        simulator.robot.set_arm_configuration(config)
        for simulator_name in trajectory_optimiser.simulators.keys():
            trajectory_optimiser.simulators[simulator_name].robot.set_arm_configuration(config)

    # Bring simulation to stability.
    for _ in range(100):
        simulator.robot.set_arm_controls([0.0]*7)
        simulator.step()

    for simulator_name in trajectory_optimiser.simulators.keys():
        for _ in range(50):
            trajectory_optimiser.simulators[simulator_name].robot.set_arm_controls([0.0]*7)
            trajectory_optimiser.simulators[simulator_name].step()

    simulator.save_state()

    for simulator_name in trajectory_optimiser.simulators.keys():
            trajectory_optimiser.simulators[simulator_name].save_state()

def update_simulator_from_real_world_state_dope(dope, simulator, trajectory_optimiser, joints=True):
    optimisation_parameters = get_optimisation_parameters(simulator.config_file_path)
    goal_object_name = optimisation_parameters['goal_object_name']

    robot_position, robot_orientation = simulator.get_object_pose('panda')
    robot_orientation = Quaternion(robot_orientation)

    other_object_names = optimisation_parameters['other_obstacle_names']
    all_object_names = [goal_object_name] + other_object_names
    for object_name in all_object_names:
        reading = dope.lookup(object_name)
        if reading is not None:
            position, orientation = reading
            orientation = Quaternion(w=orientation[3], x=orientation[0], y=orientation[1], z=orientation[2])
            position, orientation = relative_to_absolute(robot_position, robot_orientation, position, orientation)
            simulator.set_object_pose(object_name, x=position[0], y=position[1], z=position[2], quaternion=orientation)
            for simulator_name in trajectory_optimiser.simulators.keys():
                trajectory_optimiser.simulators[simulator_name].set_object_pose(object_name, x=position[0], y=position[1], z=position[2], quaternion=orientation)

    if joints:
        config = get_real_robot_joint_angles()
        simulator.robot.set_arm_configuration(config)
        for simulator_name in trajectory_optimiser.simulators.keys():
            trajectory_optimiser.simulators[simulator_name].robot.set_arm_configuration(config)

    # Bring simulation to stability.
    for _ in range(50):
        simulator.robot.set_arm_controls([0.0]*7)
        simulator.step()

    for simulator_name in trajectory_optimiser.simulators.keys():
        for _ in range(50):
            trajectory_optimiser.simulators[simulator_name].robot.set_arm_controls([0.0]*7)
            trajectory_optimiser.simulators[simulator_name].step()

    simulator.save_state()

    for simulator_name in trajectory_optimiser.simulators.keys():
        trajectory_optimiser.simulators[simulator_name].save_state()

def visualise_real_world(dope, simulator, trajectory_optimiser):
    global keyboard_interrupted, mujoco_viewer, sim
    sim = simulator
    keyboard_interrupted = False
    update_simulator_from_real_world_state_dope(dope, sim, trajectory_optimiser)

    pose = simulator.get_object_pose('Parmesan')
    print('Pose of Parmesan:', pose)

    mujoco_viewer = MujocoViewer(sim.model, sim.data, width=700,
                                 height=500, title=f'Solving',
                                 hide_menus=True)

    try:
        while mujoco_viewer.is_alive:
            mujoco_viewer.render()
    except KeyboardInterrupt:
        pass

    keyboard_interrupted = True
    mujoco_viewer.close()

def visualise(simulator, trajectory):
    global keyboard_interrupted, mujoco_viewer, sim
    sim = simulator
    keyboard_interrupted = False

    mujoco_viewer = MujocoViewer(sim.model, sim.data, width=700,
                                 height=500, title=f'Solving',
                                 hide_menus=True)

    main_thread = Thread(target=infinitely_execute_trajectory_in_simulation, args=(simulator, trajectory,))
    main_thread.start()

    global is_paused
    is_paused = False
    try:
        while mujoco_viewer.is_alive:
            is_paused = mujoco_viewer._paused
            mujoco_viewer.render()
    except KeyboardInterrupt:
        pass

    keyboard_interrupted = True
    mujoco_viewer.close()
    main_thread.join()

def view(simulator):
    global keyboard_interrupted, mujoco_viewer, sim
    sim = simulator
    keyboard_interrupted = False

    mujoco_viewer = MujocoViewer(sim.model, sim.data, width=700,
                                 height=500, title=f'Solving',
                                 hide_menus=True)

    global is_paused
    is_paused = False
    try:
        while mujoco_viewer.is_alive:
            is_paused = mujoco_viewer._paused
            mujoco_viewer.render()
    except KeyboardInterrupt:
        pass

    keyboard_interrupted = True
    mujoco_viewer.close()
