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

import logging
import argparse
import time
from threading import Thread
from mujoco_viewer import MujocoViewer
import sys
sys.path.insert(1, 'optimisers')
import utils
from optimisers.discovered_optimisers import all_optimisers
from panda import Panda
from dope import DOPE
import rospy
from sensor_msgs.msg import JointState


def reset_simulation():
    simulator.reset()
    mujoco_viewer.data = simulator.data
    mujoco_viewer.model = simulator.model


def infinitely_execute_trajectory_in_simulation(trajectory):
    global traj_visualised
    traj_visualised = False
    while not keyboard_interrupted:
        reset_simulation()
        for arm_controls, gripper_controls in trajectory:
            while is_paused and not keyboard_interrupted:
                if keyboard_interrupted:
                    return
                continue
            simulator.execute_control(arm_controls, gripper_controls)
            time.sleep(simulator.timestep)
        time.sleep(1)
        break
    traj_visualised = True

def visualise(trajectory, execute_once=True):
    global keyboard_interrupted, mujoco_viewer
    global traj_visualised
    keyboard_interrupted = False

    mujoco_viewer = MujocoViewer(simulator.model, simulator.data, width=700,
                                 height=500, title=f'Solving {args.model_filename}',
                                 hide_menus=True)

    main_thread = Thread(target=infinitely_execute_trajectory_in_simulation, args=(trajectory,))
    main_thread.start()

    global is_paused
    is_paused = False
    try:
        while not traj_visualised:
            is_paused = mujoco_viewer._paused
            mujoco_viewer.render()
    except KeyboardInterrupt:
        print('Quitting')

    keyboard_interrupted = True
    mujoco_viewer.close()
    main_thread.join()


def arg_parser():
    available_planners = ", ".join(list(all_optimisers.keys()))

    parser = argparse.ArgumentParser(description='Trajectory Optimiser Demo')
    parser.add_argument('model_filename', help='file name of the MuJoCo model to load.')
    parser.add_argument('optimiser_name', help=f'provide optimiser name (options: {available_planners}).')
    parser.add_argument('-s', '--save', action='store_true', help='save results and optimised trajectory to disk.')
    parser.add_argument('-v', '--view-solution', action='store_true', help='visualise the final trajectory.')
    parser.add_argument('-i', '--view-initial', action='store_true', help='visualise the initial trajectory.')
    parser.add_argument('--debug', action='store_true', help='run in debug mode, printing useful information to screen.')

    return parser.parse_args()

def get_real_robot_joint_angles():
    joint_angles = None
    def joint_values_callback(message):
        nonlocal joint_angles
        joint_angles = message.position

    subscriber = rospy.Subscriber('/joint_states', JointState, joint_values_callback)
    while joint_angles is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    return joint_angles

def update_real_world_state():
    dope = DOPE()
    optimisation_parameters = utils.get_optimisation_parameters(simulator.config_file_path)
    goal_object_name = optimisation_parameters['goal_object_name']
    position, orientation = dope.lookup(goal_object_name)
    simulator.set_object_pose(goal_object_name, x=position[0], y=position[1], z=position[2], quaternion=orientation)
    config = get_real_robot_joint_angles()
    simulator.robot.set_arm_configuration(config)
    simulator.save_state()

def convert_to_position_traj(traj):
    simulator.reset()
    position_traj = []
    for arm_controls, gripper_controls in traj:
        simulator.execute_control(arm_controls, gripper_controls)
        position_traj.append(simulator.robot.arm_configuration)
    return position_traj

if __name__ == '__main__':
    args = arg_parser()
    rospy.init_node('mpc_demo')

    keyboard_interrupted, mujoco_viewer = None, None

    logging_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(level=logging_level, format='%(message)s')

    # Create simulator from model_filename with a Robot class instance.
    simulator = utils.create_simulator(args.model_filename, Panda)

    update_real_world_state()

    # Factory for generating an optimiser from the optimiser_name using the
    # `simulator` object as the base simulator for the optimisation. Optimiser will
    # create copies of the `simulator` and will not change the state of that
    # specific object during optimisation.
    trajectory_optimiser = utils.optimiser_factory(args.optimiser_name, simulator)

    if args.view_initial:
        visualise(trajectory_optimiser.initial_trajectory)
    else:
        initial_trajectory = trajectory_optimiser.initial_trajectory
        experiment_result = trajectory_optimiser.optimise(initial_trajectory)
        utils.print_optimisation_result(experiment_result)

        if args.save:
            utils.save_data_to_file(args.model_filename, experiment_result)
        if args.view_solution:
            visualise(experiment_result.optimisation_result.best_trajectory)

        input('Will now execute on real robot, hit ENTER')
        optimised_traj = experiment_result.optimisation_result.best_trajectory
        position_traj = convert_to_position_traj(optimised_traj)
        simulator.robot.setup_trajectory_controller()
        simulator.robot.execute_position_trajectory(position_traj)
