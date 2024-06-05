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
import sys
sys.path.insert(1, 'optimisers')
import utils
from optimisers.discovered_optimisers import all_optimisers
from panda import Panda
from dope import DOPE
import rospy
from sensor_msgs.msg import JointState


def arg_parser():
    available_planners = ", ".join(list(all_optimisers.keys()))

    # Required arguments
    parser = argparse.ArgumentParser(description='Trajectory Optimiser Demo')
    parser.add_argument('model_filename', help='file name of the MuJoCo model to load.')
    parser.add_argument('optimiser_name', help=f'provide optimiser name (options: {available_planners}).')

    # Real-world execution
    parser.add_argument('-o', '--open-loop', action='store_true', help='real-world execution open-loop.')
    parser.add_argument('-d', '--dope-mpc', action='store_true', help='real-world execution using DOPE and MPC.')
    parser.add_argument('-p', '--pf-mpc', action='store_true', help='real-world execution using particle filtering (PF) and MPC.')

    # Visualisation
    parser.add_argument('-v', '--view-solution', action='store_true', help='visualise the final trajectory.')
    parser.add_argument('-i', '--view-initial', action='store_true', help='visualise the initial trajectory.')
    parser.add_argument('--debug', action='store_true', help='run in debug mode, printing useful information to screen.')

    return parser.parse_args()

def get_real_robot_joint_angles():
    joint_angles = None
    def joint_values_callback(message):
        nonlocal joint_angles
        joint_angles = message.position

    rospy.Subscriber('/joint_states', JointState, joint_values_callback)
    while joint_angles is None and not rospy.is_shutdown():
        rospy.sleep(0.05)

    return joint_angles

def update_simulator_from_real_world_state():
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

def real_robot_open_loop_execution(experiment_result):
    optimised_traj = experiment_result.optimisation_result.best_trajectory
    position_traj = convert_to_position_traj(optimised_traj)
    simulator.robot.execute_position_trajectory(position_traj)

def real_robot_dope_mpc_execution(experiment_result):
    while True:
        optimised_traj = experiment_result.optimisation_result.best_trajectory
        position_traj = convert_to_position_traj(optimised_traj)

        for i in range(0, 50):
            if i > len(position_traj) - 1:
                return
            simulator.robot.execute_control(position_traj[i])

        update_simulator_from_real_world_state()
        experiment_result = trajectory_optimiser.optimise(optimised_traj[50:])
        utils.print_optimisation_result(experiment_result)

def real_robot_pf_mpc_execution(experiment_result):
    optimised_traj = experiment_result.optimisation_result.best_trajectory
    position_traj = convert_to_position_traj(optimised_traj)
    simulator.robot.setup_trajectory_controller()
    simulator.robot.execute_position_trajectory(position_traj)
    while True:
        optimised_traj = experiment_result.optimisation_result.best_trajectory
        position_traj = convert_to_position_traj(optimised_traj)

        for i in range(0, 50):
            if i > len(position_traj) - 1:
                return
            simulator.robot.execute_control(position_traj[i])

        update_simulator_from_real_world_state()
        experiment_result = trajectory_optimiser.optimise(optimised_traj[50:])
        utils.print_optimisation_result(experiment_result)

if __name__ == '__main__':
    args = arg_parser()
    logging_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(level=logging_level, format='%(message)s')

    # Create simulator from model_filename with a Robot class instance.
    simulator = utils.create_simulator(args.model_filename, Panda)

    if not args.open_loop and not args.dope_mpc and not args.pf_mpc:
        logging.info('Solving and then executing in simulation ...')
    else:
        rospy.init_node('real_robot_execution')
        simulator.robot.setup_trajectory_controller()
        update_simulator_from_real_world_state()

        if args.open_loop:
            logging.info('Solving and then executing in real-world using open-loop control...')
        elif args.dope_mpc:
            logging.info('Solving and then executing in real-world using DOPE and MPC ...')
        elif args.pf_mpc:
            logging.info('Solving and then executing in real-world using particle filtering and MPC ...')

    # Factory for generating an optimiser from the optimiser_name using the
    # `simulator` object as the base simulator for the optimisation. Optimiser will
    # create copies of the `simulator` and will not change the state of that
    # specific object during optimisation.
    trajectory_optimiser = utils.optimiser_factory(args.optimiser_name, simulator)

    if args.view_initial:
        logging.info('Just visualising initial trajectory...')
        utils.visualise(simulator, trajectory_optimiser.initial_trajectory)
    else:
        initial_trajectory = trajectory_optimiser.initial_trajectory
        experiment_result = trajectory_optimiser.optimise(initial_trajectory)
        utils.print_optimisation_result(experiment_result)

        if args.view_solution:
            logging.info('Visualing the result in simulation...')
            utils.visualise(simulator, experiment_result.optimisation_result.best_trajectory)
        elif args.open_loop:
            input('Read to execute open-loop. Hit [ENTER] to start.')
            real_robot_open_loop_execution(experiment_result)
        elif args.dope_mpc:
            input('Read to execute using DOPE and MPC. Hit [ENTER] to start.')
            real_robot_dope_mpc_execution(experiment_result)
        elif args.pf_mpc:
            input('Read to execute using particle filtering and MPC. Hit [ENTER] to start.')
            real_robot_pf_mpc_execution(experiment_result)
