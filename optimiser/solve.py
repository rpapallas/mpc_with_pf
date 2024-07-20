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

import argparse
import sys
import time
import numpy as np
from copy import copy
sys.path.insert(1, 'optimisers')
import utils
from optimisers.discovered_optimisers import all_optimisers
from panda import Panda
import rospy
from pose_estimation import DOPE, PBPF
from results import OptimisationOutcome
from functools import partial

def arg_parser():
    available_planners = ", ".join(list(all_optimisers.keys()))

    # Required arguments
    parser = argparse.ArgumentParser(description='Trajectory Optimiser Demo')
    parser.add_argument('model_filename', help='file name of the MuJoCo model to load.')
    parser.add_argument('optimiser_name', help=f'provide optimiser name (options: {available_planners}).')

    # Real-world execution
    parser.add_argument('-r', '--real-world-state', action='store_true', help='real-world state visualisation.')
    parser.add_argument('-o', '--open-loop', action='store_true', help='real-world execution open-loop.')
    parser.add_argument('-d', '--dope-mpc', action='store_true', help='real-world execution using DOPE and MPC.')
    parser.add_argument('-p1', '--pf-mpc1', action='store_true', help='real-world execution using particle filtering (PF) and MPC. PF is integrated into the planning too.')
    parser.add_argument('-p2', '--pf-mpc2', action='store_true', help='real-world execution using particle filtering (PF) and MPC. PF is not integrated into the planning.')

    # Visualisation
    parser.add_argument('-v', '--view-solution', action='store_true', help='visualise the final trajectory.')
    parser.add_argument('-i', '--view-initial', action='store_true', help='visualise the initial trajectory.')
    parser.add_argument('--debug', action='store_true', help='run in debug mode, printing useful information to screen.')

    return parser.parse_args()

def convert_to_position_traj(traj, steps=None):
    simulator.reset()
    if steps is None:
        steps = len(traj)
    elif steps > len(traj):
        steps = len(traj)
        
    position_traj = []
    for i in range(steps):
        arm_controls, gripper_controls = traj[i]
        simulator.execute_control(arm_controls, gripper_controls)
        position_traj.append(simulator.robot.arm_configuration)

    for simulator_name in trajectory_optimiser.simulators.keys():
        for i in range(steps):
            arm_controls, gripper_controls = traj[i]
            trajectory_optimiser.simulators[simulator_name].execute_control(arm_controls, gripper_controls)
    
    for simulator_name in trajectory_optimiser.simulators.keys():
        trajectory_optimiser.simulators[simulator_name].save_state()

    simulator.save_state()
    return position_traj

def real_robot_open_loop_execution(experiment_result):
    optimised_traj = experiment_result.optimisation_result.best_trajectory
    position_traj = convert_to_position_traj(optimised_traj)
    trajectory_optimiser.simulators['real'].robot.execute_position_trajectory(position_traj)

def goal_object_in_goal_region(simulator, trajectory_optimiser):
    goal_object_name = trajectory_optimiser.goal_object_name

    goal_region_position = simulator.get_object_position('goal_region')
    goal_region_position = np.array([goal_region_position[0], goal_region_position[1]])
    
    goal_object_position = simulator.get_object_position(goal_object_name)
    goal_object_position = np.array([goal_object_position[0], goal_object_position[1]])
    distance_to_goal = np.linalg.norm(goal_object_position - goal_region_position)

    return distance_to_goal <= trajectory_optimiser.optimisation_parameters['distance_to_goal']

def real_robot_mpc_execution(experiment_result, update_function):
    controls_to_execute = 100
    num_controls = len(experiment_result.optimisation_result.best_trajectory)
    num_of_cycles = num_controls // controls_to_execute

    for i in range(num_of_cycles):
        optimised_traj = experiment_result.optimisation_result.best_trajectory
        position_traj = convert_to_position_traj(optimised_traj, controls_to_execute)
        for position in position_traj:
            trajectory_optimiser.simulators['real'].robot.execute_control(position)

        update_function(simulator, trajectory_optimiser)
        optimised_traj = optimised_traj[controls_to_execute:]

        experiment_result = trajectory_optimiser.optimise(optimised_traj)
       
        if experiment_result.optimisation_result.outcome != OptimisationOutcome.SUCCESS:
            print('Optimisation failed.')
            input('Press ENTER to visualise the remaining trajectory and see where the issue was')
            utils.visualise(simulator, optimised_traj)
            sys.exit('Optimisation Failed')


if __name__ == '__main__':
    args = arg_parser()
    log_level = rospy.INFO if args.debug else rospy.ERROR

    # Create simulator from model_filename with a Robot class instance.
    simulator = utils.create_simulator(args.model_filename, Panda)
    rospy.init_node('real_robot_execution', log_level=log_level)
    dope = DOPE()
    pbpf = PBPF()

    for _ in range(3000):
        simulator.robot.set_arm_controls([0.0]*7)
        simulator.step()
    simulator.save_state()

    average_particle = None

    # `simulator` object as the base simulator for the optimisation. Optimiser will
    # create copies of the `simulator` and will not change the state of that
    # specific object during optimisation.
    trajectory_optimiser = utils.optimiser_factory(args.optimiser_name, simulator)

    utils.update_simulator_from_real_world_state_dope(dope, True, simulator, trajectory_optimiser)

    if args.real_world_state:
        utils.visualise_real_world(dope, simulator, trajectory_optimiser)
    else:
        if not args.open_loop and not args.dope_mpc and not args.pf_mpc1 and not args.pf_mpc2:
            rospy.loginfo('Solving and then executing just in simulation ...')
        else:
            trajectory_optimiser.simulators['real'].robot.setup_trajectory_controller()
            if args.open_loop:
                rospy.loginfo('Solving and then executing in real-world using open-loop control...')
            elif args.dope_mpc:
                rospy.loginfo('Solving and then executing in real-world using DOPE and MPC ...')
            elif args.pf_mpc1:
                rospy.loginfo('Solving (using particle filtering in planning) and then executing in real-world using particle filtering and MPC ...')
            elif args.pf_mpc2:
                rospy.loginfo('Solving (not using particle filtering in planning) and then executing in real-world using particle filtering and MPC ...')

        if args.view_initial:
            rospy.loginfo('Just visualising initial trajectory...')
            utils.visualise(simulator, trajectory_optimiser.initial_trajectory)
        else:
            initial_trajectory = trajectory_optimiser.initial_trajectory
            experiment_result = trajectory_optimiser.optimise(initial_trajectory)
            utils.print_optimisation_result(experiment_result)

            if experiment_result.optimisation_result.outcome != OptimisationOutcome.SUCCESS:
                sys.exit('Failed')

            rospy.loginfo('Visualing the result in simulation first (no real-world execution yet). Press [ESC] in simulator window to continue ...')
            utils.visualise(simulator, experiment_result.optimisation_result.best_trajectory)

            if args.open_loop or args.dope_mpc or args.pf_mpc1 or args.pf_mpc2:
                choice = input('Ready to execute. Should I execute? [y/N]: ')
                if choice != 'y':
                    sys.exit('Exiting...')

            if args.open_loop:
                rospy.loginfo('Running open-loop')
                real_robot_open_loop_execution(experiment_result)
            elif args.dope_mpc:
                rospy.loginfo('Running with DOPE')
                update_function = partial(utils.update_simulator_from_real_world_state_dope, dope, False)
                real_robot_mpc_execution(experiment_result, update_function)
            elif args.pf_mpc1 or args.pf_mpc2:
                rospy.loginfo('Running with particle filtering')
                update_function = partial(utils.update_simulator_from_real_world_state_pbpf, pbpf, False)
                real_robot_mpc_execution(experiment_result, update_function)
