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
import sys
from pathlib import Path
import os
import mujoco
import yaml
from discovered_optimisers import all_optimisers
from threading import Thread
from simulator import Simulator
from mujoco_viewer import MujocoViewer


def optimiser_factory(optimiser_name, simulator):
    if optimiser_name.upper() in all_optimisers.keys():
        return all_optimisers[optimiser_name.upper()](simulator)
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


def reset_simulation():
    sim.reset()
    mujoco_viewer.data = sim.data
    mujoco_viewer.model = sim.model


def infinitely_execute_trajectory_in_simulation(trajectory):
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

def visualise(simulator, trajectory):
    global keyboard_interrupted, mujoco_viewer, sim
    sim = simulator
    keyboard_interrupted = False

    mujoco_viewer = MujocoViewer(sim.model, sim.data, width=700,
                                 height=500, title=f'Solving',
                                 hide_menus=True)

    main_thread = Thread(target=infinitely_execute_trajectory_in_simulation, args=(trajectory,))
    main_thread.start()

    global is_paused
    is_paused = False
    try:
        while mujoco_viewer.is_alive:
            is_paused = mujoco_viewer._paused
            mujoco_viewer.render()
    except KeyboardInterrupt:
        print('Quitting')

    keyboard_interrupted = True
    mujoco_viewer.close()
    main_thread.join()

