from abc import ABC
from abc import abstractmethod
import time
import math
import numpy as np
import panda_ik_pump
import panda_ik_hand

class PandaKinematics(ABC):
    @abstractmethod
    def solver(self, initial_joint_positions, position, orientation_quat, q7):
        pass

    def ik(self, initial_joint_positions, position, orientation_quat):
        all_valid_solutions = []
        angles = np.linspace(-2.89, 2.89, 50)

        for angle in angles:
            q7 = angle
            initial_joint_positions[-1] = q7
            solutions = self.solver(position, orientation_quat, q7, initial_joint_positions)

            for solution in solutions:
                if all([not math.isnan(solution[i]) for i in range(7)]):
                    all_valid_solutions.append(solution)

        return all_valid_solutions

class PandaWithHandKinematics(PandaKinematics):
    def solver(self, position, orientation_quat, q7, initial_joint_positions):
        return panda_ik_hand.franka_IK(position, orientation_quat, q7, initial_joint_positions)

class PandaWithPumpKinematics(PandaKinematics):
    def solver(self, position, orientation_quat, q7, initial_joint_positions):
        return panda_ik_pump.franka_IK(position, orientation_quat, q7, initial_joint_positions)

    def ik(self, initial_joint_positions, position, orientation_quat):
        solutions = super().ik(initial_joint_positions, position, orientation_quat)

        for i in range(len(solutions)):
            # When not using the hand, we need to remove 45 degrees from the
            # last joint. See discussion in README of the kinematics repo.
            solutions[i][-1] -= 0.785398

        return solutions


if __name__ == '__main__':
    kinematics = PandaWithHandKinematics()
    position = np.array([0.53, 0.07, 0.31])
    orientation_quat = np.array([ 0.0, 0.0, 0.0, 1.0]) # xyzw
    initial_joint_positions = np.array([0, 0, 0, 0, 0, 0, 0])

    start = time.time()
    print(kinematics.ik(initial_joint_positions, position, orientation_quat))
    end = time.time()
    elapsed_time = end - start
    print(f'Solution found in: {elapsed_time:.5f} seconds.')
