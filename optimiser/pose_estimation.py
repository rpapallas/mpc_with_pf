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

import numpy as np
from quaternion_averaging import averageQuaternions
from pyquaternion import Quaternion
from PBPF.msg import particle_list
from collections import defaultdict
import rospy
import time
import tf

class PBPF:
    def __init__(self):
        self.sub = rospy.Subscriber('/rob_par_list', particle_list, self.particle_filtering_callback)
        time.sleep(2)

    def __aggregate_objects_and_particles(self, message):
        objs = defaultdict(lambda: { 
            'positions': [],
            'quats': [],
            'average_position': 0.0, 
            'average_quat': 0.0,
        })

        particles = []

        # Collect data
        for i, particle in enumerate(message.particles):
            objects = defaultdict(lambda: {
                'position': None,
                'quat': None,
            })

            for obj in particle.objects:
                pos = np.array([
                    obj.pose.position.x,
                    obj.pose.position.y,
                    obj.pose.position.z,
                ])

                quat = Quaternion(
                    w=obj.pose.orientation.w,
                    x=obj.pose.orientation.x,
                    y=obj.pose.orientation.y,
                    z=obj.pose.orientation.z,
                )

                objects[obj.name]['position'] = pos
                objects[obj.name]['quat'] = quat
                objs[obj.name]['positions'].append(pos)
                objs[obj.name]['quats'].append(quat)

            particles.append(objects)

        return objs, particles

    def __compute_averages(self, objs):
        for obj_name in objs.keys():
            avg_position = np.mean(np.array(objs[obj_name]['positions']), axis=0)
            
            quats = np.array([np.array([q.w, q.x, q.y, q.z]) for q in objs[obj_name]['quats']])
            avg_quat = averageQuaternions(quats)

            objs[obj_name]['average_position'] = avg_position
            objs[obj_name]['average_quat'] = avg_quat

        return objs
    
    def __find_closest(self, objs, particles):
        translation_distance_weight = 1.0
        rotational_distance_weight = 0.0

        closest_particle = None
        min_distance = float('inf')

        for particle in particles:
            particle_distance = 0.0
            for obj_name in particle.keys():
                position_distance = np.linalg.norm(particle[obj_name]['position'] - objs[obj_name]['average_position'])
                average_quat = objs[obj_name]['average_quat']
                quat_distance = Quaternion.absolute_distance(particle[obj_name]['quat'], average_quat)
                particle_distance += translation_distance_weight * position_distance + rotational_distance_weight * quat_distance

            if particle_distance < min_distance:
                min_distance = particle_distance
                closest_particle = particle

        return closest_particle

    def particle_filtering_callback(self, message):
        objs, particles = self.__aggregate_objects_and_particles(message)
        objs = self.__compute_averages(objs)
        self.closest_particle = self.__find_closest(objs, particles)


class DOPE:
    def __init__(self, transform_to_frame='panda_link0'):
        self.listener = tf.TransformListener()
        self.transform_to_frame = transform_to_frame
        self.give_up_limit = 5

    def lookup(self, name):
        num_of_lookups = 0

        while True:
            try:
                return self.listener.lookupTransform(self.transform_to_frame, name, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                num_of_lookups += 1
                time.sleep(0.1)
            
            if num_of_lookups > self.give_up_limit:
                return

if __name__ == '__main__':
    rospy.init_node('dope_listener')
    dope = DOPE(transform_to_frame='panda_link0')
    print(dope.lookup('Parmesan'))
