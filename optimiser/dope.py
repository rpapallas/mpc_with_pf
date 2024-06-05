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
import time
import tf


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
