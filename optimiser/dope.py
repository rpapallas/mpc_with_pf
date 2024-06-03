import rospy
import time
import tf
import geometry_msgs.msg

class DOPE:
    def __init__(self, transform_to_frame='panda_link0'):
        self.listener = tf.TransformListener()
        self.transform_to_frame = transform_to_frame
        self.give_up_limit = 2

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
