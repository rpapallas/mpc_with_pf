import sys
import rospy
from controller_manager_msgs.srv import LoadController, LoadControllerRequest
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from controller_manager_msgs.srv import ListControllers, ListControllersRequest


class ControllerManager:
    def __init__(self):
        self.load_controller = rospy.ServiceProxy('controller_manager/load_controller', LoadController)
        self.load_controller.wait_for_service()

        self.switch_controller = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
        self.switch_controller.wait_for_service()
        self.active_controller = None

        self.list_controllers = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
        self.list_controllers.wait_for_service()

    @property
    def loaded_controllers(self):
        params = ListControllersRequest()
        result = self.list_controllers(params)
        loaded_controllers = [controller.name for controller in result.controller]
        return loaded_controllers
    
    def switch_to(self, controller_name):
        if controller_name not in self.loaded_controllers:
            params = LoadControllerRequest()
            params.name = controller_name
            if not self.load_controller(params):
                rospy.logerr(f'Couldn\'t load controller {controller_name!r}.')
                sys.exit(1)
            rospy.loginfo(f'Controller {controller_name!r} loaded.')

        params = SwitchControllerRequest()
        params.start_controllers = [controller_name]
        if self.active_controller:
            if self.active_controller == controller_name:
                rospy.loginfo(f'Controller {controller_name!r} already loaded.')
                return
            params.stop_controllers = [self.active_controller]
        else:
            params.stop_controllers = ['position_joint_trajectory_controller']
        params.strictness = params.STRICT
        if not self.switch_controller(params):
            rospy.logerr('Couldn\'t switch from {self.active_controller!} to {controller_name!r}.')
            sys.exit(1)
        
        rospy.loginfo(f'Controller {controller_name!r} started and is now active.')
        self.active_controller = controller_name

