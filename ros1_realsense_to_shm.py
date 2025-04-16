#!/usr/bin/env python3
import rospy    # Documentation: http://docs.ros.org/en/melodic/api/rospy/html/
import time
import copy
import numpy as np
from sensor_msgs.msg import Image, JointState, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped, WrenchStamped
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import shared_memory

class RealsenseToSHM():
    def __init__(self):

        rospy.init_node('realsense_to_shm', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        self.queue_size = 1
        self.cam_K = None

        self.bridge = CvBridge()

        self.buffer_full_first = False

        # TODO these objects also have to be set in the foundationpose tracker
        self.obj_to_track = ['cup', 'stativ']
        self.obj_to_track_shm_idx = []
        self.obj_to_track_fb = []
        for i in range(len(self.obj_to_track)):
            self.obj_to_track_fb.append(np.zeros((4, 4), dtype=np.float64))

        self.shm_list = []
        # directly create the logging arrays as in there also the shared memories are created!
        self.create_logging_arrays()

        self.register_subscribers()

        self.recording_duration = 60
        self.log_now = False

    def shutdown(self):
        rospy.loginfo("Shutting down Async Model inference node")

        try:
            rospy.loginfo("Trying to close all the shared memories..")
            self.shutdown_shm()
            rospy.sleep(2)
            rospy.loginfo("Successfully closed all the shared memories..")
        except:
            rospy.loginfo("Could not close the shared memories!")

    def create_logging_arrays(self):

        self.curr_realsense_img = np.zeros((480, 640, 3),dtype=np.uint8)
        self.curr_depth_array_log = np.zeros((480, 640), dtype=np.uint16)
        self.cam_K_log = np.zeros((3, 3),dtype=np.float64)

        # now create all the objects
        current_shm_idx = 0

        self.shm_list.append(shared_memory.SharedMemory(create=True, name='img', size=self.curr_realsense_img.nbytes))
        self.shm_list[-1].buf[:] = self.curr_realsense_img.tobytes()
        self.shm_realsense_idx = current_shm_idx
        current_shm_idx += 1

        self.shm_list.append(
            shared_memory.SharedMemory(create=True, name='depth', size=self.curr_depth_array_log.nbytes))
        self.shm_list[-1].buf[:] = self.curr_depth_array_log.tobytes()
        self.shm_realsense_depth_idx = current_shm_idx
        current_shm_idx += 1

        self.shm_list.append(
            shared_memory.SharedMemory(create=True, name='imginfo', size=self.cam_K_log.nbytes))
        self.shm_list[-1].buf[:] = self.cam_K_log.tobytes()
        self.shm_realsense_info_idx = current_shm_idx
        current_shm_idx += 1

        for i in range(len(self.obj_to_track)):
            self.shm_list.append(
                shared_memory.SharedMemory(create=True, name=self.obj_to_track[i], size=self.obj_to_track_fb[i].nbytes))
            self.shm_list[-1].buf[:] = self.obj_to_track_fb[i].tobytes()
            self.obj_to_track_shm_idx.append(current_shm_idx)
            current_shm_idx += 1



    def shutdown_shm(self):
        # important function to shutdown all the shared memories:
        for i in range(len(self.shm_list)):
            self.shm_list[i].close()
            self.shm_list[i].unlink()
        for i in range(len(self.shm_list)):
            self.shm_list.pop(0)

    def _process_realsense_rgb(self, realsense_rgb):
        self.curr_realsense_img = np.frombuffer(realsense_rgb.data, dtype=np.uint8).reshape(realsense_rgb.height, realsense_rgb.width, -1)
        self.shm_list[self.shm_realsense_idx].buf[:] = self.curr_realsense_img.tobytes()

    def _process_realsense_depth(self, data):
        try:
            # Convert the depth image using the default passthrough encoding
            depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.curr_depth_array_log = np.array(depth_image, dtype=np.uint16)
            self.shm_list[self.shm_realsense_depth_idx].buf[:] = self.curr_depth_array_log.tobytes()

        except Exception as error:
            print("An exception occurred:", error)

        # print (copy.deepcopy(
        #     np.frombuffer(self.shm_list[self.obj_to_track_shm_idx[0]].buf[:], dtype=np.float64).reshape((4,4))))

    def _subscribe_realsense_info(self, msg):
        if self.cam_K is None:  # Update cam_K only once to avoid redundant updates
            self.cam_K = np.zeros((3, 3),dtype=np.float64)
            self.cam_K[:,:] = np.array(msg.K).reshape((3, 3))
            rospy.loginfo(f"Camera intrinsic matrix initialized")
            self.shm_list[self.shm_realsense_info_idx].buf[:] = self.cam_K.tobytes()
            # self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")):




    def register_subscribers(self):
        self.subscriber_list = []
        # Now define all the topics that we want to subscribe to!
        self.subscribe_realsense_rgb = rospy.Subscriber('/camera/color/image_raw', Image, self._process_realsense_rgb, queue_size=self.queue_size)
        self.subscriber_list.append(self.subscribe_realsense_rgb)
        self.subscribe_realsense_depth = rospy.Subscriber('camera/aligned_depth_to_color/image_raw', Image, self._process_realsense_depth, queue_size=self.queue_size)
        self.subscriber_list.append(self.subscribe_realsense_depth)

        self.subscribe_realsense_info = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._subscribe_realsense_info)
        self.subscriber_list.append(self.subscribe_realsense_info)

    def unregister_subscribers(self):
        for i in range(len(self.subscriber_list)):
            self.subscriber_list[i].unregister()


if __name__ == '__main__':
    RealsenseToSHM()
    rospy.spin()

