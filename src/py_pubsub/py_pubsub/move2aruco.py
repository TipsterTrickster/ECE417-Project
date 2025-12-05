import rclpy
from rclpy.node import Node
# Detection msg format from ARUCO markers
from aruco_opencv_msgs.msg import ArucoDetection
# Twist msg format to jetbot motors
from geometry_msgs.msg import Twist
import math
import numpy as np

# what is minimum distance from the ARUCO marker at which
# the jetbot should have VMIN
MINDIST = 0.10

# what is maximum angle to the ARUCO marker at which
# the jetbot should have OMAX
OMAXANGLE = 25 * math.pi/180.
OMAX = 1.00


# what is minimum angle to the ARUCO marker at which
# the jetbot should have OMIN
OMINANGLE = 5 * math.pi/180.
# if angular velocity is smaller than this, then motors are unable to move the jetbot
OMIN = 0.50


def get_2d_rot_matrix(quat):
    """
    Get 2d rotation matrix (ignore y axis, tilt) from quaternion 
    """
    w, x, y, z = quat

    # Convert quaternion to 3x3 rotation matrix
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

    # Get 2x2 matrix without Y axis
    return np.array([[rotation_matrix[0,0], rotation_matrix[0, 2]], [rotation_matrix[2,0], rotation_matrix[2, 2]]])


class GetPos(Node):
    def __init__(self):
        super().__init__('move2aruco')
        self.sub = self.create_subscription(ArucoDetection,
        '/aruco_detections',
        self.on_aruco_detection, 10)

        # Will store the origin reference marker
        self.home_marker = None

        # Will store transformation matrices from each tranformation matrix to eachother transformation matrix
        self.aruco_transformation_matrices = {}

        # No Timer
        self.timer = None


    def on_aruco_detection(self, msg):
        """ This function will be called whenever one-single message of ARUCO
        detection message is received. (Basically all the times repeatedly, as
        long as markers are being detected)"""
        # self.get_logger().info('Received: "%s"' % msg.markers)

        # Dictionary of transformations from robot to all markers in view
        transformations = {}

        for marker in msg.markers:
            self.get_logger().info('marker: "%s"' % marker.marker_id)

            # Get rotation and translation matrices
            rotation = get_2d_rot_matrix((marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z))

            translation = np.array([marker.pose.position.x, marker.pose.position.z])

            # Create Robot to Marker Transformation matrix
            # R T M
            transformation = np.vstack([
                np.hstack([rotation, translation.reshape(2, 1)]),
                np.array([0, 0, 1])
            ])

            # Set a home marker if one hasn't been set yet
            if self.home_marker == None:
                self.home_marker = marker.marker_id
            
            transformations[marker.marker_id] = transformation
            
            # Print debug information
            self.get_logger().info('Rotation: "%s"' % rotation)
            self.get_logger().info('Translation: "%s"' % translation)
            self.get_logger().info('Transformation: "%s"' % transformation)
            # self.get_logger().info('marker: "%s"' % marker.marker_id)
            # self.get_logger().info('Z pose: "%s"' % marker.pose.position.z)
            # self.get_logger().info('X pose: "%s"' % marker.pose.position.x)




        self.get_logger().info('DONE')




def main(args=None):
    rclpy.init(args=args)

    getpos = GetPos()

    rclpy.spin(getpos)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    getpos.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
