# ROS2 python Library to intract with nodes, topics, publishers, subscribers, and actions
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

def get_3d_rot_matrix(quat):
    """
    Get 3d rotation matrix (ignore y axis, tilt) from quaternion 
    """

    # Initialiation of quarternion variables
    w, x, y, z = quat

    # Convert quaternion to 3x3 rotation matrix
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

    # Get 2x2 matrix without Y axis
    # return np.array([[rotation_matrix[0,0], rotation_matrix[0, 2]], [rotation_matrix[2,0], rotation_matrix[2, 2]]])
    return rotation_matrix


def get_inverted_transformation_matrix(T, D=3):
    Tinv = np.eye(D+1)
    Tinv[:D, :D] = T[:D, :D].T
    Tinv[:D, D] = - T[:D, :D].T @ T[:D, D]
    return Tinv



class GetPos(Node):
    def __init__(self):
        super().__init__('move2aruco')

        # Create a subscriber
        self.sub = self.create_subscription(ArucoDetection,
        '/aruco_detections',
        self.on_aruco_detection, 10)

        # Will store the origin reference marker
        self.home_marker = 0

        # Will store transformation matrices from each tranformation matrix to eachother transformation matrix
        self.aruco_transformation_matrices = {}

        # Stores the transformation matrices from each marker to the home marker
        self.aruco_home_transformation_matrices = {}

        # No Timer
        self.timer = None


    def on_aruco_detection(self, msg):
        """ This function will be called whenever one-single message of ARUCO
        detection message is received. (Basically all the times repeatedly, as
        long as markers are being detected)"""
        # self.get_logger().info('Received: "%s"' % msg.markers)

        # Dictionary of transformations from robot to all markers in view
        transformations = {}
        inverse_transformations = {}

        for marker in msg.markers:
            # Displays message along with a time stamp
            # self.get_logger().info('marker: "%s"' % marker.marker_id)

            # Get rotation and translation matrices
            rotation = get_3d_rot_matrix((marker.pose.orientation.w, marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z))

            # translation = np.array([marker.pose.position.x, marker.pose.position.z])
            translation = np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])

            # Create Robot to Marker Transformation matrix
            # R T M
            transformation = np.vstack([
                np.hstack([rotation, translation.reshape(3, 1)]),
                np.array([0, 0, 0, 1])
            ])

            # Set a home marker to 0 once the marker comes into frame
            if self.home_marker == None:
                if marker.id == 0:
                    self.home_marker = 0
            
            # Add transformation to dictionary
            transformations[marker.marker_id] = transformation

            # Add inverse matrix to dictionary
            inverse_transformation = get_inverted_transformation_matrix(transformation)
            inverse_transformations[marker.marker_id] = inverse_transformation


        # Loop through normal and inverse transformations and get the transformation matrices between every pair of markers
        for t_id in transformations.keys():
            for ti_id in inverse_transformations.keys():
                if ti_id == t_id:
                    continue

                # Add relative transformation of the ArUco markers to dictionary
                relative_transformation = inverse_transformations[ti_id] @ transformations[t_id]
                self.aruco_transformation_matrices[f"{str(ti_id)}{str(t_id)}"] = relative_transformation


        # Add all home transformations directly
        if self.home_marker is not None:
            for id in self.aruco_transformation_matrices.keys():
                if id[1] == '0':
                    self.aruco_home_transformation_matrices[int(id[0])] = self.aruco_transformation_matrices[id]

        
        # Chain aruco marker transformations to get their positions relative to the home marker
        for t_id in transformations.keys():
            
            # Ignore home marker as it doesn't have a transformation to home
            if t_id == 0:
                continue
            
            # If the transformation doesn't already have a home transformation chain together some to get one
            if t_id not in self.aruco_home_transformation_matrices.keys():

                for a_id in self.aruco_transformation_matrices.keys():

                    # Get the tranformation of the current marker to another marker that has a home transformation
                    if t_id == int(a_id[0]):
                        if int(a_id[1]) in self.aruco_home_transformation_matrices.keys():
                            
                            # Chain together with a marker that has a home transformation to get this markers home transformation
                            self.aruco_home_transformation_matrices[t_id] = self.aruco_transformation_matrices[a_id] @ self.aruco_home_transformation_matrices[t_id]


        print("HOME RELATIVE LOCATION")

        # Get the robots position relative to home
        for t_id in transformations.keys():

            # If home is visible print direct transformation
            if t_id == 0:
                print(transformations[t_id])
                break
            
            # IF home is not visible multiply a visible marker's transformation with the home tranformation of that marker
            print(transformations[t_id] @ self.aruco_home_transformation_matrices[t_id])
            break
                



def main(args=None):
    # Initializes library
    rclpy.init(args=args)

    # Get the posistion
    getpos = GetPos()

    # Program continues until node is closed / shut down
    rclpy.spin(getpos)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    getpos.destroy_node()

    # Shutdown the node
    rclpy.shutdown()
if __name__ == '__main__':
    main()