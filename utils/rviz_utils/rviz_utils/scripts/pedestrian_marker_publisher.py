#!/usr/bin/env python3

import math

import rclpy
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_rclpy_mixins.spin import spin_node
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from arena_people_msgs.msg import Pedestrians, Pedestrian, Skeleton
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA

from std_msgs.msg import ColorRGBA, Header
from tf_transformations import quaternion_from_euler, quaternion_multiply
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

class PedestrianMarkerPublisher(Node):
    """
    Node that converts arena_people_msgs/Pedestrians to visualization_msgs/MarkerArray
    for better pedestrian visualization in RViz2
    """

    def __init__(self):
        super().__init__("pedestrian_marker_publisher")

        # Get namespace parameter for multi-environment support
        namespace = self.get_namespace()

        # QoS Settings for arena_peds topic
        pedestrians_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # QoS Settings for marker output
        marker_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Build namespaced topic names
        clean_ns = namespace.strip("/")
        pedestrians_topic = f"/{clean_ns}/arena_peds" if clean_ns else "/arena_peds"
        marker_topic = f"/{clean_ns}/pedestrian_markers" if clean_ns else "/pedestrian_markers"

        # Subscriber to arena_peds topic
        self.pedestrians_subscriber = self.create_subscription(Pedestrians, pedestrians_topic, self.pedestrians_callback, pedestrians_qos)

        # Publisher for pedestrian markers
        self.marker_publisher = self.create_publisher(MarkerArray, marker_topic, marker_qos)

        # Parameters for visualization
        self.declare_parameter("body_height", 1.6)  # Height of pedestrian body
        self.declare_parameter("body_radius", 0.25)  # Radius of pedestrian body
        self.declare_parameter("head_radius", 0.15)  # Radius of pedestrian head
        self.declare_parameter("arrow_length", 0.6)  # Length of direction arrow
        self.declare_parameter("show_labels", True)  # Show name labels
        self.declare_parameter("show_velocity_arrows", True)  # Show velocity arrows
        self.declare_parameter("show_orientation_arrows", True)  # Show orientation arrows
        self.declare_parameter(
            "mesh_resource",
            "package://pal_gazebo_worlds/models/citizen_extras_male_03/meshes/mesh.dae",
        )
        self.declare_parameter("skeleton_poses_local", True)

        self.get_logger().info("Pedestrian Marker Publisher initialized")
        self.get_logger().info(f"Subscribing to {pedestrians_topic}, publishing to {marker_topic}")

    def pedestrians_callback(self, msg: Pedestrians):
        """Convert Pedestrians message to MarkerArray with stylized human figures"""

        marker_array = MarkerArray()
        markers = []

        # Get parameters
        body_height = self.get_parameter("body_height").value
        body_radius = self.get_parameter("body_radius").value
        arrow_length = self.get_parameter("arrow_length").value
        show_labels = self.get_parameter("show_labels").value
        show_velocity_arrows = self.get_parameter("show_velocity_arrows").value
        show_orientation_arrows = self.get_parameter("show_orientation_arrows").value

        # Clear existing markers first (important for dynamic number of people)
        delete_marker = Marker()
        delete_marker.header = msg.header
        delete_marker.action = Marker.DELETEALL
        markers.append(delete_marker)

        for pedestrian in msg.pedestrians:
            pedestrian_id = pedestrian.id

            # Determine colors based on animation state
            body_color, head_color = self._get_pedestrian_colors(pedestrian)

            # 1. Body (mesh)
            body_marker = self._create_body_marker(
                pedestrian,
                pedestrian_id,
                body_color,
                body_height,
                body_radius,
                msg.header,
            )
            markers.append(body_marker)

            # 3. Velocity Arrow (if pedestrian is moving)
            if show_velocity_arrows:
                velocity_magnitude = math.sqrt(pedestrian.twist.linear.x**2 + pedestrian.twist.linear.y**2)
                if velocity_magnitude > 0.1:  # Only show if moving fast enough
                    arrow_marker = self._create_velocity_arrow(pedestrian, pedestrian_id, arrow_length, body_height, msg.header)
                    markers.append(arrow_marker)

            # 4. Orientation Arrow (shows where pedestrian is facing)
            if show_orientation_arrows:
                orientation_marker = self._create_orientation_arrow(pedestrian, pedestrian_id, arrow_length, body_height, msg.header)
                markers.append(orientation_marker)

            # 5. Skeleton (optional)
            skeleton_local = self.get_parameter("skeleton_poses_local").value
            if hasattr(pedestrian, "skeleton") and len(pedestrian.skeleton.joint_poses) > 0:
                ns = f"ped_skeletons_{pedestrian.id}"
                # Use a large offset to keep ids unique across pedestrians
                base_id = int(pedestrian.id) * 1000
                for j_idx, joint_pose in enumerate(pedestrian.skeleton.joint_poses):
                    joint_marker = Marker()
                    joint_marker.header = msg.header
                    joint_marker.ns = ns
                    joint_marker.id = base_id + j_idx
                    joint_marker.type = Marker.SPHERE
                    joint_marker.action = Marker.ADD

                    # Decide whether joint_pose is local (relative to pedestrian) or already in world frame
                    if skeleton_local:
                        # Compose pedestrian.pose * joint_pose to put joint in world frame
                        try:
                            joint_marker.pose = compose_poses(pedestrian.pose, joint_pose)
                        except Exception:
                            # Fallback: use joint_pose directly
                            joint_marker.pose = joint_pose
                    else:
                        joint_marker.pose = joint_pose

                    joint_marker.scale = Vector3(x=0.05, y=0.05, z=0.05)

                    # Color by confidence if available; ensure alpha > 0 so it's visible
                    conf = 1.0
                    if hasattr(pedestrian.skeleton, "confidences") and j_idx < len(pedestrian.skeleton.confidences):
                        conf = float(pedestrian.skeleton.confidences[j_idx])
                    joint_marker.color = ColorRGBA(r=max(0.0, 1.0 - conf), g=min(1.0, conf), b=0.0, a=max(0.15, conf))

                    # Lifetime same as other markers (1 second)
                    joint_marker.lifetime.sec = 1

                    markers.append(joint_marker)

            # 6. Name Label
            if show_labels:
                label_marker = self._create_name_label(pedestrian, pedestrian_id, body_height, msg.header)
                markers.append(label_marker)

        marker_array.markers = markers
        self.marker_publisher.publish(marker_array)

    def _get_pedestrian_colors(self, pedestrian: Pedestrian) -> tuple[ColorRGBA, ColorRGBA]:
        """Get colors based on pedestrian animation state"""

        # Map animation states to colors
        if pedestrian.animation_state == Pedestrian.WALKING:
            body_color = ColorRGBA(r=0.2, g=0.7, b=0.2, a=1.0)  # Green
            head_color = ColorRGBA(r=0.9, g=0.7, b=0.5, a=1.0)  # Skin tone
        elif pedestrian.animation_state == Pedestrian.IDLE:
            body_color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)  # Gray
            head_color = ColorRGBA(r=0.9, g=0.7, b=0.5, a=1.0)  # Skin tone
        elif pedestrian.animation_state == Pedestrian.PANIC:
            body_color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)  # Red
            head_color = ColorRGBA(r=1.0, g=0.6, b=0.6, a=1.0)  # Light red
        elif pedestrian.animation_state == Pedestrian.SURPRISED:
            body_color = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)  # Yellow
            head_color = ColorRGBA(r=0.9, g=0.7, b=0.5, a=1.0)  # Skin tone
        elif pedestrian.animation_state == Pedestrian.CURIOUS:
            body_color = ColorRGBA(r=0.2, g=0.2, b=1.0, a=1.0)  # Blue
            head_color = ColorRGBA(r=0.9, g=0.7, b=0.5, a=1.0)  # Skin tone
        elif pedestrian.animation_state == Pedestrian.THREATENING:
            body_color = ColorRGBA(r=0.8, g=0.0, b=0.8, a=1.0)  # Purple
            head_color = ColorRGBA(r=0.9, g=0.7, b=0.5, a=1.0)  # Skin tone
        else:
            # Default colors
            body_color = ColorRGBA(r=0.3, g=0.3, b=0.8, a=1.0)  # Blue
            head_color = ColorRGBA(r=0.9, g=0.7, b=0.5, a=1.0)  # Skin tone

        return body_color, head_color

    def _create_body_marker(
        self,
        pedestrian: Pedestrian,
        pedestrian_id: int,
        color: ColorRGBA,
        body_height: float,
        body_radius: float,
        header: Header,
    ) -> Marker:
        """Create mesh marker for pedestrian body"""
        marker = Marker()
        marker.header = header
        marker.ns = "pedestrian_meshes"
        marker.id = pedestrian_id
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = self.get_parameter("mesh_resource").value
        marker.mesh_use_embedded_materials = True
        marker.action = Marker.ADD

        # Position - use pedestrian position
        marker.pose.position.x = pedestrian.pose.position.x
        marker.pose.position.y = pedestrian.pose.position.y
        marker.pose.position.z = 0.01  # Center

        # Align the mesh model's yaw with the true ped's pose (pi/2 offset)
        q_org = [
            pedestrian.pose.orientation.x,
            pedestrian.pose.orientation.y,
            pedestrian.pose.orientation.z,
            pedestrian.pose.orientation.w,
        ]
        rotation = quaternion_from_euler(0, 0, math.pi / 2)
        q = quaternion_multiply(q_org, rotation)

        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale = Vector3(x=body_height, y=body_height, z=body_height)

        # Color
        # marker.color = color

        # Lifetime
        marker.lifetime.sec = 1

        return marker

    def _create_orientation_arrow(
        self,
        pedestrian: Pedestrian,
        pedestrian_id: int,
        arrow_length: float,
        body_height: float,
        header: Header,
    ) -> Marker:
        """Create arrow marker showing pedestrian orientation"""
        marker = Marker()
        marker.header = header
        marker.ns = "pedestrian_orientation"
        marker.id = pedestrian_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Position at pedestrian location
        marker.pose.position.x = pedestrian.pose.position.x
        marker.pose.position.y = pedestrian.pose.position.y
        marker.pose.position.z = body_height / 2

        # Orientation from pedestrian orientation
        marker.pose.orientation = pedestrian.pose.orientation

        # Size
        marker.scale = Vector3(x=arrow_length, y=0.15, z=0.15)

        # Color (blue for orientation)
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)  # Semi-transparent blue

        # Lifetime
        marker.lifetime.sec = 1

        return marker

    def _create_velocity_arrow(
        self,
        pedestrian: Pedestrian,
        pedestrian_id: int,
        arrow_length: float,
        body_height: float,
        header: Header,
    ) -> Marker:
        """Create arrow marker showing pedestrian velocity direction"""
        marker = Marker()
        marker.header = header
        marker.ns = "pedestrian_velocity"
        marker.id = pedestrian_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Position at pedestrian location
        marker.pose.position.x = pedestrian.pose.position.x
        marker.pose.position.y = pedestrian.pose.position.y
        marker.pose.position.z = body_height / 2 + 0.2

        # Orientation from velocity direction
        vel_x = pedestrian.twist.linear.x
        vel_y = pedestrian.twist.linear.y
        yaw = math.atan2(vel_y, vel_x)

        marker.pose.orientation.z = math.sin(yaw / 2)
        marker.pose.orientation.w = math.cos(yaw / 2)

        # Size (arrow length proportional to speed)
        velocity_magnitude = math.sqrt(vel_x**2 + vel_y**2)
        actual_length = min(arrow_length * velocity_magnitude, arrow_length)
        marker.scale = Vector3(x=actual_length, y=0.1, z=0.1)

        # Color (bright for visibility)
        marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # Orange

        # Lifetime
        marker.lifetime.sec = 1

        return marker

    def compose_poses(parent: Pose, child: Pose) -> Pose:
        """Return parent * child (child expressed in parent's local frame -> world-frame)."""
        # parent and child use (x,y,z,w) quaternion ordering as used by tf_transformations
        parent_q = [parent.orientation.x, parent.orientation.y, parent.orientation.z, parent.orientation.w]
        child_q = [child.orientation.x, child.orientation.y, child.orientation.z, child.orientation.w]

        # Build parent transform matrix
        T_parent = quaternion_matrix(parent_q)
        T_parent[0:3, 3] = [parent.position.x, parent.position.y, parent.position.z]

        # Child position (homogeneous)
        child_pos = np.array([child.position.x, child.position.y, child.position.z, 1.0])
        world_pos = T_parent.dot(child_pos)

        # Combined orientation: qp * qc
        comb_q = quaternion_multiply(parent_q, child_q)  # returns [x,y,z,w]

        out = Pose()
        out.position = Point(x=float(world_pos[0]), y=float(world_pos[1]), z=float(world_pos[2]))
        out.orientation = Quaternion(x=comb_q[0], y=comb_q[1], z=comb_q[2], w=comb_q[3])
        return out

    def _create_name_label(
        self, pedestrian: Pedestrian, pedestrian_id: int, body_height: float, header
    ) -> Marker:
    def _create_name_label(self, pedestrian: Pedestrian, pedestrian_id: int, body_height: float, header: Header) -> Marker:
        """Create text marker with pedestrian name"""
        marker = Marker()
        marker.header = header
        marker.ns = "pedestrian_labels"
        marker.id = pedestrian_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # Position above head
        marker.pose.position.x = pedestrian.pose.position.x
        marker.pose.position.y = pedestrian.pose.position.y
        marker.pose.position.z = 0.0 + body_height + 0.5  # Ground + body + label offset
        marker.pose.orientation.w = 1.0

        # Text content - use pedestrian name
        marker.text = f"Agent {pedestrian.name}"

        # Size
        marker.scale.z = 0.3  # Text height

        # Color (white for visibility)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

        # Lifetime
        marker.lifetime.sec = 1

        return marker


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    spin_node(PedestrianMarkerPublisher())


if __name__ == "__main__":
    main()
