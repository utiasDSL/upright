#!/usr/bin/env python3
"""Print location of Vicon markers near particular locations.

This is useful for e.g. ensuring the location of obstacles is correct in the
real world.
"""
import numpy as np
import rospy
from vicon_bridge.msg import Markers


# positions to check
POSITIONS = np.array([[0, 0.25, 1], [1.5, 1, 1], [-0.5, 2, 1]])

# print marker position if it is within this distance of one of the above
# positions (one per row)
RADIUS = 0.5


class ViconMarkerPrinter:
    def __init__(self):
        self.marker_sub = rospy.Subscriber("/vicon/markers", Markers, self._marker_cb)

    def _marker_cb(self, msg):
        for marker in msg.markers:
            r = marker.translation
            r = np.array([r.x, r.y, r.z]) / 1000  # convert to meters
            if (np.linalg.norm(POSITIONS - r, axis=1) < RADIUS).any():
                print(f"Marker {marker.marker_name} at position = {r}")


def main():
    rospy.init_node("vicon_marker_printer")
    printer = ViconMarkerPrinter()
    rospy.spin()


main()
