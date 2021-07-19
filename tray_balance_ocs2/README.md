# Mobile manipulator example

* In rviz, move the point around and right-click to send it as an EE pose
  command for the robot to follow.
* The configuration files are the `INFO` format of boost's property trees:
  <https://www.boost.org/doc/libs/1_62_0/doc/html/property_tree/parsers.html#property_tree.parsers.info_parser>
  - every class seems to have its own `loadSettings` method to load its
    particular part of the property tree
