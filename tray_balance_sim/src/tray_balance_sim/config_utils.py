from pathlib import Path
import rospkg


def urdf_path(urdf_dict):
    """Resolve full URDF path from a dict of containing ROS package and relative path."""
    rospack = rospkg.RosPack()
    path = Path(rospack.get_path(urdf_dict["package"])) / urdf_dict["path"]
    return path.as_posix()
