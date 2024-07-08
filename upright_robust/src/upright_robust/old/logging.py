import mobile_manipulation_central as mm


# fmt: off
ROSBAG_TOPICS = [
        "/clock",
        "--regex", "/ridgeback/(.*)",
        "--regex", "/ridgeback_velocity_controller/(.*)",
        "--regex", "/vicon/(.*)",
        "--regex", "/ur10/(.*)",
]
# fmt: on


class DataRecorder(mm.DataRecorder):
    def __init__(self, name=None, notes=None):
        super().__init__(topics=ROSBAG_TOPICS, name=name, notes=notes)
