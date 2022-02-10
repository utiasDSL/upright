#!/usr/bin/env python3
import os
import numpy as np
import pinocchio
import rospkg

import IPython


rospack = rospkg.RosPack()
urdf_path = os.path.join(rospack.get_path("tray_balance_assets"), "urdf", "mm_ocs2.urdf")

root_joint = pinocchio.JointModelComposite(3)
root_joint.addJoint(pinocchio.JointModelPX())
root_joint.addJoint(pinocchio.JointModelPY())
root_joint.addJoint(pinocchio.JointModelRZ())
model = pinocchio.buildModelFromUrdf(urdf_path, root_joint)

q = np.array([2, 0, 0, 0, -0.75 * np.pi, -0.5 * np.pi, -0.75 * np.pi, -0.5 * np.pi, 0.5 * np.pi])
v = np.zeros(9)
v[:3] = [1, 0, 1]

# now consider v is actually in the base frame
C_wb = np.array([[np.cos(q[2]), -np.sin(q[2]), 0], [np.sin(q[2]), np.cos(q[2]), 0], [0, 0, 1]])

v[:3] = C_wb @ v[:3]

tool_name = "thing_tool"
tool_id = model.getBodyId(tool_name)

data = model.createData()
pinocchio.forwardKinematics(model, data, q, v)
pinocchio.updateFramePlacements(model, data)
v_ee = pinocchio.getFrameVelocity(model, data, tool_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
print(f"r = {data.oMi[-1].translation}")
print(f"v = {v_ee}")
print(f"r = {data.oMi[-1].translation}")

for frame in model.frames:
    if frame.parent == 7:
        print(frame)

IPython.embed()


