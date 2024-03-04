# Calibration

The main things that need to calibrated are the tray transform with respect to
the robot's end effector and the friction coefficients between the various
objects.

## Building the tray Vicon object

The important thing to note is that it is assumed that the z-axis of the tray
Vicon object is normal to the surface of the tray. The friction coefficient
computation and the balancing constraints in the controller depend on this
assumption. To build the tray object:

1. Add at least four markers to the tray object, preferably in a non-planar
   configuration. Try to put them out of the way of the balanced objects but
   in locations that are reasonably visible to the Vicon cameras.
2. When actually constructing the object in the Vicon interface, place (but do
   not attach a marker at the desired center point. Add this to the object,
   make it so that it is half the marker's width above the object origin, and
   then remove the marker from the Vicon object in the interface. Save the object.
3. Assuming the Vicon world frame's z-axis is gravity-aligned, perform a
   zero-pose calibration of the tray resting on the flat surface to obtain the
   rotational offset between the tray's frame and the world frame. Only the
   rotation is important; the translation is arbitrary. Run
   ```
   rosrun vicon_bridge calibrate <TrayObjectName> <TrayObjectName>
   ```
   and then take the quaternion values from the parameter server.

## Calibrating the tray transform

Grip the tray with the end effector, in the position you are using for
experiments. Then use [this script](https://github.com/utiasDSL/mobile_manipulation_central/blob/main/scripts/calibration/collect_arm_calibration_data.py) to move the robot through a sequence of
configurations and collect the data. Then run the [calibration script](https://github.com/utiasDSL/mobile_manipulation_central/blob/main/scripts/calibration/calibrate_ee.py) to
compute the desired transforms and output them as a YAML file. This YAML file
is consumed by the xacro'd URDFs to accurately incorporate the tray into the
robot model. More details on this calibration routine can be found [here](https://github.com/utiasDSL/mobile_manipulation_central/blob/main/docs/calibration/calibration.pdf).

## Computing the friction coefficients

Friction coefficients are obtained by placing the tray on the (flat) ground
and stacking the desired objects on top of the tray (possibly on each other,
if computing the friction coefficient between two objects). While recording the
data by running
```
rosbag record --regex "/vicon/(.*)" -o <bag_prefix>
```
slowly tilt the tray until the object slides substantially. Then use
`upright_cmd/scripts/tools/compute_mu_from_vicon_data.py` to compute the
friction coefficient from the recorded bag file. We repeat this procedure
three times and use the lowest friction coefficient in the controller.
