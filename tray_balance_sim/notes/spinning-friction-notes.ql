= Spinning friction =
* It is not important that PyBullet enforces the exact same mathematical
  formulation as us
* It **is** important that our formulation corresponds to the **behaviour** of
  PyBullet: without anisotropic or spinning friction, does applied torque
  eventually result in slip? And if so, what value of r_tau does that
  correspond to?
* It turns out that the torque is actually just very small on these objects
  even for high angular acceleration, so the r_tau we have assuming uniform
  friction is much higher than required unless friction is very low

Now trying with spinning friction to see if it makes a difference
* Without anisotropic friction, it does seem like there is some yaw slip when
  bullet r_tau=0
* However, it has been difficult to identify what spinning friction is
  equivalent to just not setting it
* Suspect spinning friction = μ * r_tau (it wouldn't make sense otherwise)
* Spinning friction is not the same as damping (a.k.a. viscous friction, which
  only acts against velocity)
* Seems like it should be unnecessary to apply a limit on spinning friction
  given that lateral friction already works here, but perhaps spinning friction
  puts an extra limit on this?
  - setting spinning friction to a small (**but > 0**) value seems to reduce
    the frictional torque, but the reduced value is more than I predicted

Want to find out what Bullet uses if I just apply torque?
* Based on the fact that Bullet returns 4 contact points for the cuboid, I
  suspect it is just using a discrete set of contact points.
* The friction at each point is not always the maximum I would predict at each
  contact point, given mu and normal force
* Total torque applied to the cube (via friction) is within the allowable
  bounds assuming it could be concentrated at the farthest points on the SA

Basically, it is clear that Bullet must use some representation for
transferring frictional torque, but it isn't using exactly the four corner
points of the cube SA (for example). Furthermore, it is not clear how setting
the spinning friction coefficient affects things, so my current recommendation
is to not set it at all and just proceed with my assumption of uniform pressure
distribution everywhere.
