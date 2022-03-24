= Nonholonomic base =
Better way of doing non-holonomic constraint:
* impose the constraint in the optimization problem itself (basic equality)
* possibly apply corrections on the PyBullet side

Implementation:
* Ideally I could use `LinearStateInputConstraint`, rollout is immediately
  unstable because it appears that neither ILQR not SLQ support input equality
* Instead, I use:
  - state constraint to keep v_y = 0
  - set a_y = 0 in Pinocchio mapping
  - set a_y = 0 in dynamics
  This seems to enforce non-holonomic base behaviour, but the resulting motion
  is somewhat unintuitive. In future I will have to compare to the "proper"
  case where I actually explicity reduce the number of inputs to the system.
