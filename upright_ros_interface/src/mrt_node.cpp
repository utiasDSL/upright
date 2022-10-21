#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/fwd.hpp>

#include <signal.h>
#include <iostream>

#include <pybind11/embed.h>
#include <ros/init.h>
#include <ros/package.h>

#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_ros_interfaces/mrt/MRT_ROS_Interface.h>

#include <mobile_manipulation_central/projectile.h>
#include <mobile_manipulation_central/robot_interfaces.h>

#include <upright_control/controller_interface.h>
#include <upright_control/dynamics/system_pinocchio_mapping.h>
#include <upright_control/reference_trajectory.h>
#include <upright_ros_interface/parsing.h>

using namespace upright;


const double PROJECTILE_ACTIVATION_HEIGHT = 1.0; // meters
const double MIN_POLICY_UPDATE_TIME = 0.01; // seconds

const double STATE_VIOLATION_MARGIN = 0.1;
const double INPUT_VIOLATION_MARGIN = 1.0;
const double EE_POSITION_VIOLATION_MARGIN = 0.1;


// Robot is a global variable so we can send it a brake command in the SIGINT
// handler
std::unique_ptr<mm::RobotROSInterface> robot_ptr;

// Custom SIGINT handler
void sigint_handler(int sig) {
    std::cerr << "Received SIGINT." << std::endl;
    std::cerr << "Braking robot." << std::endl;
    robot_ptr->brake();
    ros::shutdown();
}

// Double integration using semi-implicit Euler method
std::tuple<VecXd, VecXd> double_integrate(const VecXd& v, const VecXd& a,
                                          const VecXd& u,
                                          ocs2::scalar_t timestep) {
    VecXd a_new = a + timestep * u;
    VecXd v_new = v + timestep * a_new;
    return std::tuple<VecXd, VecXd>(v_new, a_new);
}

std::tuple<VecXd, VecXd> double_integrate_repeated(const VecXd& v,
                                                   const VecXd& a,
                                                   const VecXd& u,
                                                   ocs2::scalar_t timestep,
                                                   size_t n) {
    ocs2::scalar_t dt = timestep / n;
    VecXd v_new = v;
    VecXd a_new = a;
    for (int i = 0; i < n; ++i) {
        std::tie(v_new, a_new) = double_integrate(v_new, a_new, u, dt);
    }
    return std::tuple<VecXd, VecXd>(v_new, a_new);
}

class SafetyMonitor {
   public:
    SafetyMonitor(const ControllerSettings& settings,
                  const ocs2::PinocchioInterface& pinocchio_interface)
        : settings_(settings), pinocchio_interface_(pinocchio_interface) {
        SystemPinocchioMapping<TripleIntegratorPinocchioMapping<ocs2::scalar_t>,
                               ocs2::scalar_t>
            mapping(settings.dims);
        kinematics_ptr_.reset(new ocs2::PinocchioEndEffectorKinematics(
            pinocchio_interface, mapping, {settings.end_effector_link_name}));
        kinematics_ptr_->setPinocchioInterface(pinocchio_interface_);
    }

    bool limits_violated(const VecXd& x, const VecXd& u) const {
        VecXd x_robot = x.head(settings_.dims.robot.x);
        VecXd u_robot = u.head(settings_.dims.robot.u);

        if (((x_robot - settings_.state_limit_lower).array() < -STATE_VIOLATION_MARGIN).any()) {
            std::cout << "x = " << x_robot.transpose() << std::endl;
            std::cout << "State violated lower limits!" << std::endl;
            return true;
        }
        if (((settings_.state_limit_upper - x_robot).array() < -STATE_VIOLATION_MARGIN).any()) {
            std::cout << "x = " << x_robot.transpose() << std::endl;
            std::cout << "State violated upper limits!" << std::endl;
            return true;
        }
        if (((u_robot - settings_.input_limit_lower).array() < -INPUT_VIOLATION_MARGIN).any()) {
            std::cout << "u = " << u_robot.transpose() << std::endl;
            std::cout << "Input violated lower limits!" << std::endl;
            return true;
        }
        if (((settings_.input_limit_upper - u_robot).array() < -INPUT_VIOLATION_MARGIN).any()) {
            std::cout << "u = " << u_robot.transpose() << std::endl;
            std::cout << "Input violated upper limits!" << std::endl;
            return true;
        }
        return false;
    }

    bool end_effector_position_violated(const ocs2::TargetTrajectories& target,
                                        ocs2::scalar_t t, const VecXd& x) {
        VecXd q = VecXd::Zero(settings_.dims.q());
        q.tail(settings_.dims.robot.q) = x.head(settings_.dims.robot.q);

        pinocchio::forwardKinematics(pinocchio_interface_.getModel(),
                                     pinocchio_interface_.getData(), q);
        pinocchio::updateFramePlacements(pinocchio_interface_.getModel(),
                                         pinocchio_interface_.getData());
        Vec3d actual_position = kinematics_ptr_->getPosition(x).front();
        Vec3d desired_position = interpolate_end_effector_pose(t, target).first;

        VecXd position_constraint(6);
        position_constraint
            << desired_position + settings_.xyz_upper - actual_position,
            actual_position - (desired_position + settings_.xyz_lower);

        if (position_constraint.minCoeff() < -EE_POSITION_VIOLATION_MARGIN) {
            std::cerr << "Controller violated position limits!" << std::endl;
            std::cerr << "Controller position = " << actual_position.transpose()
                      << std::endl;
            std::cerr << "Desired position = " << desired_position.transpose()
                      << std::endl;
            std::cerr << "q = " << q.transpose() << std::endl;
            return true;
        }
        return false;
    }

   private:
    ControllerSettings settings_;
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> kinematics_ptr_;
    ocs2::PinocchioInterface pinocchio_interface_;
};

int main(int argc, char** argv) {
    const std::string robot_name = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, robot_name + "_mrt");
    ros::NodeHandle nh;
    std::string config_path = std::string(argv[1]);

    // controller interface
    // Python interpreter required for now because we actually load the control
    // settings and the target trajectories using Python - not ideal but easier
    // than re-implementing the parsing logic in C++
    py::scoped_interpreter guard{};
    ControllerSettings settings = parse_control_settings(config_path);
    std::cout << settings << std::endl;
    ControllerInterface interface(settings);

    SafetyMonitor monitor(settings, interface.getPinocchioInterface());

    // MRT
    ocs2::MRT_ROS_Interface mrt(robot_name);
    mrt.initRollout(&interface.getRollout());
    mrt.launchNodes(nh);

    // nominal initial state and interface to the real robot
    VecXd x0 = interface.getInitialState();

    // Initialize the robot interface
    if (settings.robot_base_type == RobotBaseType::Fixed) {
        robot_ptr.reset(new mm::UR10ROSInterface(nh));
    } else if (settings.robot_base_type == RobotBaseType::Omnidirectional) {
        robot_ptr.reset(new mm::MobileManipulatorROSInterface(nh));
    } else {
        throw std::runtime_error("Unsupported base type.");
    }

    // Set up a custom SIGINT handler to brake the robot before shutting down
    // (this is why we set it up after the robot is initialized)
    signal(SIGINT, sigint_handler);

    // Initialize interface to dynamic obstacle estimator
    mm::ProjectileROSInterface projectile(nh, "Projectile");
    bool avoid_dynamic_obstacle = false;

    ocs2::scalar_t timestep = 1.0 / settings.rate;
    ros::Rate rate(settings.rate);

    // wait until we get feedback from the robot
    while (ros::ok()) {
        ros::spinOnce();
        if (robot_ptr->ready()) {
            break;
        }
        rate.sleep();
    }
    std::cout << "Received feedback from robot." << std::endl;

    // update to the real initial state
    x0.head(settings.dims.robot.q) = robot_ptr->q();

    // reset MPC
    ocs2::TargetTrajectories target = parse_target_trajectory(config_path, x0);
    mrt.resetMpcNode(target);

    ocs2::SystemObservation observation;
    observation.state = x0;
    observation.input.setZero(settings.dims.u());
    observation.time = ros::Time::now().toSec();
    mrt.setCurrentObservation(observation);

    // Let MPC generate the initial plan
    while (ros::ok()) {
        mrt.spinMRT();
        if (mrt.initialPolicyReceived()) {
            break;
        }
        rate.sleep();
    }
    mrt.updatePolicy();
    std::cout << "Received first policy." << std::endl;

    VecXd x = x0;
    VecXd xd = VecXd::Zero(x.size());
    VecXd u = VecXd::Zero(settings.dims.u());
    size_t mode = 0;

    VecXd v_ff = VecXd::Zero(settings.dims.robot.v);
    VecXd a_ff = VecXd::Zero(settings.dims.robot.v);

    ros::Time now = ros::Time::now();
    ros::Time last_policy_update_time = now;
    ros::Duration policy_update_delay(MIN_POLICY_UPDATE_TIME);

    ocs2::scalar_t t = now.toSec();
    ocs2::scalar_t last_t = t;

    while (ros::ok()) {
        now = ros::Time::now();
        last_t = t;
        t = now.toSec();
        ocs2::scalar_t dt = t - last_t;

        // Robot feedback
        VecXd q = robot_ptr->q();
        // VecXd v = robot.get_joint_velocity_raw();

        // Integrate our internal model to get velocity and acceleration
        // "feedback"
        VecXd u_robot = u.head(settings.dims.robot.u);
        std::tie(v_ff, a_ff) = double_integrate(v_ff, a_ff, u_robot, dt);

        // Current state is built from robot feedback for q and v; for
        // acceleration we just assume we are tracking well since we don't get
        // good feedback on this
        x.head(settings.dims.robot.x) << q, v_ff, a_ff;

        // Dynamic obstacle
        if (settings.dims.o > 0 && projectile.ready()) {
            Vec3d q_obs = projectile.q();
            if (q_obs(2) > PROJECTILE_ACTIVATION_HEIGHT) {
                avoid_dynamic_obstacle = true;
                std::cout << "  q_obs = " << q_obs.transpose() << std::endl;
            } else {
                std::cout << "~ q_obs = " << q_obs.transpose() << std::endl;
            }

            // TODO we could also have this trigger a case where we now assume
            // the trajectory of the object is perfect
            //
            // TODO we could have the MPC reset if the projectile was inside
            // the "awareness zone" but then leaves, such that the robot is
            // ready for the next throw

            if (avoid_dynamic_obstacle) {
                Vec3d v_obs = projectile.v();
                Vec3d a_obs = settings.obstacle_settings.dynamic_obstacles[0]
                                  .acceleration;
                x.tail(9) << q_obs, v_obs, a_obs;
            }
        }

        // Compute optimal state and input using current policy
        mrt.evaluatePolicy(t, x, xd, u, mode);

        // Check that controller is not making the end effector leave allowed
        // region
        if (monitor.end_effector_position_violated(target, t, xd)) {
            break;
        }

        // Check that the controller has provided sane values.
        if (monitor.limits_violated(xd, u)) {
            break;
        }

        robot_ptr->publish_cmd_vel(v_ff, /* bodyframe = */ false);

        // Send observation to MPC
        observation.time = t;
        observation.state = x;
        observation.input = u;
        mrt.setCurrentObservation(observation);

        ros::spinOnce();

        // Get new policy messages and update the policy if available
        mrt.spinMRT();
        if (now - last_policy_update_time >= policy_update_delay) {
            mrt.updatePolicy();
            last_policy_update_time = now;
        }

        rate.sleep();
    }

    // stop the robot when we're done
    std::cout << "Braking robot." << std::endl;
    robot_ptr->brake();
    ros::shutdown();

    // Successful exit
    return 0;
}
