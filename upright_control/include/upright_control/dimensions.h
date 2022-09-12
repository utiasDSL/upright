#pragma once

#include <cstddef>
#include <ostream>

namespace upright {

// Dimensions of the optimization problem
class OptimizationDimensions {
    OptimizationDimensions() {}

    // TODO we likely have to deal with incremental building of this
    // OptimizationDimensions(const std::vector<RobotDimensions>& robots, size_t num_contacts)
    //     : robots_(robots), c_(num_contacts) {
    //     for (const auto& robot : robots) {
    //         q_ += robot.q;
    //         v_ += robot.v;
    //         x_ += robot.x;
    //         u_ += robot.u;
    //     }
    // }

    public:
    void push_back(const RobotDimensions& robot) {
        robots_.push_back(robot);
        q_ += robot.q;
        v_ += robot.v;
        x_ += robot.x;
        u_ += robot.u;
    }

    void set_num_contacts(size_t num_contacts) {
        c_ = num_contacts;
    }

    // TODO this would be more convenient as an accessor for each individual
    // element
    const std::vector<RobotDimensions>& robots() const {
        return robots_;
    }

    size_t q() const { return q_; }
    size_t v() const { return v_; }
    size_t x() const { return x_; }

    // Number of optimized inputs is the number of robot inputs plus a 3-dim
    // force per contact point
    size_t u() const { return u_ + 3 * c_; }

   private:
    // Dimensions of different robots in the environment
    // Dynamic obstacles are also represented as (autonomous) robots
    std::vector<RobotDimensions> robots_;

    // Number of contact points
    size_t c_ = 0;

    size_t q_ = 0;
    size_t v_ = 0;
    size_t x_ = 0;
    size_t u_ = 0;
};

// Dimensions of a single robot
struct RobotDimensions {
    size_t q;  // Configuration vector dimension
    size_t v;  // Generalized velocity vector dimension
    size_t x;  // State dimension
    size_t u;  // Input dimension

    // // Number of constraint forces. Only relevant if the contact force-based
    // // balancing constraints are used. Each force has three components.
    // size_t f = 0;
    //
    // // Dimension of optimization state variable
    // size_t ox() const { return x; }
    //
    // // Dimension of optimization "input" variable
    // size_t ou() const { return u + 3 * f; }
};

inline std::ostream& operator<<(std::ostream& out,
                                const RobotDimensions& dims) {
    out << "nq = " << dims.q << std::endl
        << "nv = " << dims.v << std::endl
        << "nx = " << dims.x << std::endl
        << "nu = " << dims.u << std::endl return out;
}

inline std::ostream& operator<<(std::ostream& out,
                                const OptimizationDimensions& dims) {
    out << "robot = " << dims.robot << std::endl
        << "nx = " << dims.x() << std::endl
        << "nu = " << dims.u() << std::endl
        << "nf = " << dims.f << std::endl
        << "no = " << dims.o << std::endl;
    return out;
}

}  // namespace upright
