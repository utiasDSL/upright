#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

namespace upright {

// Dimensions of a single robot
struct RobotDimensions {
    size_t q = 0;  // Configuration vector dimension
    size_t v = 0;  // Generalized velocity vector dimension
    size_t x = 0;  // State dimension
    size_t u = 0;  // Input dimension
};

// Dimensions of the optimization problem
struct OptimizationDimensions {
    // Dimensions of the (for now, single) robot
    RobotDimensions robot;

    // Number of dynamic obstacles
    size_t o = 0;

    // Number of contact points
    size_t c = 0;

    // Number of rigid bodies being balanced.
    size_t b = 0;

    // Dimension of each contact force variable. In general this is three, but
    // if we assume that friction coefficient is zero we can set this to one.
    size_t nf = 3;

    // Total configuration vector dimension
    size_t q() const { return robot.q + 3 * o; }

    // Total velocity vector dimension
    size_t v() const { return robot.v + 3 * o; }

    // Total state vector dimension
    size_t x() const { return robot.x + 9 * o; }

    // Contact force vector dimension
    size_t f() const { return nf * c; }

    // Input vector dimension
    size_t u() const { return robot.u + f(); }
};

inline std::ostream& operator<<(std::ostream& out,
                                const RobotDimensions& dims) {
    out << "nq = " << dims.q << std::endl
        << "nv = " << dims.v << std::endl
        << "nx = " << dims.x << std::endl
        << "nu = " << dims.u << std::endl;
    return out;
}

inline std::ostream& operator<<(std::ostream& out,
                                const OptimizationDimensions& dims) {
    out << "nq = " << dims.q() << std::endl
        << "nv = " << dims.v() << std::endl
        << "nx = " << dims.x() << std::endl
        << "nu = " << dims.u() << std::endl
        << "nf = " << dims.f() << std::endl
        << "nc = " << dims.c << std::endl
        << "no = " << dims.o << std::endl;
    return out;
}

}  // namespace upright
