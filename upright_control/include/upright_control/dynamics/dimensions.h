#pragma once

#include <cstddef>
#include <ostream>

namespace upright {

struct RobotDimensions {
    size_t q; // Configuration vector dimension
    size_t v; // Generalized velocity vector dimension
    size_t x; // State dimension
    size_t u; // Input dimension

    // Number of constraint forces. Only relevant if the contact force-based
    // balancing constraints are used. Each force has three components.
    size_t f = 0;

    // Dimension of optimization state variable
    size_t ox() const {
        return x;
    }

    // Dimension of optimization "input" variable
    size_t ou() const {
        return u + 3 * f;
    }
};

inline std::ostream& operator<<(std::ostream& out,
                                const RobotDimensions& dims) {
    out << "nq = " << dims.q << std::endl
        << "nv = " << dims.v << std::endl
        << "nx = " << dims.x << std::endl
        << "nu = " << dims.u << std::endl
        << "nf = " << dims.f << std::endl
        << "nox = " << dims.ox() << std::endl
        << "nou = " << dims.ou() << std::endl;
    return out;
}

}  // namespace upright
