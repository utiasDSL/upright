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
    // balancing constraints are used.
    size_t f = 0;
};

inline std::ostream& operator<<(std::ostream& out,
                                const RobotDimensions& dims) {
    out << "nq = " << dims.q << std::endl
        << "nv = " << dims.v << std::endl
        << "nx = " << dims.x << std::endl
        << "nu = " << dims.u << std::endl
        << "nf = " << dims.f << std::endl;
    return out;
}

}  // namespace upright
