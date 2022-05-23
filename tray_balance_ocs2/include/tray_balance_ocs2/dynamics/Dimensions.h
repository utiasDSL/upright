#pragma once

#include <ostream>
#include <cstddef>

struct RobotDimensions {
    size_t q;
    size_t v;
    size_t x;
    size_t u;
};

inline std::ostream& operator<<(std::ostream& out, const RobotDimensions& dims) {
    out << "nq = " << dims.q << std::endl
        << "nv = " << dims.v << std::endl
        << "nx = " << dims.x << std::endl
        << "nu = " << dims.u << std::endl;
    return out;
}
