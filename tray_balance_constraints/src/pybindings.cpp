#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "tray_balance_constraints/robust.h"
#include "tray_balance_constraints/types.h"

using namespace pybind11::literals;

PYBIND11_MODULE(bindings, m) {
    using Scalar = double;

    pybind11::class_<Ball<Scalar>>(m, "Ball")
        .def(pybind11::init<const Vec3<Scalar>&, const Scalar>(), "center"_a,
             "radius"_a)
        .def_readwrite("center", &Ball<Scalar>::center)
        .def_readwrite("radius", &Ball<Scalar>::radius);
}
