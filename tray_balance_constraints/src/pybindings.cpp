#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tray_balance_constraints/robust.h"
#include "tray_balance_constraints/robust2.h"
#include "tray_balance_constraints/types.h"

using namespace pybind11::literals;

PYBIND11_MODULE(bindings, m) {
    using Scalar = double;

    pybind11::class_<Ball<Scalar>>(m, "Ball")
        .def(pybind11::init<const Vec3<Scalar>&, const Scalar>(), "center"_a,
             "radius"_a)
        .def_readwrite("center", &Ball<Scalar>::center)
        .def_readwrite("radius", &Ball<Scalar>::radius);

    pybind11::class_<Ellipsoid<Scalar>>(m, "Ellipsoid")
        .def(pybind11::init<const Vec3<Scalar>&, const std::vector<Scalar>&,
                            const std::vector<Vec3<Scalar>>&, const size_t>(),
             "center"_a, "half_lengths_vec"_a, "directions_vec"_a, "rank"_a)
        .def("center", &Ellipsoid<Scalar>::center)
        .def("half_lengths", &Ellipsoid<Scalar>::half_lengths)
        .def("directions", &Ellipsoid<Scalar>::directions)
        .def("rank", &Ellipsoid<Scalar>::rank)
        .def("rangespace", &Ellipsoid<Scalar>::rangespace)
        .def("nullspace", &Ellipsoid<Scalar>::nullspace)
        .def("E", &Ellipsoid<Scalar>::E)
        .def("Einv", &Ellipsoid<Scalar>::Einv)
        .def("scaled", &Ellipsoid<Scalar>::scaled, "a"_a)
        .def("contains", &Ellipsoid<Scalar>::contains, "x"_a)
        .def("sample", &Ellipsoid<Scalar>::sample, "boundary"_a = false)
        .def_static("point", &Ellipsoid<Scalar>::point, "center"_a)
        .def_static("segment", &Ellipsoid<Scalar>::segment, "v1"_a, "v2"_a)
        .def_static("bounding_ellipsoid",
                    &Ellipsoid<Scalar>::bounding_ellipsoid, "points"_a,
                    "eps"_a);

    pybind11::class_<RigidBodyBounds<Scalar>>(m, "RigidBodyBounds")
        .def(pybind11::init<const Scalar&, const Scalar&, const Scalar&,
                            const Ellipsoid<Scalar>&>(),
             "mass_min"_a, "mass_max"_a, "r_gyr"_a, "com_ellipsoid"_a)
        .def("sample", &RigidBodyBounds<Scalar>::sample, "boundary"_a = false)
        .def_static("combined_rank", &RigidBodyBounds<Scalar>::combined_rank,
                    "bodies"_a);
}
