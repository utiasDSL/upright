#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tray_balance_constraints/bounded.h"
#include "tray_balance_constraints/ellipsoid.h"
#include "tray_balance_constraints/nominal.h"
#include "tray_balance_constraints/support_area.h"
#include "tray_balance_constraints/types.h"

// we include this directly here rather than from ellipsoid.h because other
// compilation units complain about the C++14 features imported there
#include "tray_balance_constraints/impl/bounding_ellipsoid.tpp"

using namespace pybind11::literals;

PYBIND11_MODULE(bindings, m) {
    using Scalar = double;

    pybind11::class_<RigidBody<Scalar>>(m, "RigidBody")
        .def(pybind11::init<const Scalar, const Mat3<Scalar>&,
                            const Vec3<Scalar>&>(),
             "mass"_a, "inertia"_a, "com"_a)
        .def_readwrite("mass", &RigidBody<Scalar>::mass)
        .def_readwrite("inertia", &RigidBody<Scalar>::inertia)
        .def_readwrite("com", &RigidBody<Scalar>::com);

    pybind11::class_<BalancedObject<Scalar>>(m, "BalancedObject")
        .def(
            pybind11::init<const RigidBody<Scalar>&, Scalar,
                           const PolygonSupportArea<Scalar>&, Scalar, Scalar>(),
            "body"_a, "com_height"_a, "support_area"_a, "r_tau"_a, "mu"_a)
        .def_static("compose", &BalancedObject<Scalar>::compose, "objects"_a);

    pybind11::class_<BalanceConstraintsEnabled>(m, "BalanceConstraintsEnabled")
        .def(pybind11::init<>())
        .def_readwrite("normal", &BalanceConstraintsEnabled::normal)
        .def_readwrite("friction", &BalanceConstraintsEnabled::friction)
        .def_readwrite("zmp", &BalanceConstraintsEnabled::zmp);

    // pybind11::class_<TrayBalanceConfiguration<Scalar>>(
    //     m, "TrayBalanceConfiguration")
    //     .def(pybind11::init<>())
    //     .def_readwrite("objects", &TrayBalanceConfiguration<Scalar>::objects)
    //     .def_readwrite("enabled", &TrayBalanceConfiguration<Scalar>::enabled)
    //     .def("num_constraints",
    //          &TrayBalanceConfiguration<Scalar>::num_constraints);

    pybind11::class_<PolygonSupportArea<Scalar>>(m, "PolygonSupportArea")
        .def(pybind11::init<const std::vector<Vec2<Scalar>>&>(), "vertices"_a)
        .def_readonly("vertices", &PolygonSupportArea<Scalar>::vertices)
        .def("offset", &PolygonSupportArea<Scalar>::offset, "offset"_a)
        .def("distance_outside", &PolygonSupportArea<Scalar>::distance_outside,
             "point"_a)
        .def_static("circle", &PolygonSupportArea<Scalar>::circle, "radius"_a)
        .def_static("equilateral_triangle",
                    &PolygonSupportArea<Scalar>::equilateral_triangle,
                    "side_length"_a)
        .def_static("axis_aligned_rectangle",
                    &PolygonSupportArea<Scalar>::axis_aligned_rectangle, "sx"_a,
                    "sy"_a);

    pybind11::class_<Ellipsoid<Scalar>>(m, "Ellipsoid")
        .def(pybind11::init<const Vec3<Scalar>&, const std::vector<Scalar>&,
                            const std::vector<Vec3<Scalar>>&>(),
             "center"_a, "half_lengths_vec"_a, "directions_vec"_a)
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
        .def_static("bounding", &Ellipsoid<Scalar>::bounding, "points"_a,
                    "eps"_a);

    pybind11::class_<BoundedRigidBody<Scalar>>(m, "BoundedRigidBody")
        .def(pybind11::init<const Scalar&, const Scalar&, const Vec3<Scalar>&,
                            const Vec3<Scalar>&, const Ellipsoid<Scalar>&>(),
             "mass_min"_a, "mass_max"_a, "radii_of_gyration_min"_a,
             "radii_of_gyration_max"_a, "com_ellipsoid"_a)
        .def("sample", &BoundedRigidBody<Scalar>::sample, "boundary"_a = false)
        .def("is_exact", &BoundedRigidBody<Scalar>::is_exact)
        .def_static("combined_rank", &BoundedRigidBody<Scalar>::combined_rank,
                    "bodies"_a)
        .def_readonly("mass_min", &BoundedRigidBody<Scalar>::mass_min)
        .def_readonly("mass_max", &BoundedRigidBody<Scalar>::mass_max)
        .def_readonly("radii_of_gyration_min",
                      &BoundedRigidBody<Scalar>::radii_of_gyration_min)
        .def_readonly("radii_of_gyration_max",
                      &BoundedRigidBody<Scalar>::radii_of_gyration_max)
        .def_readonly("com_ellipsoid",
                      &BoundedRigidBody<Scalar>::com_ellipsoid);

    pybind11::class_<BoundedBalancedObject<Scalar>>(m, "BoundedBalancedObject")
        .def(
            pybind11::init<const BoundedRigidBody<Scalar>&, Scalar,
                           const PolygonSupportArea<Scalar>&, Scalar, Scalar>(),
            "body"_a, "com_height"_a, "support_area_min"_a, "r_tau_min"_a,
            "mu_min"_a)
        .def("get_parameters", &BoundedBalancedObject<Scalar>::get_parameters)
        .def_readonly("body", &BoundedBalancedObject<Scalar>::body)
        .def_readonly("com_height", &BoundedBalancedObject<Scalar>::com_height)
        .def_readonly("support_area_min",
                      &BoundedBalancedObject<Scalar>::support_area_min)
        .def_readonly("r_tau_min", &BoundedBalancedObject<Scalar>::r_tau_min)
        .def_readonly("mu_min", &BoundedBalancedObject<Scalar>::mu_min);

    // Compute balancing constraints for the list of objects given
    m.def("balancing_constraints",
          [](const std::vector<BoundedBalancedObject<Scalar>> objects,
             const Vec3<Scalar>& gravity, const Mat3<Scalar>& orientation,
             const Vec3<Scalar>& angular_vel, const Vec3<Scalar>& linear_acc,
             const Vec3<Scalar>& angular_acc) {
              BalanceConstraintsEnabled enabled;  // default: all enabled
              return balancing_constraints(objects, gravity, enabled,
                                           orientation, angular_vel, linear_acc,
                                           angular_acc);
          });
}
