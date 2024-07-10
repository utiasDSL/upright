#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "upright_core/contact.h"
#include "upright_core/contact_constraints.h"
// #include "upright_core/nominal.h"
#include "upright_core/rigid_body.h"
#include "upright_core/types.h"

using namespace pybind11::literals;
using namespace upright;

PYBIND11_MODULE(bindings, m) {
    using Scalar = double;

    pybind11::class_<RigidBody<Scalar>>(m, "RigidBody")
        .def(pybind11::init<const Scalar, const Mat3<Scalar>&,
                            const Vec3<Scalar>&>(),
             "mass"_a, "inertia"_a, "com"_a)
        .def_readwrite("mass", &RigidBody<Scalar>::mass)
        .def_readwrite("inertia", &RigidBody<Scalar>::inertia)
        .def_readwrite("com", &RigidBody<Scalar>::com);

    // pybind11::class_<BalancedObject<Scalar>>(m, "BalancedObject")
    //     .def(
    //         pybind11::init<const RigidBody<Scalar>&, Scalar,
    //                        const PolygonSupportArea<Scalar>&, Scalar, Scalar>(),
    //         "body"_a, "com_height"_a, "support_area"_a, "r_tau"_a, "mu"_a)
    //     .def_readonly("body", &BalancedObject<Scalar>::body)
    //     // .def_readonly("com_height", &BalancedObject<Scalar>::com_height)
    //     // .def_readonly("support_area", &BalancedObject<Scalar>::support_area)
    //     // .def_readonly("r_tau", &BalancedObject<Scalar>::r_tau)
    //     .def_readonly("mu", &BalancedObject<Scalar>::mu)
    //     .def_static("compose", &BalancedObject<Scalar>::compose, "objects"_a);

    pybind11::class_<ContactPoint<Scalar>>(m, "ContactPoint")
        .def(pybind11::init<>())
        .def_readwrite("object1_name", &ContactPoint<Scalar>::object1_name)
        .def_readwrite("object2_name", &ContactPoint<Scalar>::object2_name)
        .def_readwrite("mu", &ContactPoint<Scalar>::mu)
        .def_readwrite("r_co_o1", &ContactPoint<Scalar>::r_co_o1)
        .def_readwrite("r_co_o2", &ContactPoint<Scalar>::r_co_o2)
        .def_readwrite("normal", &ContactPoint<Scalar>::normal)
        .def_readwrite("span", &ContactPoint<Scalar>::span);

    pybind11::class_<Pose<Scalar>>(m, "Pose")
        .def(pybind11::init<>())
        .def_readwrite("position", &Pose<Scalar>::position)
        .def_readwrite("orientation", &Pose<Scalar>::orientation)
        .def("Zero", &Pose<Scalar>::Zero);

    pybind11::class_<Twist<Scalar>>(m, "Twist")
        .def(pybind11::init<>())
        .def_readwrite("linear", &Twist<Scalar>::linear)
        .def_readwrite("angular", &Twist<Scalar>::angular)
        .def("Zero", &Twist<Scalar>::Zero);

    pybind11::class_<RigidBodyState<Scalar>>(m, "RigidBodyState")
        .def(pybind11::init<>())
        .def_readwrite("pose", &RigidBodyState<Scalar>::pose)
        .def_readwrite("velocity", &RigidBodyState<Scalar>::velocity)
        .def_readwrite("acceleration", &RigidBodyState<Scalar>::acceleration)
        .def("Zero", &RigidBodyState<Scalar>::Zero);

    // // TODO is this used?
    // pybind11::class_<BalancedObjectArrangement<Scalar>>(
    //     m, "BalancedObjectArrangement")
    //     .def(
    //         pybind11::init<const std::map<std::string, BalancedObject<Scalar>>&,
    //                        const Vec3<Scalar>&>())
    //     .def("balancing_constraints",
    //          &BalancedObjectArrangement<Scalar>::balancing_constraints,
    //          "state"_a);

    m.def("compute_object_dynamics_constraints",
          &compute_object_dynamics_constraints<Scalar>);
    m.def("compute_contact_force_constraints_linearized",
          &compute_contact_force_constraints_linearized<Scalar>);
}
