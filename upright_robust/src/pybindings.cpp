#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <upright_robust/robust.h>

using namespace pybind11::literals;
using namespace upright;

PYBIND11_MODULE(bindings, m) {
    using Scalar = double;

    pybind11::class_<RobustBounds<Scalar>>(m, "RobustBounds")
        .def(pybind11::init<const MatX<Scalar>&, const MatX<Scalar>&,
                            const MatX<Scalar>&>(),
             "RT"_a, "F"_a, "A_ineq"_a)
        .def("compute_scale", &RobustBounds<Scalar>::compute_scale, "V"_a,
             "G"_a);
}
