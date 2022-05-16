#include <pybind11/pybind11.h>
#include <upright_ros_interface/mpc_node.h>


using namespace pybind11::literals;


PYBIND11_MODULE(bindings, m) {
    m.def("run_mpc_node", &run_mpc_node, "settings"_a);
}
