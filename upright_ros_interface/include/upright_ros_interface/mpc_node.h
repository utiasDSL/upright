#pragma once

#include <tray_balance_ocs2/ControllerSettings.h>

using namespace ocs2;
using namespace mobile_manipulator;

// Run the MPC node.
int run_mpc_node(const ControllerSettings& settings);
