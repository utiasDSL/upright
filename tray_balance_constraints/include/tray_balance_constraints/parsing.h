#pragma once

#include <vector>
#include <string>

#include <tray_balance_constraints/bounded.h>

std::vector<BoundedBalancedObject<double>> parse_objects(std::string config_file_path);
