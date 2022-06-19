#pragma once

#include <string>

namespace upright {

enum class RobotBaseType {
    Fixed,
    Nonholonomic,
    Omnidirectional,
    Floating,
};

inline RobotBaseType robot_base_type_from_string(const std::string& s) {
    if (s == "fixed") {
        return RobotBaseType::Fixed;
    } else if (s == "nonholonomic") {
        return RobotBaseType::Nonholonomic;
    } else if (s == "omnidirectional") {
        return RobotBaseType::Omnidirectional;
    } else if (s == "floating") {
        return RobotBaseType::Floating;
    }
    throw std::runtime_error("Cannot parse RobotBaseType from string.");
}

inline std::string robot_base_type_to_string(const RobotBaseType& type) {
    if (type == RobotBaseType::Fixed) {
        return "fixed";
    } else if (type == RobotBaseType::Nonholonomic) {
        return "nonholonomic";
    } else if (type == RobotBaseType::Omnidirectional) {
        return "omnidirectional";
    } else if (type == RobotBaseType::Floating) {
        return "floating";
    }
    throw std::runtime_error(
        "No string for RobotBaseType - this should not happen.");
}

}  // namespace upright
