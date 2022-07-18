#pragma once

#include <string>

namespace upright {

enum class ConstraintType {
    Soft,
    Hard,
};

inline std::string constraint_type_to_string(const ConstraintType& c) {
    if (c == ConstraintType::Soft) {
        return "soft";
    } else {
        return "hard";
    }
}

inline ConstraintType constraint_type_from_string(const std::string& s) {
    if (s == "soft") {
        return ConstraintType::Soft;
    } else if (s == "hard") {
        return ConstraintType::Hard;
    }
    throw std::runtime_error("Could not parse ConstraintType from string.");
}

}  // namespace upright
