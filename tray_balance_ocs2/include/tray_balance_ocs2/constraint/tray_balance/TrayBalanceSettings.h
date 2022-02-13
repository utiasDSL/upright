#pragma once

#include <ocs2_core/misc/LoadData.h>
#include <tray_balance_constraints/robust.h>
#include <tray_balance_ocs2/definitions.h>
#include <tray_balance_ocs2/constraint/ConstraintType.h>
#include <tray_balance_constraints/inequality_constraints.h>

namespace ocs2 {
namespace mobile_manipulator {

struct TrayBalanceConfiguration {
    std::vector<BalancedObject<scalar_t>> objects;

    // TODO debatable whether this should be here or in TrayBalanceSettings
    BalanceConstraintsEnabled enabled;

    size_t num_constraints() const {
        size_t n = 0;
        for (auto& obj : objects) {
            n += obj.num_constraints();
        }
        return n;
    }

    size_t num_parameters() const {
        size_t n = 0;
        for (auto& obj : objects) {
            n += obj.num_parameters();
        }
        return n;
    }
};

struct TrayBalanceSettings {
    bool enabled = false;
    bool robust = false;

    TrayBalanceConfiguration config;
    RobustParameterSet<scalar_t> robust_params;

    ConstraintType constraint_type = ConstraintType::Soft;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
