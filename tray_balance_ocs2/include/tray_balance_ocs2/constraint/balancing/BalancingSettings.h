#pragma once

#include <ocs2_core/misc/LoadData.h>
#include <tray_balance_constraints/nominal.h>
#include <tray_balance_constraints/bounded.h>
#include <tray_balance_ocs2/constraint/ConstraintType.h>
#include <tray_balance_ocs2/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

struct TrayBalanceSettings {
    bool enabled = false;
    bool bounded = false;

    TrayBalanceConfiguration<scalar_t> nominal_config;
    BoundedTrayBalanceConfiguration<scalar_t> bounded_config;

    ConstraintType constraint_type = ConstraintType::Soft;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
