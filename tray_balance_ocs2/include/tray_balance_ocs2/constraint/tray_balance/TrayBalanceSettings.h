#pragma once

#include <ocs2_core/misc/LoadData.h>
#include <tray_balance_constraints/robust.h>
#include <tray_balance_ocs2/definitions.h>
#include <tray_balance_ocs2/constraint/ConstraintType.h>
#include <tray_balance_ocs2/constraint/tray_balance/TrayBalanceConfigurations.h>

namespace ocs2 {
namespace mobile_manipulator {


struct TrayBalanceSettings {

    bool enabled = false;
    bool robust = false;

    TrayBalanceConfiguration config;
    RobustParameterSet<scalar_t> robust_params;

    ConstraintType constraint_type = ConstraintType::Soft;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;

    void set_constraint_type(const std::string& s) {
        if (s == "soft") {
            constraint_type = ConstraintType::Soft;
        } else if (s == "hard") {
            constraint_type = ConstraintType::Hard;
        } else {
            throw std::runtime_error("Invalid constraint type: " + s);
        }
    }

    static TrayBalanceSettings load(
        const std::string& filename,
        const std::string& prefix = "trayBalanceConstraints",
        bool verbose = true) {
        TrayBalanceSettings settings;

        boost::property_tree::ptree pt;
        boost::property_tree::read_info(filename, pt);
        std::cerr << "\n #### " << prefix << " Settings: ";
        std::cerr
            << "\n #### "
               "============================================================="
               "================\n";
        loadData::loadPtreeValue(pt, settings.enabled, prefix + ".enabled",
                                 verbose);
        loadData::loadPtreeValue(pt, settings.robust, prefix + ".robust",
                                 verbose);
        loadData::loadPtreeValue(pt, settings.mu, prefix + ".mu", verbose);
        loadData::loadPtreeValue(pt, settings.delta, prefix + ".delta",
                                 verbose);
        loadData::loadPtreeValue(pt, settings.config.num,
                                 prefix + ".num_objects", verbose);

        std::string constraint_type;
        loadData::loadPtreeValue(pt, constraint_type,
                                 prefix + ".constraint_type", verbose);
        settings.set_constraint_type(constraint_type);

        std::string config_type;
        loadData::loadPtreeValue(pt, config_type, prefix + ".config_type",
                                 verbose);
        settings.config.set_arrangement(config_type);

        std::cerr
            << " #### "
               "============================================================="
               "================"
            << std::endl;

        return settings;
    };
};

}  // namespace mobile_manipulator
}  // namespace ocs2
