#pragma once

#include <ocs2_core/misc/LoadData.h>
#include <tray_balance_ocs2/definitions.h>
#include <tray_balance_ocs2/constraint/tray_balance/TrayBalanceConfigurations.h>

namespace ocs2 {
namespace mobile_manipulator {

enum TrayBalanceConstraintType {
    Soft,
    Hard,
};

struct TrayBalanceSettings {
    bool enabled = true;
    bool robust = false;

    TrayBalanceConfiguration config;

    TrayBalanceConstraintType constraint_type = Soft;
    scalar_t mu = 1e-2;
    scalar_t delta = 1e-3;

    void set_constraint_type(const std::string& constraint_type) {
        if (constraint_type == "soft") {
            this->constraint_type = Soft;
        } else if (constraint_type == "hard") {
            this->constraint_type = Hard;
        } else {
            throw std::runtime_error("Invalid constraint type: " +
                                     constraint_type);
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
        settings.config.set_type(config_type);

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
