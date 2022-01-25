#pragma once

// Configurations for tray balancing experiments. Called in
// TrayBalanceConstraints.h.

#include <tray_balance_constraints/inequality_constraints.h>
#include <tray_balance_ocs2/definitions.h>

namespace ocs2 {
namespace mobile_manipulator {

struct TrayBalanceConfiguration {
    enum class Arrangement {
        Flat,
        Stacked,
    };

    Arrangement arrangement = Arrangement::Flat;
    size_t num = 0; // TODO remove
    std::vector<BalancedObject<scalar_t>> objects;

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

    void set_arrangement(const std::string& s) {
        if (s == "flat") {
            arrangement = Arrangement::Flat;
        } else if (s == "stacked") {
            arrangement = Arrangement::Stacked;
        } else {
            throw std::runtime_error("Invalid arrangement: " + s);
        }
    }
};

template <typename Scalar>
BalancedObject<Scalar> build_tray_object() {
    Scalar tray_mass(0.5);
    Scalar tray_com_height(0.01);
    Scalar tray_height = tray_com_height * 2;
    Scalar tray_radius(0.2);
    Scalar tray_mu(0.5);
    Scalar ee_side_length(0.2);
    Scalar tray_zmp_margin(0.0);
    Vec3<Scalar> tray_com(3);  // wrt the EE frame
    tray_com << Scalar(0), Scalar(0), tray_com_height + Scalar(0.02);

    Mat3<Scalar> tray_inertia =
        cylinder_inertia_matrix<Scalar>(tray_mass, tray_radius, tray_height);
    RigidBody<Scalar> tray_body(tray_mass, tray_inertia, tray_com);

    Scalar tray_r_tau =
        equilateral_triangle_inscribed_radius<Scalar>(ee_side_length);
    Vec2<Scalar> tray_support_offset = Vec2<Scalar>::Zero();

    // CircleSupportArea<ad_scalar_t> tray_support_area(
    //     tray_r_tau, tray_support_offset, ad_scalar_t(0));
    std::vector<Vec2<Scalar>> tray_support_vertices =
        equilateral_triangle_support_vertices(ee_side_length);
    PolygonSupportArea<Scalar> tray_support_area(
        tray_support_vertices, tray_support_offset, tray_zmp_margin);

    BalancedObject<Scalar> tray(tray_body, tray_com_height, tray_support_area,
                                tray_r_tau, tray_mu);
    return tray;
}

template <typename Scalar>
BalancedObject<Scalar> build_cuboid_object() {
    Scalar cuboid_mass(0.5);
    Scalar cuboid_mu(0.5);
    Scalar cuboid_com_height(0.075);
    Scalar cuboid_zmp_margin(0.0);
    Vec2<Scalar> cuboid_support_offset = Vec2<Scalar>::Zero();
    Vec3<Scalar> cuboid_side_lengths(Scalar(0.15), Scalar(0.15),
                                     cuboid_com_height * 2);
    Mat3<Scalar> cuboid_inertia =
        cuboid_inertia_matrix(cuboid_mass, cuboid_side_lengths);
    Vec3<Scalar> cuboid_com(Scalar(0), Scalar(0), Scalar(0.115));
    RigidBody<Scalar> cuboid_body(cuboid_mass, cuboid_inertia, cuboid_com);

    // TODO should try using a cuboid r_tau
    // Scalar cuboid_r_tau = circle_r_tau(cuboid_side_lengths(0) * 0.5);
    Scalar cuboid_r_tau =
        rectangle_r_tau(cuboid_side_lengths(0), cuboid_side_lengths(1));

    std::vector<Vec2<Scalar>> cuboid_vertices =
        cuboid_support_vertices(cuboid_side_lengths);
    PolygonSupportArea<Scalar> cuboid_support_area(
        cuboid_vertices, cuboid_support_offset, cuboid_zmp_margin);

    BalancedObject<Scalar> cuboid(cuboid_body, cuboid_com_height,
                                  cuboid_support_area, cuboid_r_tau, cuboid_mu);
    return cuboid;
}

template <typename Scalar>
BalancedObject<Scalar> build_cylinder1(const TrayBalanceConfiguration& config) {
    Scalar cylinder_mass(0.5);
    Scalar cylinder_mu(0.5);
    Scalar cylinder_com_height(0.075);
    Scalar cylinder_radius(0.05);
    Scalar cylinder_height = cylinder_com_height * 2;
    Scalar cylinder_r_tau = circle_r_tau(cylinder_radius);
    Scalar cylinder_zmp_margin(0.01);
    Vec2<Scalar> cylinder_support_offset = Vec2<Scalar>::Zero();

    // position changes based on configuration
    Vec3<Scalar> cylinder_com;
    if (config.arrangement == TrayBalanceConfiguration::Arrangement::Flat) {
        Scalar ee_side_length(0.2);
        Vec2<Scalar> com_xy =
            equilateral_triangle_cup_location(ee_side_length, Scalar(0.08), 0);
        cylinder_com << com_xy, Scalar(0.115);
    } else if (config.arrangement ==
               TrayBalanceConfiguration::Arrangement::Stacked) {
        cylinder_com << Scalar(0), Scalar(0), Scalar(0.115);
    }

    Mat3<Scalar> cylinder_inertia = cylinder_inertia_matrix(
        cylinder_mass, cylinder_radius, cylinder_height);
    RigidBody<Scalar> cylinder_body(cylinder_mass, cylinder_inertia,
                                    cylinder_com);

    // Approximate support area using the large rectangle that fits in the
    // circle: this is better for solver stability.
    Scalar s = cylinder_radius * Scalar(sqrt(2.0));
    std::vector<Vec2<Scalar>> cylinder_vertices = cuboid_support_vertices(s, s);
    PolygonSupportArea<Scalar> cylinder_support_area(
        cylinder_vertices, cylinder_support_offset, cylinder_zmp_margin);
    // CircleSupportArea<Scalar> cylinder_support_area(
    //     cylinder_radius, cylinder_support_offset, cylinder_zmp_margin);

    BalancedObject<Scalar> cylinder(cylinder_body, cylinder_com_height,
                                    cylinder_support_area, cylinder_r_tau,
                                    cylinder_mu);
    return cylinder;
}

template <typename Scalar>
BalancedObject<Scalar> build_cylinder2(const TrayBalanceConfiguration& config) {
    Scalar cylinder_mass(0.5);
    Scalar cylinder_mu(0.5);
    Scalar cylinder_com_height(0.075);
    Scalar cylinder_radius(0.05);
    Scalar cylinder_height = cylinder_com_height * 2;
    Scalar cylinder_r_tau = circle_r_tau(cylinder_radius);
    Scalar cylinder_zmp_margin(0.01);
    Vec2<Scalar> cylinder_support_offset = Vec2<Scalar>::Zero();

    Vec3<Scalar> cylinder_com;
    if (config.arrangement == TrayBalanceConfiguration::Arrangement::Flat) {
        Scalar ee_side_length(0.2);
        Vec2<Scalar> com_xy =
            equilateral_triangle_cup_location(ee_side_length, Scalar(0.08), 1);
        cylinder_com << com_xy, Scalar(0.115);
    } else if (config.arrangement ==
               TrayBalanceConfiguration::Arrangement::Stacked) {
        cylinder_com << Scalar(0), Scalar(0), Scalar(0.265);
    }

    Mat3<Scalar> cylinder_inertia = cylinder_inertia_matrix(
        cylinder_mass, cylinder_radius, cylinder_height);
    RigidBody<Scalar> cylinder_body(cylinder_mass, cylinder_inertia,
                                    cylinder_com);

    Scalar s = cylinder_radius * Scalar(sqrt(2.0));
    std::vector<Vec2<Scalar>> cylinder_vertices = cuboid_support_vertices(s, s);
    PolygonSupportArea<Scalar> cylinder_support_area(
        cylinder_vertices, cylinder_support_offset, cylinder_zmp_margin);
    // CircleSupportArea<Scalar> cylinder_support_area(
    //     cylinder_radius, cylinder_support_offset, cylinder_zmp_margin);

    BalancedObject<Scalar> cylinder(cylinder_body, cylinder_com_height,
                                    cylinder_support_area, cylinder_r_tau,
                                    cylinder_mu);
    return cylinder;
}

template <typename Scalar>
BalancedObject<Scalar> build_cylinder3(const TrayBalanceConfiguration& config) {
    Scalar cylinder_mass(0.5);
    Scalar cylinder_mu(0.5);
    Scalar cylinder_com_height(0.075);
    Scalar cylinder_radius(0.05);
    Scalar cylinder_height = cylinder_com_height * 2;
    Scalar cylinder_r_tau = circle_r_tau(cylinder_radius);
    Scalar cylinder_zmp_margin(0.01);
    Vec2<Scalar> cylinder_support_offset = Vec2<Scalar>::Zero();

    Vec3<Scalar> cylinder_com;
    if (config.arrangement == TrayBalanceConfiguration::Arrangement::Flat) {
        Scalar ee_side_length(0.2);
        Vec2<Scalar> com_xy =
            equilateral_triangle_cup_location(ee_side_length, Scalar(0.08), 2);
        cylinder_com << com_xy, Scalar(0.115);
    } else if (config.arrangement ==
               TrayBalanceConfiguration::Arrangement::Stacked) {
        cylinder_com << Scalar(0), Scalar(0), Scalar(0.415);
    }

    Mat3<Scalar> cylinder_inertia = cylinder_inertia_matrix(
        cylinder_mass, cylinder_radius, cylinder_height);
    RigidBody<Scalar> cylinder_body(cylinder_mass, cylinder_inertia,
                                    cylinder_com);

    Scalar s = cylinder_radius * Scalar(sqrt(2.0));
    std::vector<Vec2<Scalar>> cylinder_vertices = cuboid_support_vertices(s, s);
    PolygonSupportArea<Scalar> cylinder_support_area(
        cylinder_vertices, cylinder_support_offset, cylinder_zmp_margin);
    // CircleSupportArea<Scalar> cylinder_support_area(
    //     cylinder_radius, cylinder_support_offset, cylinder_zmp_margin);

    BalancedObject<Scalar> cylinder(cylinder_body, cylinder_com_height,
                                    cylinder_support_area, cylinder_r_tau,
                                    cylinder_mu);
    return cylinder;
}

template <typename Scalar>
std::vector<BalancedObject<Scalar>> build_objects(
    const TrayBalanceConfiguration& config) {
    auto tray = build_tray_object<Scalar>();
    auto cylinder1 = build_cylinder1<Scalar>(config);
    auto cylinder2 = build_cylinder2<Scalar>(config);
    auto cylinder3 = build_cylinder3<Scalar>(config);

    // With no extra objects, type doesn't matter: it is just the tray
    if (config.num == 0) {
        return {tray};
    }

    if (config.arrangement == TrayBalanceConfiguration::Arrangement::Flat) {
        std::cerr << "Using Flat arrangement with " << config.num << " objects."
                  << std::endl;
        if (config.num == 1) {
            auto composite_tray_cylinder1 =
                BalancedObject<Scalar>::compose({tray, cylinder1});
            return {composite_tray_cylinder1, cylinder1};
        } else if (config.num == 2) {
            auto composite_tray_cylinder12 =
                BalancedObject<Scalar>::compose({tray, cylinder1, cylinder2});
            return {composite_tray_cylinder12, cylinder1, cylinder2};
        } else if (config.num == 3) {
            auto composite_tray_cylinder123 = BalancedObject<Scalar>::compose(
                {tray, cylinder1, cylinder2, cylinder3});
            return {composite_tray_cylinder123, cylinder1, cylinder2,
                    cylinder3};
        }
        throw std::runtime_error("Unsupported object configuration.");
    } else if (config.arrangement ==
               TrayBalanceConfiguration::Arrangement::Stacked) {
        std::cerr << "Using Stacked arrangement with " << config.num
                  << " objects." << std::endl;
        if (config.num == 1) {
            auto composite_tray_cylinder1 =
                BalancedObject<Scalar>::compose({tray, cylinder1});
            return {composite_tray_cylinder1, cylinder1};
        } else if (config.num == 2) {
            auto composite_tray_cylinder12 =
                BalancedObject<Scalar>::compose({tray, cylinder1, cylinder2});
            auto composite_cylinder12 =
                BalancedObject<Scalar>::compose({cylinder1, cylinder2});
            return {composite_tray_cylinder12, composite_cylinder12, cylinder2};
        } else if (config.num == 3) {
            auto composite_tray_cylinder123 = BalancedObject<Scalar>::compose(
                {tray, cylinder1, cylinder2, cylinder3});
            auto composite_cylinder123 = BalancedObject<Scalar>::compose(
                {cylinder1, cylinder2, cylinder3});
            auto composite_cylinder23 =
                BalancedObject<Scalar>::compose({cylinder2, cylinder3});

            return {composite_tray_cylinder123, composite_cylinder123,
                    composite_cylinder23, cylinder3};
        }
        throw std::runtime_error("Unsupported object configuration.");
    }
    throw std::runtime_error("Unsupported object configuration.");
}

}  // namespace mobile_manipulator
}  // namespace ocs2
