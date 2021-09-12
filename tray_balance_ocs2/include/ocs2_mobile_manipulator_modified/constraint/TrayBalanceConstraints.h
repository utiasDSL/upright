#pragma once

#include <memory>

// #include <ocs2_mobile_manipulator_modified/constraint/TrayBalanceUtil.h>
#include <ocs2_mobile_manipulator_modified/definitions.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include <tray_balance_constraints/inequality_constraints.h>

namespace ocs2 {
namespace mobile_manipulator {

// TODO can this be a class? convenient for defining, say, num_constraints
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
BalancedObject<Scalar> build_cuboid_object2() {
    Scalar cuboid_mass(0.5);
    Scalar cuboid_mu(0.5);
    Scalar cuboid_com_height(0.075);
    Scalar cuboid_zmp_margin(0.0);
    Vec2<Scalar> cuboid_support_offset = Vec2<Scalar>::Zero();
    Vec3<Scalar> cuboid_side_lengths(Scalar(0.15), Scalar(0.15),
                                     cuboid_com_height * 2);
    Mat3<Scalar> cuboid_inertia =
        cuboid_inertia_matrix(cuboid_mass, cuboid_side_lengths);
    Vec3<Scalar> cuboid_com(Scalar(0), Scalar(0), Scalar(0.265));
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
BalancedObject<Scalar> build_cuboid_object3() {
    Scalar cuboid_mass(0.5);
    Scalar cuboid_mu(0.5);
    Scalar cuboid_com_height(0.075);
    Scalar cuboid_zmp_margin(0.0);
    Vec2<Scalar> cuboid_support_offset = Vec2<Scalar>::Zero();
    Vec3<Scalar> cuboid_side_lengths(Scalar(0.1), Scalar(0.1),
                                     cuboid_com_height * 2);
    Mat3<Scalar> cuboid_inertia =
        cuboid_inertia_matrix(cuboid_mass, cuboid_side_lengths);
    Vec3<Scalar> cuboid_com(Scalar(0), Scalar(0), Scalar(0.415));
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
BalancedObject<Scalar> build_cylinder1() {
    Scalar cylinder_mass(0.5);
    Scalar cylinder_mu(0.5);
    Scalar cylinder_com_height(0.075);
    Scalar cylinder_radius(0.05);
    Scalar cylinder_height = cylinder_com_height * 2;
    Scalar cylinder_r_tau = circle_r_tau(cylinder_radius);
    Scalar cylinder_zmp_margin(0.0);
    Vec2<Scalar> cylinder_support_offset = Vec2<Scalar>::Zero();

    // Scalar ee_side_length(0.2);
    // Vec2<Scalar> com_xy =
    //     equilateral_triangle_cup_location(ee_side_length, Scalar(0.08), 0);
    Vec3<Scalar> cylinder_com;
    cylinder_com << Scalar(0), Scalar(0), Scalar(0.115);

    Mat3<Scalar> cylinder_inertia = cylinder_inertia_matrix(
        cylinder_mass, cylinder_radius, cylinder_height);
    RigidBody<Scalar> cylinder_body(cylinder_mass, cylinder_inertia,
                                    cylinder_com);

    // CircleSupportArea<Scalar> cylinder_support_area(
    //     cylinder_radius, cylinder_support_offset, cylinder_zmp_margin);

    // Approximate support area using the large rectangle that fits in the
    // circle: this is better for solver stability.
    Scalar s = cylinder_radius * Scalar(sqrt(2.0));
    std::vector<Vec2<Scalar>> cylinder_vertices = cuboid_support_vertices(s, s);
    PolygonSupportArea<Scalar> cylinder_support_area(
        cylinder_vertices, cylinder_support_offset, cylinder_zmp_margin);

    BalancedObject<Scalar> cylinder(cylinder_body, cylinder_com_height,
                                    cylinder_support_area, cylinder_r_tau,
                                    cylinder_mu);
    return cylinder;
}

template <typename Scalar>
BalancedObject<Scalar> build_cylinder2() {
    Scalar cylinder_mass(0.5);
    Scalar cylinder_mu(0.5);
    Scalar cylinder_com_height(0.075);
    Scalar cylinder_radius(0.05);
    Scalar cylinder_height = cylinder_com_height * 2;
    Scalar cylinder_r_tau = circle_r_tau(cylinder_radius);
    Scalar cylinder_zmp_margin(0.0);
    Vec2<Scalar> cylinder_support_offset = Vec2<Scalar>::Zero();

    // Scalar ee_side_length(0.2);
    // Vec2<Scalar> com_xy =
    //     equilateral_triangle_cup_location(ee_side_length, Scalar(0.08), 1);
    Vec3<Scalar> cylinder_com;
    cylinder_com << Scalar(0), Scalar(0), Scalar(0.265);

    Mat3<Scalar> cylinder_inertia = cylinder_inertia_matrix(
        cylinder_mass, cylinder_radius, cylinder_height);
    RigidBody<Scalar> cylinder_body(cylinder_mass, cylinder_inertia,
                                    cylinder_com);

    Scalar s = cylinder_radius * Scalar(sqrt(2.0));
    std::vector<Vec2<Scalar>> cylinder_vertices = cuboid_support_vertices(s, s);
    PolygonSupportArea<Scalar> cylinder_support_area(
        cylinder_vertices, cylinder_support_offset, cylinder_zmp_margin);

    BalancedObject<Scalar> cylinder(cylinder_body, cylinder_com_height,
                                    cylinder_support_area, cylinder_r_tau,
                                    cylinder_mu);
    return cylinder;
}

template <typename Scalar>
BalancedObject<Scalar> build_cylinder3() {
    Scalar cylinder_mass(0.5);
    Scalar cylinder_mu(0.5);
    Scalar cylinder_com_height(0.075);
    Scalar cylinder_radius(0.05);
    Scalar cylinder_height = cylinder_com_height * 2;
    Scalar cylinder_r_tau = circle_r_tau(cylinder_radius);
    Scalar cylinder_zmp_margin(0.0);
    Vec2<Scalar> cylinder_support_offset = Vec2<Scalar>::Zero();

    // Scalar ee_side_length(0.2);
    // Vec2<Scalar> com_xy =
    //     equilateral_triangle_cup_location(ee_side_length, Scalar(0.08), 2);
    Vec3<Scalar> cylinder_com;
    cylinder_com << Scalar(0), Scalar(0), Scalar(0.415);

    Mat3<Scalar> cylinder_inertia = cylinder_inertia_matrix(
        cylinder_mass, cylinder_radius, cylinder_height);
    RigidBody<Scalar> cylinder_body(cylinder_mass, cylinder_inertia,
                                    cylinder_com);

    Scalar s = cylinder_radius * Scalar(sqrt(2.0));
    std::vector<Vec2<Scalar>> cylinder_vertices = cuboid_support_vertices(s, s);

    // Rotate support area by 45 degrees
    // Mat2<Scalar> R;
    // Scalar PI_4 = Scalar(M_PI / 4.0);
    // R << cos(PI_4), -sin(PI_4), sin(PI_4), cos(PI_4);
    // for (int i = 0; i < cylinder_vertices.size(); ++i) {
    //     cylinder_vertices[i] = R * cylinder_vertices[i];
    // }

    PolygonSupportArea<Scalar> cylinder_support_area(
        cylinder_vertices, cylinder_support_offset, cylinder_zmp_margin);

    BalancedObject<Scalar> cylinder(cylinder_body, cylinder_com_height,
                                    cylinder_support_area, cylinder_r_tau,
                                    cylinder_mu);
    return cylinder;
}

class TrayBalanceConstraints final : public StateInputConstraintCppAd {
   public:
    using ad_quaternion_t =
        PinocchioEndEffectorKinematicsCppAd::ad_quaternion_t;
    using ad_rotmat_t = PinocchioEndEffectorKinematicsCppAd::ad_rotmat_t;

    using ad_vec2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;
    using ad_vec3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
    using ad_mat3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;

    TrayBalanceConstraints(
        const PinocchioEndEffectorKinematicsCppAd& pinocchioEEKinematics)
        : StateInputConstraintCppAd(ConstraintOrder::Linear),
          pinocchioEEKinPtr_(pinocchioEEKinematics.clone()) {
        if (pinocchioEEKinematics.getIds().size() != 1) {
            throw std::runtime_error(
                "[EndEffectorConstraint] endEffectorKinematics has wrong "
                "number of "
                "end effector IDs.");
        }
        // initialize everything, mostly the CppAD interface
        initialize(STATE_DIM, INPUT_DIM, 0, "tray_balance_constraints",
                   "/tmp/ocs2", true, true);
    }

    // TrayBalanceConstraints() override = default;

    TrayBalanceConstraints* clone() const override {
        return new TrayBalanceConstraints(*pinocchioEEKinPtr_);
    }

    size_t getNumConstraints(scalar_t time) const override {
        // TODO this depends on the configuration at hand
        size_t n_tray_con = 2 + 3;
        size_t n_cuboid_con = 2 + 4;
        // size_t n_cylinder_con = 2 + 1;
        return n_tray_con + 0 * n_cuboid_con;
    }

    size_t getNumConstraints() const { return getNumConstraints(0); }

   protected:
    ad_vector_t constraintFunction(
        ad_scalar_t time, const ad_vector_t& state, const ad_vector_t& input,
        const ad_vector_t& parameters) const override {
        ad_rotmat_t C_we = pinocchioEEKinPtr_->getOrientationCppAd(state);
        ad_vector_t angular_vel =
            pinocchioEEKinPtr_->getAngularVelocityCppAd(state, input);
        ad_vector_t angular_acc =
            pinocchioEEKinPtr_->getAngularAccelerationCppAd(state, input);
        ad_vector_t linear_acc =
            pinocchioEEKinPtr_->getAccelerationCppAd(state, input);

        BalancedObject<ad_scalar_t> tray = build_tray_object<ad_scalar_t>();

        ///// Boxes //////

        // BalancedObject<ad_scalar_t> cuboid1 =
        //     build_cuboid_object<ad_scalar_t>();
        // BalancedObject<ad_scalar_t> cuboid2 =
        //     build_cuboid_object2<ad_scalar_t>();
        // BalancedObject<ad_scalar_t> cuboid3 =
        //     build_cuboid_object3<ad_scalar_t>();

        // auto composite_tray_cuboid1 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cuboid1});

        // auto composite_tray_cuboid12 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cuboid1, cuboid2});
        // auto composite_cuboid12 =
        //     BalancedObject<ad_scalar_t>::compose({cuboid1, cuboid2});

        // auto composite_tray_cuboid123 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cuboid1, cuboid2,
        //     cuboid3});
        // auto composite_cuboid123 =
        //     BalancedObject<ad_scalar_t>::compose({cuboid1, cuboid2,
        //     cuboid3});
        // auto composite_cuboid23 =
        //     BalancedObject<ad_scalar_t>::compose({cuboid2, cuboid3});
        //
        // ad_vector_t constraints = balancing_constraints<ad_scalar_t>(
        //     C_we, angular_vel, linear_acc, angular_acc,
        //  {composite_tray_cuboid123, composite_cuboid123, composite_cuboid23,
        //  cuboid3});

        ///// Stack /////

        // BalancedObject<ad_scalar_t> cylinder1 = build_cylinder1<ad_scalar_t>();
        // BalancedObject<ad_scalar_t> cylinder2 = build_cylinder2<ad_scalar_t>();
        // BalancedObject<ad_scalar_t> cylinder3 =
        // build_cylinder3<ad_scalar_t>();

        // auto composite_tray_cylinder1 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cylinder1});

        // auto composite_tray_cylinder12 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cylinder1, cylinder2});
        // auto composite_cylinder12 =
        //     BalancedObject<ad_scalar_t>::compose({cylinder1, cylinder2});

        // auto composite_tray_cylinder123 =
        // BalancedObject<ad_scalar_t>::compose(
        //     {tray, cylinder1, cylinder2, cylinder3});
        // auto composite_cylinder123 = BalancedObject<ad_scalar_t>::compose(
        //     {cylinder1, cylinder2, cylinder3});
        // auto composite_cylinder23 =
        //     BalancedObject<ad_scalar_t>::compose({cylinder2, cylinder3});

        ad_vector_t constraints = balancing_constraints<ad_scalar_t>(
            C_we, angular_vel, linear_acc, angular_acc, {tray});

        ///// Cups /////

        // BalancedObject<ad_scalar_t> cylinder1 =
        // build_cylinder1<ad_scalar_t>(); BalancedObject<ad_scalar_t> cylinder2
        // = build_cylinder2<ad_scalar_t>(); BalancedObject<ad_scalar_t>
        // cylinder3 = build_cylinder3<ad_scalar_t>();

        // auto composite_tray_cylinder1 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cylinder1});

        // auto composite_tray_cylinder12 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cylinder1,
        //     cylinder2});

        // auto composite_tray_cylinder123 =
        //     BalancedObject<ad_scalar_t>::compose({tray, cylinder1, cylinder2,
        //     cylinder3});

        // ad_vector_t constraints = balancing_constraints<ad_scalar_t>(
        //     C_we, angular_vel, linear_acc, angular_acc,
        //     {composite_tray_cylinder1, cylinder1});

        return constraints;
    }

   private:
    TrayBalanceConstraints(const TrayBalanceConstraints& other) = default;

    std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> pinocchioEEKinPtr_;
};

}  // namespace mobile_manipulator
}  // namespace ocs2
