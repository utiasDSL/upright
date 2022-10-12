#pragma once

namespace upright {

template <typename Scalar>
Scalar circle_r_tau(Scalar radius) {
    return Scalar(2.0) * radius / Scalar(3.0);
}

template <typename Scalar>
Scalar alpha_rect(Scalar w, Scalar h) {
    // alpha_rect for half of the rectangle
    Scalar d = sqrt(h * h + w * w);
    return (w * h * d + w * w * w * (log(h + d) - log(w))) / 12.0;
}

template <typename Scalar>
Scalar rectangle_r_tau(Scalar w, Scalar h) {
    // (see pushing notes)
    return (alpha_rect(w, h) + alpha_rect(h, w)) / (w * h);
}

template <typename Scalar>
Mat3<Scalar> cylinder_inertia_matrix(Scalar mass, Scalar radius,
                                     Scalar height) {
    // diagonal elements
    Scalar xx =
        mass * (Scalar(3.0) * radius * radius + height * height) / Scalar(12.0);
    Scalar yy = xx;
    Scalar zz = Scalar(0.5) * mass * radius * radius;

    // construct the inertia matrix
    Mat3<Scalar> I = Mat3<Scalar>::Zero();
    I.diagonal() << xx, yy, zz;
    return I;
}

template <typename Scalar>
Mat3<Scalar> cuboid_inertia_matrix(Scalar mass,
                                   const Vec3<Scalar>& side_lengths) {
    Scalar lx = side_lengths(0);
    Scalar ly = side_lengths(1);
    Scalar lz = side_lengths(2);

    Scalar xx = mass * (squared(ly) + squared(lz)) / Scalar(12.0);
    Scalar yy = mass * (squared(lx) + squared(lz)) / Scalar(12.0);
    Scalar zz = mass * (squared(lx) + squared(ly)) / Scalar(12.0);

    Mat3<Scalar> I = Mat3<Scalar>::Zero();
    I.diagonal() << xx, yy, zz;
    return I;
}

template <typename Scalar>
RigidBody<Scalar> RigidBody<Scalar>::compose(
    const std::vector<RigidBody<Scalar>>& bodies) {
    // Compute new mass and center of mass.
    Scalar mass = Scalar(0);
    Vec3<Scalar> com = Vec3<Scalar>::Zero();
    for (int i = 0; i < bodies.size(); ++i) {
        mass += bodies[i].mass;
        com += bodies[i].mass * bodies[i].com;
    }
    com = com / mass;

    // Parallel axis theorem to compute new moment of inertia.
    Mat3<Scalar> inertia = Mat3<Scalar>::Zero();
    for (int i = 0; i < bodies.size(); ++i) {
        Vec3<Scalar> r = bodies[i].com - com;
        Mat3<Scalar> R = skew3(r);
        inertia += bodies[i].inertia - bodies[i].mass * R * R;
    }

    return RigidBody<Scalar>(mass, inertia, com);
}

template <typename Scalar>
RigidBody<Scalar> RigidBody<Scalar>::from_parameters(
    const VecX<Scalar>& parameters, const size_t index) {
    Scalar mass(parameters(index));
    Vec3<Scalar> com(parameters.template segment<3>(index + 1));
    VecX<Scalar> I_vec(parameters.template segment<9>(index + 4));
    Mat3<Scalar> inertia(Eigen::Map<Mat3<Scalar>>(I_vec.data(), 3, 3));
    return RigidBody(mass, inertia, com);
}

template <typename Scalar>
VecX<Scalar> RigidBody<Scalar>::get_parameters() const {
    VecX<Scalar> p(num_parameters());
    VecX<Scalar> I_vec(
        Eigen::Map<const VecX<Scalar>>(inertia.data(), inertia.size()));
    p << mass, com, I_vec;
    return p;
}

}  // namespace upright
