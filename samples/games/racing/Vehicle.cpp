/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Vehicle model of the racing tech demo (rigid body on four spring-damper wheels, tire friction).

#include "games/racing/Vehicle.h"

namespace Saiga
{
namespace Game
{
namespace
{
constexpr double kGravity = 9.81;
}

void Vehicle::resetOnGround(double x, double z, double heading)
{
    reset(Vec3(x, (ground ? ground(x, z) : 0) + restingHeight(), z), heading);
}

void Vehicle::reset(const Vec3& position, double heading)
{
    state             = VehicleKinematics();
    state.orientation = yawRotation(heading);
    state.position    = position;
    // Each wheel's actual compression at the spawn pose, so the damper does not see a jump in the first step.
    // Regression: all wheels got the flat-ground resting compression, which on a slope differs per wheel and gave the
    // stationary car an upward kick.
    Mat3 R = state.orientation.matrix();
    for (int i = 0; i < 4; ++i)
    {
        Vec3 mount           = state.position + R * params.wheelMounts[i];
        double g             = ground ? ground(mount(0), mount(2)) : 0;
        double comp          = params.restLength + params.wheelRadius - (mount(1) - g);
        state.compression[i] = std::min(std::max(comp, 0.0), params.restLength + params.wheelRadius);
    }
}

double Vehicle::restingHeight() const
{
    const auto& p = params;
    double mountY = 0;
    for (auto& m : p.wheelMounts) mountY += m(1) / 4;
    // Each spring carries a quarter of the weight
    return p.restLength + p.wheelRadius - p.mass * kGravity / (4 * p.stiffness) - mountY;
}

int Vehicle::wheelsOnGround() const
{
    int n = 0;
    for (bool c : state.contact) n += c;
    return n;
}

void Vehicle::step(const VehicleControls& c, double dt)
{
    int n = std::max(params.subSteps, 1);
    for (int i = 0; i < n; ++i) subStep(c, dt / n);
}

void Vehicle::subStep(const VehicleControls& c, double dt)
{
    auto& s       = state;
    const auto& p = params;

    double throttle = clampd(c.throttle, -1, 1);
    // Steering: less at speed, smoothed
    double speed       = s.velocity.norm();
    double steerTarget = clampd(c.steer, -1, 1) * p.maxSteer / (1.0 + speed / 25.0);
    s.steerAngle += (steerTarget - s.steerAngle) * smoothFactor(8, dt);

    Mat3 R   = s.orientation.matrix();
    Vec3 up  = R * Vec3(0, 1, 0);
    Vec3 fwd = R * Vec3(0, 0, -1);

    Vec3 force  = Vec3(0, -p.mass * kGravity, 0);
    Vec3 torque = Vec3(0, 0, 0);
    double vf   = s.velocity.dot(fwd);

    for (int i = 0; i < 4; ++i)
    {
        Vec3 r     = R * p.wheelMounts[i];
        Vec3 mount = s.position + r;
        double g   = ground ? ground(mount(0), mount(2)) : 0;
        // Distance along the body's down axis from the mount to the ground
        double dist = (mount(1) - g) / std::max(up(1), 0.2);
        double comp = p.restLength + p.wheelRadius - dist;
        double prev = s.compression[i];
        // Regression: on its side the low mounts were underground and every such wheel reported full compression,
        // which threw the car into the air. A wheel of a tipped body has no contact (the body rests on the ground).
        if (comp <= 0 || up(1) < 0.3)
        {
            s.compression[i] = 0;
            s.contact[i]     = false;
            continue;
        }
        comp             = std::min(comp, p.restLength + p.wheelRadius);
        s.compression[i] = comp;
        s.contact[i]     = true;
        double load      = std::max(p.stiffness * comp + p.damping * (comp - prev) / dt, 0.0);

        Vec3 n         = ground ? groundNormal(ground, mount(0), mount(2), 0.5) : Vec3(0, 1, 0);
        Vec3 contactPt = Vec3(mount(0), g, mount(2));
        Vec3 arm       = contactPt - s.position;
        Vec3 vel       = s.velocity + s.angularVelocity.cross(arm);

        // Wheel directions in the ground plane (front wheels steered; positive steer turns right)
        Vec3 wf = fwd;
        if (i < 2) wf = axisAngle(up, -s.steerAngle) * fwd;
        wf = (wf - n * wf.dot(n));
        if (wf.norm() < 1e-6) continue;
        wf.normalize();
        Vec3 ws = n.cross(wf).normalized() * -1.0;

        double maxF = p.friction * load;
        // Longitudinal: drive (rear), brakes, rolling resistance
        double vLong = vel.dot(wf);
        // Rolling resistance, engine braking without throttle
        double fl = -vLong * (p.rollingResistance + (throttle == 0 ? 120.0 : 0.0));
        if (throttle > 0 && i >= 2)
            fl += 0.5 * p.engineForce * throttle * clampd(1.0 - vf / p.maxSpeed, 0, 1);
        else if (throttle < 0)
        {
            if (vf > 1.0)
                fl -= 0.25 * p.brakeForce * -throttle * (vLong > 0 ? 1 : -1);
            else if (i >= 2)
                fl += 0.5 * 0.4 * p.engineForce * throttle;  // reverse
        }
        // Lateral: against the slip, saturating
        double vLat = vel.dot(ws);
        double fs   = -maxF * vLat / (std::abs(vLat) + p.slipSpeed);
        // Friction circle
        double ft = std::sqrt(fl * fl + fs * fs);
        if (ft > maxF && ft > 0)
        {
            fl *= maxF / ft;
            fs *= maxF / ft;
        }
        // The load along the ground normal (along the tilted body axis it would push the car sideways at rest)
        Vec3 fn = n * load, ft2 = wf * fl + ws * fs;
        force += fn + ft2;
        // The tire forces act at the roll centre (below the center of mass, between the wheels' mounts)
        Vec3 rollArm = R * Vec3(p.wheelMounts[i](0), -p.rollCenterDepth, p.wheelMounts[i](2));
        torque += arm.cross(fn) + rollArm.cross(ft2);

        s.spin[i] += vLong / p.wheelRadius * dt;
    }

    // Air drag
    force -= s.velocity * (p.airDrag * speed);

    // Integrate (inertia of the chassis box in the body frame)
    const Vec3& h = p.halfExtents;
    Vec3 I(p.mass / 3.0 * (h(1) * h(1) + h(2) * h(2)), p.mass / 3.0 * (h(0) * h(0) + h(2) * h(2)),
           p.mass / 3.0 * (h(0) * h(0) + h(1) * h(1)));
    Vec3 wb = R.transpose() * s.angularVelocity;
    Vec3 tb = R.transpose() * torque;
    Vec3 Iw(I(0) * wb(0), I(1) * wb(1), I(2) * wb(2));
    Vec3 dw = tb - wb.cross(Iw);
    wb += Vec3(dw(0) / I(0), dw(1) / I(1), dw(2) / I(2)) * dt;
    wb *= 1.0 - smoothFactor(0.5, dt);
    s.angularVelocity = R * wb;

    s.velocity += force / p.mass * dt;
    s.position += s.velocity * dt;
    s.orientation = integrateWorld(s.orientation, s.angularVelocity, dt);

    // The chassis does not sink into the ground: the extent of the body box towards the ground (along the ground
    // normal, so a car on a slope is not lifted). Regression: a fixed 0.25 m let a car on its side or roof sink half
    // into the ground.
    double g       = ground ? ground(s.position(0), s.position(2)) : 0;
    Vec3 n         = ground ? groundNormal(ground, s.position(0), s.position(2), 0.5) : Vec3(0, 1, 0);
    Mat3 Rn        = s.orientation.matrix();
    double ny      = n.dot(Rn.col(1));
    double support = std::abs(n.dot(Rn.col(0))) * p.halfExtents(0) + (ny >= 0 ? ny * p.bodyBottom : -ny * p.bodyTop) +
                     std::abs(n.dot(Rn.col(2))) * p.halfExtents(2);
    support = std::max(support, 0.25) / std::max(n(1), 0.2);
    if (s.position(1) < g + support)
    {
        s.position(1) = g + support;
        s.velocity(1) = std::max(s.velocity(1), 0.0);
        // Regression: on its side or roof the chassis slid on without friction
        double damp =
            smoothFactor(p.chassisFriction * kGravity / std::max(std::hypot(s.velocity(0), s.velocity(2)), 1.0), dt);
        s.velocity(0) *= 1 - damp;
        s.velocity(2) *= 1 - damp;
        s.angularVelocity *= 1 - damp;
    }
}

}  // namespace Game
}  // namespace Saiga
