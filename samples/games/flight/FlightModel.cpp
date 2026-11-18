/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Flight model of the flight simulator tech demo (forces, body rates, ground contact).

#include "games/flight/FlightModel.h"

namespace Saiga
{
namespace Game
{
namespace
{
constexpr double kGravity = 9.81;
}

void FlightModel::reset(const Vec3& position, double heading, double speed)
{
    state             = AircraftState();
    state.position    = position;
    state.orientation = yawRotation(heading);
    state.velocity    = forwardOf(state.orientation) * speed;
}

double FlightModel::groundHeight() const
{
    return ground ? ground(state.position(0), state.position(2)) : -1e30;
}

double FlightModel::angleOfAttack() const
{
    Vec3 vb  = state.orientation.inverse() * state.velocity;
    double f = -vb(2);
    if (std::abs(f) + std::abs(vb(1)) < 1e-6) return 0;
    return std::atan2(-vb(1), f);
}

double FlightModel::sideSlip() const
{
    Vec3 vb  = state.orientation.inverse() * state.velocity;
    double f = -vb(2);
    if (std::abs(f) + std::abs(vb(0)) < 1e-6) return 0;
    return std::atan2(vb(0), f);
}

double FlightModel::bankAngle() const
{
    // Full range (-pi, pi]: an inverted aircraft has a bank of about +-pi, not 0 (asin would fold it back)
    return std::atan2(-right()(1), up()(1));
}

double FlightModel::pitchAngle() const
{
    return std::asin(clampd(forward()(1), -1, 1));
}

double FlightModel::liftCoefficient(double alpha) const
{
    const auto& p = params;
    double a      = std::abs(alpha);
    double sign   = alpha < 0 ? -1 : 1;
    if (a <= p.stallAngle) return p.cl0 + p.clAlpha * alpha;
    // After the stall the lift breaks down: from the stall value to 40% of it 10 degrees later, then flat
    double atStall = p.cl0 + p.clAlpha * p.stallAngle * sign;
    double t       = clampd((a - p.stallAngle) / (10.0 * kPi / 180.0), 0, 1);
    return atStall * (1.0 - 0.6 * t);
}

double FlightModel::trimSpeed() const
{
    const auto& p = params;
    return std::sqrt(2 * p.mass * kGravity / (p.airDensity * p.wingArea * p.cl0));
}

double FlightModel::trimThrottle(double speed) const
{
    const auto& p = params;
    double q      = 0.5 * p.airDensity * speed * speed * p.wingArea;
    // At the lift coefficient that carries the weight at this speed
    double cl = p.mass * kGravity / std::max(q, 1e-6);
    double d  = q * (p.cd0 + p.inducedDrag * cl * cl);
    return clampd(d / p.maxThrust, 0, 1);
}

void FlightModel::step(const AircraftControls& c, double dt)
{
    auto& s       = state;
    const auto& p = params;
    if (s.crashed) return;

    double throttle = clampd(c.throttle, 0, 1);
    Vec3 fwd = forward(), rgt = right();
    double speed = s.velocity.norm();

    // Rotation: body rates follow the controls (effective with airspeed) and the stability terms
    double effect = clampd(speed / p.referenceSpeed, 0, 1.3);
    double alpha = angleOfAttack(), beta = sideSlip();
    if (s.onGround) alpha = beta = 0;
    Vec3 target(p.pitchRate * clampd(c.pitch, -1, 1) * effect - p.pitchStability * alpha * effect,
                -p.yawRate * clampd(c.yaw, -1, 1) * effect - p.yawStability * beta * effect,
                -p.rollRate * clampd(c.roll, -1, 1) * effect);
    if (s.onGround)
    {
        // On the gear: no rolling, pitching only to rotate for take off
        target(2) = 0;
        target(0) = std::max(target(0), 0.0) * (speed > 0.8 * trimSpeed() ? 1.0 : 0.0);
    }
    s.bodyRates += (target - s.bodyRates) * smoothFactor(p.rateResponse, dt);
    s.orientation = integrateBody(s.orientation, s.bodyRates, dt);

    // Forces
    Vec3 force = fwd * (throttle * p.maxThrust) + Vec3(0, -p.mass * kGravity, 0);
    if (speed > 0.1)
    {
        Vec3 vdir = s.velocity / speed;
        double q  = 0.5 * p.airDensity * speed * speed * p.wingArea;
        double cl = liftCoefficient(alpha);
        // Perpendicular to the velocity in the plane of the nose and the up axis
        Vec3 liftDir = rgt.cross(vdir);
        if (liftDir.norm() > 1e-6) force += liftDir.normalized() * (q * cl);
        force -= vdir * (q * (p.cd0 + p.inducedDrag * cl * cl));
        // Side slip
        Vec3 vb = s.orientation.inverse() * s.velocity;
        force -= rgt * (p.sideForce * vb(0));
    }

    s.velocity += force / p.mass * dt;
    s.position += s.velocity * dt;

    // Ground contact
    double g = groundHeight() + p.gearHeight;
    if (s.position(1) <= g)
    {
        bool hard = s.velocity(1) < -p.crashSinkRate || std::abs(bankAngle()) > p.crashBank ||
                    pitchAngle() < -p.crashPitch || pitchAngle() > p.crashPitch;
        s.position(1) = g;
        if (hard)
        {
            s.crashed   = true;
            s.velocity  = Vec3(0, 0, 0);
            s.bodyRates = Vec3(0, 0, 0);
            return;
        }
        s.onGround    = true;
        s.velocity(1) = std::max(s.velocity(1), 0.0);
        // Rolling friction, wheel brakes with the throttle at 0
        double friction = (throttle < 0.05 ? 0.4 : 0.03) * kGravity * dt;
        double hs       = Vec3(s.velocity(0), 0, s.velocity(2)).norm();
        if (hs > 1e-6) s.velocity -= Vec3(s.velocity(0), 0, s.velocity(2)) * (std::min(friction, hs) / hs);
        // Level the wings and keep the nose from digging in
        Vec3 f         = forward();
        double heading = headingOf(f);
        double pitch   = std::max(pitchAngle(), 0.0);
        s.orientation  = yawRotation(heading) * axisAngle(Vec3(1, 0, 0), pitch);
    }
    else if (s.position(1) > g + 0.2)
    {
        s.onGround = false;
    }
}

}  // namespace Game
}  // namespace Saiga
