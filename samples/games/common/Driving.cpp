/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The driver-entity helpers of the tech demos (S16b).

#include "games/common/Driving.h"

#include "saiga/engine/app/EngineAppSetup.h"
#include "saiga/engine/gameplay/DestroyOwnedBy.h"

#include <iostream>

namespace Saiga
{
namespace Game
{
using namespace Engine;

void registerDrivingComponents(Registry& r)
{
    r.registerComponent<Driver>(Replicate::Host);
    r.registerComponent<VehicleSeat>(Replicate::Host);
    r.registerComponent<DemoPilot>(Replicate::Never);
}

Entity drivenVehicle(const World& w, Entity driver)
{
    if (!driver.valid() || !w.alive(driver) || !w.has<Driver>(driver)) return Entity{};
    const Entity v = w.read<Driver>(driver).vehicle;
    return v.valid() && w.alive(v) ? v : Entity{};
}

bool drives(const World& w, Entity player, Entity vehicle)
{
    return player.valid() && drivenVehicle(w, player) == vehicle;
}

void removeDriver(World& host, uint8_t peer)
{
    destroyOwnedBy(host, peer);
    host.view<VehicleSeat>().each(
        [&](Entity, VehicleSeat& seat)
        {
            if (seat.driver.valid() && !host.alive(seat.driver)) seat.driver = Entity{};
        });
}

bool clampCoopDemoDelay(AppOptions& options, std::string&)
{
    if (options.coop && options.has("--demo") && options.coopDelay > 8)
    {
        std::cout << "--coop --demo: the delay is clamped to 8 steps (the demo pilots' gains)" << std::endl;
        options.coopDelay = 8;
    }
    return true;
}

}  // namespace Game
}  // namespace Saiga
