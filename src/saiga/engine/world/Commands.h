/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <memory>
#include <utility>

#include <type_traits>

namespace Saiga
{
namespace Engine
{
class World;

/**
 * A deferred structural change (docs/engine_design.md §7.6; engine-internal): recorded during a step, applied at the
 * end of the phase in recording order. A closure that may own move-only values (a component moved into an add).
 */
class Command
{
   public:
    virtual ~Command()           = default;
    virtual void apply(World& w) = 0;
};

template <class F>
class ClosureCommand final : public Command
{
   public:
    explicit ClosureCommand(F f) : f(std::move(f)) {}
    void apply(World& w) override { f(w); }

   private:
    F f;
};

template <class F>
std::unique_ptr<Command> makeCommand(F&& f)
{
    return std::make_unique<ClosureCommand<std::decay_t<F>>>(std::forward<F>(f));
}

}  // namespace Engine
}  // namespace Saiga
