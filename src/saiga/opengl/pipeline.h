/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/rendering/PipelineState.h"
#include "saiga/opengl/opengl.h"

#include <memory>
#include <string>

namespace Saiga
{
class Shader;

// Sets the complete fixed-function state of `state` in the current context: every value, also those that equal
// OpenGL's defaults, so the result doesn't depend on what ran before. stencilReference is the reference value of the
// stencil test (dynamic state in Vulkan). Primitive restart needs OpenGL 4.3 or GL_ARB_ES3_compatibility; without it, a
// state with primitive restart is a fatal error.
//
// With the PipelineStateCache enabled, only the values that differ from the last applied state are set.
SAIGA_OPENGL_API void applyPipelineState(const PipelineState& state, uint32_t stencilReference = 0);

// The state the current context has. Code that draws inside a pass of someone else (in the forward pass of a renderer)
// changes a copy of it, applies that and applies this again afterwards. With the PipelineStateCache enabled it is the
// cached state; otherwise it is read back with queryPipelineState, so it also has changes made with gl calls directly.
SAIGA_OPENGL_API PipelineState currentPipelineState();
SAIGA_OPENGL_API uint32_t currentStencilReference();

// Clears the given buffers of the bound draw framebuffer. glClear obeys the write masks and the scissor test, so this
// first enables writing all color channels, depth and all stencil bits, and the scissor test only with scissor (the
// rectangle stays as set). It leaves that state behind; the next pass sets its own.
SAIGA_OPENGL_API void clearFramebuffer(ClearBufferMask buffers, bool scissor = false);

// The fixed-function state the current context has, read back with glGet. It reads many values and stalls the
// pipeline: for tests, the validation mode and currentPipelineState without the cache. OpenGL clamps the stencil
// reference in queries to the stencil bits of the bound framebuffer, 0 without a stencil buffer.
SAIGA_OPENGL_API PipelineState queryPipelineState(uint32_t* stencilReference = nullptr);

// Lists the fields in which a and b differ, empty if they are equal.
SAIGA_OPENGL_API std::string pipelineStateDifference(const PipelineState& a, const PipelineState& b);

// Remembers the state applyPipelineState set last, so that it can skip the calls that wouldn't change anything.
//
// Disabled by default: the cache is only correct if all state changes go through applyPipelineState and
// clearFramebuffer. Code that changes the state with gl calls directly must call invalidate() afterwards, or restore
// what it found before the next applyPipelineState. The cache belongs to the current thread; call invalidate() after
// making another context current on it.
//
// The validation mode reads the state back after every applyPipelineState and asserts that it is the requested one,
// and, with the cache enabled, that nobody changed it since the last call. It is slow; use it to find code that
// bypasses the pipeline state.
struct SAIGA_OPENGL_API PipelineStateCache
{
    static void setEnabled(bool enabled);
    static bool enabled();

    static void setValidation(bool validation);
    static bool validation();

    // The next applyPipelineState sets every value again.
    static void invalidate();
};

// A shader with the fixed-function state it draws with.
class SAIGA_OPENGL_API GraphicsPipeline
{
   public:
    GraphicsPipeline() = default;
    GraphicsPipeline(std::shared_ptr<Shader> shader, const PipelineState& state)
        : shader(std::move(shader)), state(state)
    {
    }

    // Applies the state and binds the shader. False if the shader can't be used (see Shader::bind); the state is
    // applied either way.
    [[nodiscard]] bool bind(uint32_t stencilReference = 0);
    void unbind();

    std::shared_ptr<Shader> shader;
    PipelineState state;
};

}  // namespace Saiga
