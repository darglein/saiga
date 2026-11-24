/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The content key of pipeline reuse (VulkanBase::reusePipelines): the fields of a create info written one by one (no
// struct bytes: padding), the shader modules by their SPIR-V. A key is only built for what is written here: an
// unknown pNext or a specialization makes it invalid (no reuse; the pipeline is created as usual).

#pragma once

#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/svulkan.h"

#include <cstdint>
#include <string>

#include <type_traits>

namespace Saiga
{
namespace Vulkan
{
struct PipelineKeyWriter
{
    std::string bytes;
    bool valid = true;

    template <typename T>
    void raw(const T& v)
    {
        static_assert(std::is_arithmetic_v<T>, "raw: numbers only (no padding)");
        bytes.append(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    template <typename T>
    void e(T v)  // an enum
    {
        raw(int64_t(v));
    }
    template <typename B>
    void f(vk::Flags<B> v)
    {
        raw(uint64_t(typename vk::Flags<B>::MaskType(v)));
    }
    void b(vk::Bool32 v) { raw(uint32_t(v)); }
    void str(const char* s)
    {
        const std::string t = s ? s : "";
        raw(uint64_t(t.size()));
        bytes += t;
    }
    // A pNext that must be null (an extension the writer does not know: no key)
    void none(const void* next)
    {
        if (next) valid = false;
    }
    void stage(const vk::PipelineShaderStageCreateInfo& s, const ShaderModule* m)
    {
        none(s.pNext);
        f(s.flags);
        e(s.stage);
        str(s.pName);
        if (s.pSpecializationInfo) valid = false;
        if (!m || m->spirvSize == 0)
        {
            valid = false;
            return;
        }
        raw(m->spirvSize);
        raw(m->spirvHashA);
        raw(m->spirvHashB);
    }
    void stencil(const vk::StencilOpState& s)
    {
        e(s.failOp);
        e(s.passOp);
        e(s.depthFailOp);
        e(s.compareOp);
        raw(s.compareMask);
        raw(s.writeMask);
        raw(s.reference);
    }
};

}  // namespace Vulkan
}  // namespace Saiga
