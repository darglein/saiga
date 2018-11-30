//
// Created by Peter Eichinger on 2018-11-30.
//

#pragma once

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
template <typename T>
struct MemoryType
{
    T usageFlags;
    vk::MemoryPropertyFlags memoryFlags;

    bool operator==(const MemoryType<T>& rhs) const
    {
        return std::tie(usageFlags, memoryFlags) == std::tie(rhs.usageFlags, rhs.memoryFlags);
    }

    bool operator!=(const MemoryType<T>& rhs) const { return !(rhs == *this); }

    inline bool valid(const MemoryType<T>& other) const
    {
        return ((usageFlags & other.usageFlags) == other.usageFlags) &&
               ((memoryFlags & other.memoryFlags) == other.memoryFlags);
    }
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga

namespace std
{
template <typename T>
struct hash<Saiga::Vulkan::Memory::MemoryType<T>>
{
    typedef Saiga::Vulkan::Memory::MemoryType<T> argument_type;
    typedef std::size_t result_type;
    result_type operator()(argument_type const& s) const noexcept
    {
        result_type const h1(std::hash<unsigned int>{}(static_cast<unsigned int>(s.usageFlags)));
        result_type const h2(std::hash<unsigned int>{}(static_cast<unsigned int>(s.memoryFlags)));
        return h1 ^ (h2 << 1);
    }
};
}  // namespace std