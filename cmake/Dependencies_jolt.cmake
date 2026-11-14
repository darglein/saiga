# Jolt Physics (https://github.com/jrouwe/JoltPhysics), downloaded and built with FetchContent when SAIGA_WITH_JOLT is
# ON. It is not a submodule, so projects that do not need physics do not clone it. None of the saiga libraries link it;
# the target `Jolt` is available to the samples and to projects that add saiga with add_subdirectory.
#
# Defines the following output variables:
#
# JOLT_FOUND:  True if the target Jolt exists.
#
# The compile definitions of Jolt (JPH_CROSS_PLATFORM_DETERMINISTIC, JPH_DOUBLE_PRECISION, JPH_DEBUG_RENDERER, ...) are
# PUBLIC on the target. Code that includes Jolt headers must therefore link the target, otherwise it is compiled with
# different definitions than the library.

unset(JOLT_FOUND)

if (NOT SAIGA_WITH_JOLT)
    return()
endif ()

include(FetchContent)

# Jolt's own options are cache variables, so they can be overridden on the command line (e.g.
# -DCROSS_PLATFORM_DETERMINISTIC=ON for lockstep / deterministic networking). The defaults below differ from Jolt's
# defaults so that it builds inside saiga.

# Static library (saiga sets BUILD_SHARED_LIBS, which Jolt would follow); position independent, see below.
set(JPH_BUILD_SHARED_LIBS OFF CACHE BOOL "Compile Jolt as a shared library")
# Jolt's default turns on -Wall -Werror.
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "Enable all warnings and warnings as errors")
# Saiga uses the dynamic MSVC runtime; mixing both fails to link.
set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "Use the static MSVC runtime library")
# LTO objects in a static library only link with the same compiler and an LTO-aware linker.
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "Enable interprocedural optimizations")
# Do not add Jolt to saiga's install.
set(ENABLE_INSTALL OFF CACHE BOOL "Generate installation target")
# The GPU compute backends are only used by Jolt's hair simulation and need the dxc shader compiler.
set(JPH_USE_VK OFF CACHE BOOL "Use Vulkan")
set(JPH_USE_DX12 OFF CACHE BOOL "Use DirectX")
set(JPH_USE_MTL OFF CACHE BOOL "Use Metal")
set(JPH_USE_CPU_COMPUTE OFF CACHE BOOL "Use CPU Compute")

FetchContent_Declare(
        JoltPhysics
        GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
        GIT_TAG e77f175595e64cb44218cc9d9d56fc365ad0e36a  # v5.6.0
        GIT_SHALLOW FALSE
        SOURCE_SUBDIR Build
)
FetchContent_MakeAvailable(JoltPhysics)

if (TARGET Jolt)
    # Allows linking the static library into shared libraries (e.g. a game's own physics module).
    set_target_properties(Jolt PROPERTIES POSITION_INDEPENDENT_CODE ON FOLDER external)
    set(JOLT_FOUND 1)
    message(STATUS "Package Jolt                       Yes, FetchContent v5.6.0")
else ()
    message(WARNING "SAIGA_WITH_JOLT is ON, but the Jolt target was not created.")
endif ()
