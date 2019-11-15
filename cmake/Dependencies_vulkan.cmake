# Defines the following output variables:
#
# VULKAN_INCLUDES:    The list of required include directories
# VULKAN_LIBS:        The list of required libraries for link_target
# VULKAN_TARGETS:     The list of required targets
# MODULE_VULKAN:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_VULKAN)

if(NOT MODULE_CORE)
    return()
endif()

##### Vulkan #####
if(SAIGA_MODULE_VULKAN)
    find_package(Vulkan QUIET)
    PackageHelperTarget(Vulkan::Vulkan VULKAN_FOUND)

    find_package(GLslang QUIET)
    PackageHelper(GLslang "${GLSLANG_FOUND}" "${GLSLANG_SPIRV_INCLUDE_DIR}" "${GLSLANG_LIBRARIES}")
    if(${VULKAN_FOUND} AND ${GLSLANG_FOUND})
        message(STATUS "Saiga vulkan enabled.")
        SET(SAIGA_USE_VULKAN 1)
    endif()
else()
    UNSET(SAIGA_USE_VULKAN)
endif()


set(VULKAN_INCLUDES ${PACKAGE_INCLUDES})
set(VULKAN_LIBS ${LIBS})
set(VULKAN_TARGETS saiga_core ${LIB_TARGETS})

if(SAIGA_USE_VULKAN)
    set(MODULE_VULKAN 1)
endif()
