# The game engine module saiga_engine (docs/engine_design.md §5.3).
#
# Defines the following output variables:
#
# ENGINE_INCLUDES:  The list of required include directories
# ENGINE_LIBS:      The list of required libraries for link_target
# ENGINE_TARGETS:   The list of required targets
# MODULE_ENGINE:    True if all required dependencies are found.
#
# The engine needs the vulkan module. Jolt is not linked yet; from stage S28 the engine requires it (MODULE_ENGINE only
# if JOLT_FOUND, SAIGA_WITH_JOLT defaults to ON with the engine).

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_ENGINE)
unset(SAIGA_USE_ENGINE)

if(NOT MODULE_CORE)
    return()
endif()

if(MODULE_VULKAN)
    message(STATUS "Saiga engine enabled.")
    set(SAIGA_USE_ENGINE 1)
endif()

set(ENGINE_INCLUDES ${PACKAGE_INCLUDES})
set(ENGINE_LIBS ${LIBS})
set(ENGINE_TARGETS saiga_vulkan ${LIB_TARGETS})

if(SAIGA_USE_ENGINE)
    set(MODULE_ENGINE 1)
endif()
