
find_package(GTest QUIET)


if(GTEST_FOUND)
    if(MODULE_CORE)
        add_subdirectory(core)
    endif()
    if(MODULE_VISION)
        add_subdirectory(vision)
    endif()
else()
    message(STATUS "GTest not found. Tests disabled.")
endif()





