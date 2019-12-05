
find_package(GTest QUIET)


if(GTEST_FOUND)
    if(MODULE_CORE)
        add_subdirectory(core)
    endif()
else()
    message(STATUS "GTest not found. Tests disabled.")
endif()





