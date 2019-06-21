#enable_testing()
find_package(GTest QUIET)


if(GTEST_FOUND)
if(MODULE_CORE)
    add_subdirectory(core)
endif()
endif()





