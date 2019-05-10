
if (EIGEN3_FOUND)
    add_subdirectory(eigen)
    add_subdirectory(stress_test)
endif ()



if (GPHOTO2_FOUND)
    add_subdirectory(gphoto)
endif ()


