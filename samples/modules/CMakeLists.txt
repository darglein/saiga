
if (EIGEN3_FOUND)
    add_subdirectory(eigen)
endif ()

if (OPENNI2_FOUND AND Boost_FOUND)
    add_subdirectory(openni2Client)
    add_subdirectory(openni2Server)
endif ()

if (GPHOTO2_FOUND)
    add_subdirectory(gphoto)
endif ()


