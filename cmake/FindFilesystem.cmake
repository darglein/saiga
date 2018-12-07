

include(FindPackageHandleStandardArgs)

if(CMAKE_COMPILER_IS_GNUCXX)

    find_library(
        FILESYSTEM_LIBRARY
        NAMES stdc++fs
        PATH_SUFFIXES
        # linux gcc location
        gcc/x86_64-linux-gnu/8
        )

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(Filesystem DEFAULT_MSG FILESYSTEM_LIBRARY)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")


    find_library(
        FILESYSTEM_LIBRARY
        NAMES stdc++fs
        PATH_SUFFIXES
        # linux gcc location
        gcc/x86_64-linux-gnu/8
        )

    if(NOT FILESYSTEM_LIBRARY)
        find_library(
            FILESYSTEM_LIBRARY
            NAMES c++fs
            HINTS
            # mac clang location
            /usr/local/opt/llvm/lib
            )
    endif()

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(Filesystem DEFAULT_MSG FILESYSTEM_LIBRARY)
else()
    #todo windows check
    set(FILESYSTEM_LIBRARY "")
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(Filesystem DEFAULT_MSG)

endif()




