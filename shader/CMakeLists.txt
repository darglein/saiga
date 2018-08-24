
#if(UNIX)
#SET(SHADER_INSTALL_DIR share/${LIB_NAME}/shader )
#else()
#SET(SHADER_INSTALL_DIR shader )
#endif(UNIX)


#install(
#	DIRECTORY ./
#	DESTINATION ${INSTALL_SHADER}
#	FILES_MATCHING
#		PATTERN "*"
#		PATTERN ".txt" EXCLUDE
#)

install(
        DIRECTORY ./
        DESTINATION ${INSTALL_SHADER}
        FILES_MATCHING PATTERN "*.glsl"
        PATTERN "*.vert"
        PATTERN "*.frag"
        PATTERN "*.comp"
        PATTERN "*.h"
)

