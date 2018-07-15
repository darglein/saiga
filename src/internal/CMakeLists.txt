# This file collects all source and header files
# The following variables are defined with "PARENT_SCOPE"
#
# SAIGA_INTERNAL_SOURCE
# SAIGA_INTERNAL_HEADERS

############# Core Source Files ###############

FILE(GLOB_RECURSE internal_SRC  ./*.cpp)

SET(SAIGA_INTERNAL_SOURCE 
	${internal_SRC}
	PARENT_SCOPE
	)
	

############# All Header Files ###############

FILE(GLOB_RECURSE PROG_HEADER  ./*.h)
set(SAIGA_INTERNAL_HEADERS ${PROG_HEADER} PARENT_SCOPE)

