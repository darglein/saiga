include(saiga_sample_macros)
list(APPEND required_modules "saiga_core" ${Gphoto2_LIBRARIES})
saiga_make_sample(required_modules)
