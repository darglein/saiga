include(saiga_sample_macros)
list(APPEND required_modules "saiga_vision")

saiga_make_benchmark_sample()
saiga_make_sample(required_modules)
