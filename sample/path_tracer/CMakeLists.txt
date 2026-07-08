add_executable(nrc_reference_path_tracer)

target_sources(nrc_reference_path_tracer PRIVATE
    placeholder.cpp)

target_link_libraries(nrc_reference_path_tracer PRIVATE
    nrc_core
    nrc_shaders)
