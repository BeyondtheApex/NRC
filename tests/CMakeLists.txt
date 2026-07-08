add_executable(nrc_profile_layout_test
    profile_layout_test.cpp)

target_link_libraries(nrc_profile_layout_test PRIVATE nrc_core)

add_test(NAME nrc_profile_layout_test COMMAND nrc_profile_layout_test)
