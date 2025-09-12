#==============================================================
# Project defines:
#==============================================================
#

    set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/data/test)
    set(TRAIN_DATA_DIR ${CMAKE_SOURCE_DIR}/data/train)
    set(VARIABLES_PATH ${CMAKE_SOURCE_DIR}/variables/000.json)

#
#==============================================================


cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(neural_network)


add_executable(AI_test)
target_sources(AI_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/a.cpp
        ${CMAKE_SOURCE_DIR}/src/components/Neuron.cpp
)   
target_include_directories(AI_test 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/inc
        ${CMAKE_SOURCE_DIR}/inc/ext/Eigen
)  


# #==============================================================
# # AI_execute
# #==============================================================

# add_executable(AI_execute)
# target_sources(AI_execute 
#     PRIVATE 
#         ${CMAKE_SOURCE_DIR}/src/execute.cpp
#         ${CMAKE_SOURCE_DIR}/src/components/AI.cpp
#         ${CMAKE_SOURCE_DIR}/src/components/Digit.cpp
#         ${CMAKE_SOURCE_DIR}/src/components/Neuron.cpp

# )
# target_include_directories(AI_execute 
#     PRIVATE 
#         ${CMAKE_SOURCE_DIR}/inc
# )
# target_compile_definitions(AI_execute
#     PRIVATE
#         VARIABLES_PATH="${VARIABLES_PATH}"
# )



# #==============================================================
# # AI_test
# #==============================================================

# add_executable(AI_test)
# target_sources(AI_test 
#     PRIVATE 
#         ${CMAKE_SOURCE_DIR}/src/test.cpp
#         ${CMAKE_SOURCE_DIR}/src/components/AI.cpp
#         ${CMAKE_SOURCE_DIR}/src/components/Digit.cpp
#         ${CMAKE_SOURCE_DIR}/src/components/Neuron.cpp

# )
# target_include_directories(AI_test 
#     PRIVATE 
#         ${CMAKE_SOURCE_DIR}/inc
# )
# target_compile_definitions(AI_test
#     PRIVATE
#         TEST_DATA_DIR="${TEST_DATA_DIR}"
#         VARIABLES_PATH="${VARIABLES_PATH}"
# )
