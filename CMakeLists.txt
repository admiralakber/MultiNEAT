# EZMake MultiNEAT/CMakeLists.txt

cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
project(MultiNEAT VERSION 0.1.0)
set(PROJECT_DESCRIPTION "MultiNEAT")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(ezmake/EZMake)
include(CTest)

option(MultiNEAT_WITH_TESTING "Build tests/examples" ON)
option(MultiNEAT_NO_INSTALL "Skip installation process" OFF)

ez_proj_init()

ez_proj_ext_doxygen()
ez_proj_ext_boost(COMPONENTS date_time system filesystem python3 serialization)

#add_subdirectory(${PROJECT_SOURCE_DIR}/extern)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/bin)

ez_proj_export()
