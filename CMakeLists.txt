# CMakeList.txt : CMake project for sineApprox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("sineApprox")

set(SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

message(${SOURCE_DIR})

file(GLOB_RECURSE sources "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.hpp")

add_subdirectory(matplotplusplus)

# Add source to this project's executable.
add_executable (sineApprox ${sources})
target_link_libraries(sineApprox PUBLIC matplot)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET sineApprox PROPERTY CXX_STANDARD 20)

  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()
