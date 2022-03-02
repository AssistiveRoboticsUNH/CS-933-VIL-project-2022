project(NSF)
cmake_minimum_required(VERSION 3.16.3)


SET(MATLAB_ROOT ${MATLAB_ROOT} CACHE FILEPATH "directory")


set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_definitions(/DMATLAB_MEX_FILE)
add_definitions(/DMX_COMPAT_32)

find_package(Matlab REQUIRED)

IF(MATLAB_FOUND)
    message(STATUS "MATLAB Found, MATLAB MEX will be compiled.")
    add_subdirectory(processTextMex)

ELSE(MATLAB_FOUND)
    message( FATAL_ERROR "MATLAB not found...nothing will be built." )
ENDIF(MATLAB_FOUND)
