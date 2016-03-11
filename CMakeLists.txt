PROJECT(CoMeDiC)

#Required in DGtal
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(DGtal REQUIRED)
INCLUDE_DIRECTORIES(${DGTAL_INCLUDE_DIRS})
LINK_DIRECTORIES(${DGTAL_LIBRARY_DIRS})

SET(Boost_USE_STATIC_LIBS   OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(Boost_FOUND FALSE)
FIND_PACKAGE(Boost 1.46.0 REQUIRED COMPONENTS program_options)
if ( Boost_FOUND )
  message(STATUS "Boost and boost_program_options found.")
  include_directories( ${Boost_INCLUDE_DIRS} )
  SET(CurvatureLibDependencies ${CurvatureLibDependencies}
     ${Boost_LIBRAIRIES}
     ${Boost_PROGRAM_OPTIONS_LIBRARY})
   SET(DGtalLibInc ${Boost_INCLUDE_DIRS})
endif( Boost_FOUND )
include_directories( /usr/local/include/eigen3 )
add_definitions( -DWITH_EIGEN )

ADD_EXECUTABLE(at at)
ADD_EXECUTABLE(at-cremers at-cremers)
ADD_EXECUTABLE(at-TM-0p-0p-matrix at-TM-0p-0p-matrix)
ADD_EXECUTABLE(at-TM-0p-1p-matrix at-TM-0p-1p-matrix)
ADD_EXECUTABLE(3d-at-normals-u0-v1 3d-at-normals-u0-v1)
ADD_EXECUTABLE(3d-at-normals-u2-v1 3d-at-normals-u2-v1)

# TARGET_LINK_LIBRARIES(grayball ${DGTAL_LIBRARIES})
target_link_libraries (at ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
target_link_libraries (at-cremers ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
target_link_libraries (at-TM-0p-0p-matrix ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
target_link_libraries (at-TM-0p-1p-matrix ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
target_link_libraries (3d-at-normals-u0-v1 ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
target_link_libraries (3d-at-normals-u2-v1 ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})

