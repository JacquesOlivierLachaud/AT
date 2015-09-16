PROJECT(CoMeDiC)

#Required in DGtal
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(DGtal REQUIRED)
INCLUDE_DIRECTORIES(${DGTAL_INCLUDE_DIRS})
LINK_DIRECTORIES(${DGTAL_LIBRARY_DIRS})

SET(Boost_USE_STATIC_LIBS   ON)
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

ADD_EXECUTABLE(at at)
ADD_EXECUTABLE(at-cremers at-cremers)

# TARGET_LINK_LIBRARIES(grayball ${DGTAL_LIBRARIES})
target_link_libraries (at ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
target_link_libraries (at-cremers ${DGTAL_LIBRARIES} ${CurvatureLibDependencies})
