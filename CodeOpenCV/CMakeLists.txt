cmake_minimum_required(VERSION 3.5)
project( AT )
set (CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET( MAIN_EXEC
     at-gradient-superres
     at-gradient )

FOREACH(FILE ${MAIN_EXEC})
            add_executable( ${FILE} ${FILE}.cpp )
            target_link_libraries( ${FILE} ${OpenCV_LIBS} )
ENDFOREACH(FILE)
#add_executable( clustering clustering.cpp )
#target_link_libraries( clustering ${OpenCV_LIBS} )
#add_executable( spatial_clustering spatial_clustering.cpp )
#target_link_libraries( spatial_clustering ${OpenCV_LIBS} )
