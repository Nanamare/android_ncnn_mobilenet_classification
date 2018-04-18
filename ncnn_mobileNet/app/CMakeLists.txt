cmake_minimum_required(VERSION 3.4.1)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})

include_directories(src/main/cpp/include
                    src/main/cpp/)

file(GLOB MOBILNET_SRC src/main/cpp/*.h
                    src/main/cpp/*.cpp)

add_library(libncnn STATIC IMPORTED)
set_target_properties(libncnn
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libncnn.a)

add_library(objectclassification SHARED ${MOBILNET_SRC})

find_library(log-lib log)
target_link_libraries(objectclassification
                       libncnn
                       jnigraphics
                       z
                       ${log-lib})
