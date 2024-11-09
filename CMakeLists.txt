cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(tiny_bvh LANGUAGES CXX)

if (APPLE)
	find_library(COCOA_LIBRARY Cocoa)
elseif (UNIX)
	find_package(X11)
endif()

enable_testing()

add_executable(tiny_bvh_test tiny_bvh_test.cpp)
add_executable(tiny_bvh_renderer tiny_bvh_renderer.cpp)
add_executable(tiny_bvh_speedtest tiny_bvh_speedtest.cpp)
add_executable(tiny_bvh_fenster tiny_bvh_fenster.cpp)

if (NOT MSVC)
	set(common_cxx_flags
		-g -march=native
	)
	target_compile_options(tiny_bvh_test PRIVATE ${common_cxx_flags})
	target_compile_options(tiny_bvh_renderer PRIVATE ${common_cxx_flags})
	if (NOT APPLE)
		target_compile_options(tiny_bvh_speedtest PRIVATE ${common_cxx_flags} -fopenmp)
		target_link_options(tiny_bvh_speedtest PRIVATE -fopenmp)
	else()
		# No openmp support in default compiler
		target_compile_options(tiny_bvh_speedtest PRIVATE ${common_cxx_flags})
		target_link_options(tiny_bvh_speedtest PRIVATE)
	endif()
	target_compile_options(tiny_bvh_fenster PRIVATE ${common_cxx_flags})
	if (WIN32)
		target_link_libraries(tiny_bvh_fenster -mwindows)
	elseif (APPLE)
		target_link_libraries(tiny_bvh_fenster ${COCOA_LIBRARY})
	elseif (X11_FOUND)
		target_link_libraries(tiny_bvh_fenster ${X11_LIBRARIES})
	endif()
else()
	if (WIN32)
		set_target_properties(tiny_bvh_fenster PROPERTIES WIN32_EXECUTABLE TRUE)
	endif()
endif()

add_test(NAME "tiny_bvh_test" COMMAND tiny_bvh_test)
