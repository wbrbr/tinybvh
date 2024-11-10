cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(tiny_bvh LANGUAGES CXX)

if (APPLE)
	find_library(COCOA_LIBRARY Cocoa)
elseif (UNIX AND NOT EMSCRIPTEN)
	find_package(X11)
elseif (EMSCRIPTEN)
	# To auto generate test pages
	set(CMAKE_EXECUTABLE_SUFFIX ".html")
endif()

enable_testing()

add_executable(tiny_bvh_test tiny_bvh_test.cpp)
add_executable(tiny_bvh_renderer tiny_bvh_renderer.cpp)
add_executable(tiny_bvh_speedtest tiny_bvh_speedtest.cpp)
if (NOT EMSCRIPTEN)
	# EMSCRIPTEN doesn't render anything by default (you would need WebGL/WebGPU)
	add_executable(tiny_bvh_fenster tiny_bvh_fenster.cpp)
endif()

if (NOT MSVC)
	# Produce debug symbols
	set(common_cxx_flags
		-g
	)

	if (NOT EMSCRIPTEN)
		# EMSCRIPTEN doesn't know about archs
		set(common_cxx_flags
			${common_cxx_flags} -march=native
		)
	else()
		# EMSCRIPTEN flags:
		#	- -sALLOW_MEMORY_GROWTH (allow runtime allocations) (note linker will warn as CMAKE passes CMAKE_CXX_FLAGS to it as well)
		set(CMAKE_CXX_FLAGS 
			${CMAKE_CXX_FLAGS} -sALLOW_MEMORY_GROWTH=1
		)
	endif()

	target_compile_options(tiny_bvh_test PRIVATE ${common_cxx_flags})
	target_compile_options(tiny_bvh_renderer PRIVATE ${common_cxx_flags})

	if (NOT APPLE AND NOT EMSCRIPTEN)
		target_compile_options(tiny_bvh_speedtest PRIVATE ${common_cxx_flags} -fopenmp)
		target_link_options(tiny_bvh_speedtest PRIVATE -fopenmp)
	else()
		# No openmp support in default compiler
		target_compile_options(tiny_bvh_speedtest PRIVATE ${common_cxx_flags})
		target_link_options(tiny_bvh_speedtest PRIVATE)
	endif()

	if (TARGET tiny_bvh_fenster)
		# Not all platforms support all targets. E. g.:
		#	- EMSCRIPTEN doesn't render anything by default (you would need WebGL/WebGPU)
		target_compile_options(tiny_bvh_fenster PRIVATE ${common_cxx_flags})
		if (WIN32)
			target_link_libraries(tiny_bvh_fenster -mwindows)
		elseif (APPLE)
			target_link_libraries(tiny_bvh_fenster ${COCOA_LIBRARY})
		elseif (X11_FOUND)
			target_link_libraries(tiny_bvh_fenster ${X11_LIBRARIES})
		endif()
	endif()
else()
	if (WIN32)
		set_target_properties(tiny_bvh_fenster PROPERTIES WIN32_EXECUTABLE TRUE)
	endif()
endif()

add_test(NAME "tiny_bvh_test" COMMAND tiny_bvh_test)
