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

	option(enable_asan "Enable ASAN on the build (it'll slow down the app)" OFF)

	set(emcc_simd "fixed" CACHE STRING "SIMD type to use in the build (none, fixed, relaxed)")
	set_property(CACHE emcc_simd PROPERTY STRINGS none fixed relaxed)
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
	# Produce debug symbols and set optimization level
	set(common_cxx_flags
		-g $<$<CONFIG:Release>:-O3>
	)

	set(common_link_flags
		-g
	)

	if (NOT EMSCRIPTEN)
		# EMSCRIPTEN doesn't know about archs
		set(common_cxx_flags
			${common_cxx_flags} -march=native
		)
	else()
		# EMSCRIPTEN flags:
		# -sALLOW_MEMORY_GROWTH (allow runtime allocations) (for some reason cmake passes this as a linker flag when in theory it's a compiler one)
		set(common_link_flags ${common_link_flags} -sALLOW_MEMORY_GROWTH=1)

		# -fsanitize=address (enable ASAN)
		# More at https://emscripten.org/docs/debugging/Sanitizers.html#address-sanitizer 
		if (enable_asan) 
			set(common_cxx_flags ${common_cxx_flags} -fsanitize=address)
			set(common_link_flags ${common_link_flags} -fsanitize=address)
		endif()

		string(TOLOWER "${emcc_simd}" emcc_simd_lower)
		if (emcc_simd_lower STREQUAL fixed)
			set(common_cxx_flags ${common_cxx_flags} -msimd128 -mavx)
		elseif (emcc_simd_lower STREQUAL relaxed)
			set(common_cxx_flags ${common_cxx_flags} -mrelaxed-simd -mavx)
		endif()

	endif()

	target_compile_options(tiny_bvh_test PRIVATE ${common_cxx_flags})
	target_link_options(tiny_bvh_test PRIVATE ${common_link_flags})

	target_compile_options(tiny_bvh_renderer PRIVATE ${common_cxx_flags})
	target_link_options(tiny_bvh_renderer PRIVATE ${common_link_flags})

	if (NOT APPLE AND NOT EMSCRIPTEN)
		target_compile_options(tiny_bvh_speedtest PRIVATE ${common_cxx_flags} -fopenmp)
		target_link_options(tiny_bvh_speedtest PRIVATE ${common_link_flags} -fopenmp)
	else()
		# No openmp support in default compiler
		target_compile_options(tiny_bvh_speedtest PRIVATE ${common_cxx_flags})
		target_link_options(tiny_bvh_speedtest PRIVATE ${common_link_flags})
	endif()

	if (TARGET tiny_bvh_fenster)
		# Not all platforms support all targets. E. g.:
		#	- EMSCRIPTEN doesn't render anything by default (you would need WebGL/WebGPU)
		target_compile_options(tiny_bvh_fenster PRIVATE ${common_cxx_flags})
		target_link_options(tiny_bvh_fenster PRIVATE ${common_link_flags})
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
