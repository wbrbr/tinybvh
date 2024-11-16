#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

// 'screen resolution': see tiny_bvh_fenster.cpp; this program traces the
// same rays, but without visualization - just performance statistics.
#define SCRWIDTH	800
#define SCRHEIGHT	600

// scene selection
#define LOADSPONZA

// GPU ray tracing
#define ENABLE_OPENCL

// tests to perform
#define BUILD_REFERENCE
#define BUILD_AVX
#define BUILD_NEON
// #define BUILD_SBVH
#define TRAVERSE_2WAY_ST
#define TRAVERSE_ALT2WAY_ST
#define TRAVERSE_SOA2WAY_ST
#define TRAVERSE_2WAY_MT
#define TRAVERSE_2WAY_MT_PACKET
#define TRAVERSE_2WAY_MT_DIVERGENT
#define TRAVERSE_OPTIMIZED_ST
// #define EMBREE_BUILD // win64-only for now.
// #define EMBREE_TRAVERSE // win64-only for now.

// GPU rays: only if ENABLE_OPENCL is defined.
#define GPU_2WAY
#define GPU_4WAY
#define GPU_CWBVH

using namespace tinybvh;

#ifdef _MSC_VER
#include "stdio.h"		// for printf
#include "stdlib.h"		// for rand
#else
#include <cstdio>
#endif
#ifdef _WIN32
#include <intrin.h>		// for __cpuidex
#elif defined ENABLE_OPENCL
#undef ENABLE_OPENCL
#endif
#if defined(__GNUC__) && defined(__x86_64__)
#include <cpuid.h>
#endif
#ifdef __EMSCRIPTEN__ 
#include <emscripten/version.h> // for __EMSCRIPTEN_major__, __EMSCRIPTEN_minor__
#endif

#ifdef LOADSPONZA
bvhvec4* triangles = 0;
#include <fstream>
#else
ALIGNED( 16 ) bvhvec4 triangles[259 /* level 3 */ * 6 * 2 * 49 * 3]{};
#endif
int verts = 0;
BVH bvh;

#if defined EMBREE_BUILD || defined EMBREE_TRAVERSE
#include "embree4/rtcore.h"
static RTCScene embreeScene;
void embreeError( void* userPtr, enum RTCError error, const char* str )
{
	printf( "error %d: %s\n", error, str );
}
#endif

#ifdef ENABLE_OPENCL
#define TINY_OCL_IMPLEMENTATION
#include "cl.h"
#include "tiny_ocl.h"
#endif

float uniform_rand() { return (float)rand() / (float)RAND_MAX; }

#include <chrono>
struct Timer
{
	Timer() { reset(); }
	float elapsed() const
	{
		auto t2 = std::chrono::high_resolution_clock::now();
		return (float)std::chrono::duration_cast<std::chrono::duration<double>>(t2 - start).count();
	}
	void reset() { start = std::chrono::high_resolution_clock::now(); }
	std::chrono::high_resolution_clock::time_point start;
};

void sphere_flake( float x, float y, float z, float s, int d = 0 )
{
	// procedural tesselated sphere flake object
#define P(F,a,b,c) p[i+F*64]={(float)a ,(float)b,(float)c}
	bvhvec3 p[384], pos( x, y, z ), ofs( 3.5 );
	for (int i = 0, u = 0; u < 8; u++) for (int v = 0; v < 8; v++, i++)
		P( 0, u, v, 0 ), P( 1, u, 0, v ), P( 2, 0, u, v ),
		P( 3, u, v, 7 ), P( 4, u, 7, v ), P( 5, 7, u, v );
	for (int i = 0; i < 384; i++) p[i] = normalize( p[i] - ofs ) * s + pos;
	for (int i = 0, side = 0; side < 6; side++, i += 8)
		for (int u = 0; u < 7; u++, i++) for (int v = 0; v < 7; v++, i++)
			triangles[verts++] = p[i], triangles[verts++] = p[i + 8],
			triangles[verts++] = p[i + 1], triangles[verts++] = p[i + 1],
			triangles[verts++] = p[i + 9], triangles[verts++] = p[i + 8];
	if (d < 3) sphere_flake( x + s * 1.55f, y, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x - s * 1.5f, y, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, y + s * 1.5f, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, x - s * 1.5f, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, y, z + s * 1.5f, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, y, z - s * 1.5f, s * 0.5f, d + 1 );
}

int main()
{
	int minor = TINY_BVH_VERSION_MINOR;
	int major = TINY_BVH_VERSION_MAJOR;
	int sub = TINY_BVH_VERSION_SUB;
	printf( "tiny_bvh version %i.%i.%i performance statistics ", major, minor, sub );

	// determine compiler
#ifdef _MSC_VER
	printf( "(MSVC %i build)\n", _MSC_VER );
#elif defined __EMSCRIPTEN__
	// EMSCRIPTEN needs to be before clang or gcc
	printf( "(emcc %i.%i build)\n", __EMSCRIPTEN_major__, __EMSCRIPTEN_minor__ );
#elif defined __clang__
	printf( "(clang %i.%i build)\n", __clang_major__, __clang_minor__ );
#elif defined __GNUC__
	printf( "(gcc %i.%i build)\n", __GNUC__, __GNUC_MINOR__ );
#else
	printf( "\n" );
#endif

	// determine what CPU is running the tests.
#if (defined(__x86_64__) || defined(_M_X64)) && (defined (_WIN32) || defined(__GNUC__))
	char model[64]{};
	for (unsigned i = 0; i < 3; ++i)
	{
	#ifdef _WIN32
		__cpuidex( (int*)(model + i * 16), i + 0x80000002, 0 );
	#elif defined(__GNUC__)
		__get_cpuid( i + 0x80000002,
			(unsigned*)model + i * 4 + 0, (unsigned*)model + i * 4 + 1,
			(unsigned*)model + i * 4 + 2, (unsigned*)model + i * 4 + 3 );
	#endif
	}
	printf( "running on %s\n", model );
#endif
	printf( "----------------------------------------------------------------\n" );

#ifdef LOADSPONZA
	// load raw vertex data for Crytek's Sponza
	std::string filename{ "../testdata/cryteksponza.bin" };
	std::fstream s{ filename, s.binary | s.in };
	if (!s.is_open())
	{
		// try again, look in .\testdata
		filename = std::string{ "./testdata/cryteksponza.bin" };
		s = std::fstream{ filename, s.binary | s.in };
		assert( s.is_open() );
	}
	s.seekp( 0 );
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)ALIGNED_MALLOC( verts * 16 );
	s.read( (char*)triangles, verts * 16 );
#else
	// generate a sphere flake scene
	printf( "Creating sphere flake (%i tris).\n", verts / 3 );
	sphere_flake( 0, 0, 0, 1.5f );
#endif

	// setup view pyramid for a pinhole camera: 
	// eye, p1 (top-left), p2 (top-right) and p3 (bottom-left)
#ifdef LOADSPONZA
	bvhvec3 eye( 0, 30, 0 ), view = normalize( bvhvec3( -8, 2, -1.7f ) );
#else
	bvhvec3 eye( -3.5f, -1.5f, -6.5f ), view = normalize( bvhvec3( 3, 1.5f, 5 ) );
#endif
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) );
	bvhvec3 up = 0.8f * cross( view, right ), C = eye + 2 * view;
	bvhvec3 p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;

	// generate primary rays in a cacheline-aligned buffer - and, for data locality:
	// organized in 4x4 pixel tiles, 16 samples per pixel, so 256 rays per tile.
	int N = 0;
	Ray* rays = (Ray*)ALIGNED_MALLOC( SCRWIDTH * SCRHEIGHT * 16 * sizeof( Ray ) );
	for (int ty = 0; ty < SCRHEIGHT / 4; ty++) for (int tx = 0; tx < SCRWIDTH / 4; tx++)
	{
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
		{
			int pixel_x = tx * 4 + x;
			int pixel_y = ty * 4 + y;
			for (int s = 0; s < 16; s++) // 16 samples per pixel
			{
				float u = (float)(pixel_x * 4 + (s & 3)) / (SCRWIDTH * 4);
				float v = (float)(pixel_y * 4 + (s >> 2)) / (SCRHEIGHT * 4);
				bvhvec3 P = p1 + u * (p2 - p1) + v * (p3 - p1);
				rays[N++] = Ray( eye, normalize( P - eye ) );
			}
		}
	}

	//  T I N Y _ B V H   P E R F O R M A N C E   M E A S U R E M E N T S

	Timer t;
	float mrays;

	// measure single-core bvh construction time - warming caches
	printf( "BVH construction speed\n" );
	printf( "warming caches...\n" );
	bvh.Build( triangles, verts / 3 );

#ifdef BUILD_REFERENCE

	// measure single-core bvh construction time - reference builder
	printf( "- reference builder: " );
	t.reset();
	for (int pass = 0; pass < 3; pass++) bvh.Build( triangles, verts / 3 );
	float buildTime = t.elapsed() / 3.0f;
	printf( "%7.2fms for %7i triangles ", buildTime * 1000.0f, verts / 3 );
	printf( "- %6i nodes, SAH=%.2f\n", bvh.usedBVHNodes, bvh.SAHCost() );

#endif

#ifdef BUILD_AVX
#ifdef BVH_USEAVX

	// measure single-core bvh construction time - AVX builder
	printf( "- fast AVX builder:  " );
	t.reset();
	for (int pass = 0; pass < 3; pass++) bvh.BuildAVX( triangles, verts / 3 );
	float buildTimeAVX = t.elapsed() / 3.0f;
	printf( "%7.2fms for %7i triangles ", buildTimeAVX * 1000.0f, verts / 3 );
	printf( "- %6i nodes, SAH=%.2f\n", bvh.usedBVHNodes, bvh.SAHCost() );

#endif
#endif

#ifdef BUILD_NEON
#ifdef BVH_USENEON

	// measure single-core bvh construction time - NEON builder
	printf( "- fast NEON builder: " );
	t.reset();
	for (int pass = 0; pass < 3; pass++) bvh.BuildNEON( triangles, verts / 3 );
	float buildTimeNEON = t.elapsed() / 3.0f;
	printf( "%7.2fms for %7i triangles ", buildTimeNEON * 1000.0f, verts / 3 );
	printf( "- %6i nodes, SAH=%.2f\n", bvh.usedBVHNodes, bvh.SAHCost() );

#endif
#endif

#ifdef BUILD_SBVH

	// measure single-core bvh construction time - AVX builder
	printf( "- HQ (SBVH) builder: " );
	t.reset();
	for (int pass = 0; pass < 3; pass++) bvh.BuildHQ( triangles, verts / 3 );
	float buildTimeHQ = t.elapsed() / 3.0f;
	printf( "%7.2fms for %7i triangles ", buildTimeHQ * 1000.0f, verts / 3 );
	printf( "- %6i nodes, SAH=%.2f\n", bvh.usedBVHNodes, bvh.SAHCost() );

#endif

#if defined EMBREE_BUILD || defined EMBREE_TRAVERSE

	// convert data to correct format for Embree and build a BVH
	printf( "- Embree builder:    " );
	RTCDevice embreeDevice = rtcNewDevice( NULL );
	rtcSetDeviceErrorFunction( embreeDevice, embreeError, NULL );
	embreeScene = rtcNewScene( embreeDevice );
	RTCGeometry embreeGeom = rtcNewGeometry( embreeDevice, RTC_GEOMETRY_TYPE_TRIANGLE );
	float* vertices = (float*)rtcSetNewGeometryBuffer( embreeGeom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof( float ), verts );
	unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer( embreeGeom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof( unsigned ), verts / 3 );
	for (int i = 0; i < verts; i++)
	{
		vertices[i * 3 + 0] = triangles[i].x, vertices[i * 3 + 1] = triangles[i].y;
		vertices[i * 3 + 2] = triangles[i].z, indices[i] = i; // Note: not using shared vertices.
	}
	rtcSetGeometryBuildQuality( embreeGeom, RTC_BUILD_QUALITY_MEDIUM ); // no spatial splits
	rtcCommitGeometry( embreeGeom );
	rtcAttachGeometry( embreeScene, embreeGeom );
	rtcReleaseGeometry( embreeGeom );
	rtcSetSceneBuildQuality( embreeScene, RTC_BUILD_QUALITY_MEDIUM );
	t.reset();
	rtcCommitScene( embreeScene ); // assuming this is where (supposedly threaded) BVH build happens.
	float embreeBuildTime = t.elapsed();
	printf( "%7.2fms for %7i triangles\n", embreeBuildTime * 1000.0f, verts / 3 );

#endif

	// trace all rays once to warm the caches
	printf( "BVH traversal speed\n" );

#ifdef TRAVERSE_2WAY_ST

	// trace all rays three times to estimate average performance
	// - single core version
	printf( "- CPU, coherent,   basic 2-way layout, ST: " );
	t.reset();
	for (int pass = 0; pass < 3; pass++)
		for (int i = 0; i < N; i++) bvh.Intersect( rays[i] );
	float traceTimeST = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeST;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeST * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

#ifdef TRAVERSE_ALT2WAY_ST

	// trace all rays three times to estimate average performance
	// - single core version, alternative bvh layout
	printf( "- CPU, coherent,   alt 2-way layout,   ST: " );
	bvh.Convert( BVH::WALD_32BYTE, BVH::AILA_LAINE );
	t.reset();
	for (int pass = 0; pass < 3; pass++)
		for (int i = 0; i < N; i++) bvh.Intersect( rays[i], BVH::AILA_LAINE );
	float traceTimeAlt = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeAlt;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeAlt * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

#ifdef TRAVERSE_SOA2WAY_ST

	// trace all rays three times to estimate average performance
	// - single core version, alternative bvh layout 2
	printf( "- CPU, coherent,   soa 2-way layout,   ST: " );
	bvh.Convert( BVH::WALD_32BYTE, BVH::ALT_SOA );
	t.reset();
	for (int pass = 0; pass < 3; pass++)
		for (int i = 0; i < N; i++) bvh.Intersect( rays[i], BVH::ALT_SOA );
	float traceTimeAlt2 = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeAlt2;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeAlt2 * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

#ifdef TRAVERSE_2WAY_MT

	// trace all rays three times to estimate average performance
	// - multi-core version (using OpenMP and batches of 10,000 rays)
	printf( "- CPU, coherent,   basic 2-way layout, MT: " );
	t.reset();
	for (int j = 0; j < 3; j++)
	{
		const int batchCount = N / 10000;
	#pragma omp parallel for schedule(dynamic)
		for (int batch = 0; batch < batchCount; batch++)
		{
			const int batchStart = batch * 10000;
			for (int i = 0; i < 10000; i++) bvh.Intersect( rays[batchStart + i] );
		}
	}
	float traceTimeMT = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeMT;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeMT * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

#ifdef TRAVERSE_2WAY_MT_PACKET

	// trace all rays three times to estimate average performance
	// - coherent distribution, multi-core, packet traversal
	printf( "- CPU, coherent,   2-way, packets,     MT: " );
	t.reset();
	for (int j = 0; j < 3; j++)
	{
		const int batchCount = N / (30 * 256); // batches of 30 packets of 256 rays
	#pragma omp parallel for schedule(dynamic)
		for (int batch = 0; batch < batchCount; batch++)
		{
			const int batchStart = batch * 30 * 256;
			for (int i = 0; i < 30; i++) bvh.Intersect256Rays( rays + batchStart + i * 256 );
		}
	}
	float traceTimeMTP = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeMTP;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeMTP * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#ifdef BVH_USEAVX

	// trace all rays three times to estimate average performance
	// - coherent distribution, multi-core, packet traversal, SSE version
	printf( "- CPU, coherent,   2-way, packets/SSE, MT: " );
	t.reset();
	for (int j = 0; j < 3; j++)
	{
		const int batchCount = N / (30 * 256); // batches of 30 packets of 256 rays
	#pragma omp parallel for schedule(dynamic)
		for (int batch = 0; batch < batchCount; batch++)
		{
			const int batchStart = batch * 30 * 256;
			for (int i = 0; i < 30; i++) bvh.Intersect256RaysSSE( rays + batchStart + i * 256 );
		}
	}
	float traceTimeMTPS = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeMTPS;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeMTPS * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

#endif

#ifdef TRAVERSE_OPTIMIZED_ST

	// trace all rays three times to estimate average performance
	// - single core version, alternative bvh layout
	printf( "Optimizing BVH, regular...   " );
	t.reset();
	if (!bvh.refittable)
	{
		bvh.Optimize( 1000000 ); // optimize the raw SBVH
	}
	else
	{
		bvh.buildFlag = BVH::FULLSPLIT;
		bvh.Build( triangles, verts / 3 ); // rebuild with full splitting.
		bvh.Optimize( 1000000 );
		bvh.MergeLeafs();
		printf( "done (%.2fs). New: %i nodes, SAH=%.2f\n", t.elapsed(), bvh.NodeCount( BVH::WALD_32BYTE ), bvh.SAHCost() );
		bvh.Convert( BVH::WALD_32BYTE, BVH::ALT_SOA );
		for (int i = 0; i < N; i += 2) bvh.Intersect( rays[i], BVH::ALT_SOA ); // re-warm
		printf( "- CPU, coherent,   2-way optimized,    ST: " );
		t.reset();
		for (int pass = 0; pass < 3; pass++)
			for (int i = 0; i < N; i++) bvh.Intersect( rays[i], BVH::ALT_SOA );
		float traceTimeOpt = t.elapsed() / 3.0f;
		mrays = (float)N / traceTimeOpt;
		printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeOpt * 1000, (float)N * 1e-6f, mrays * 1e-6f );
	}

#endif

#if defined EMBREE_TRAVERSE && defined EMBREE_BUILD

	// trace all rays three times to estimate average performance
	// - coherent, Embree, single-threaded
	printf( "- CPU, coherent,   Embree BVH,  Embree ST: " );
	struct RTCRayHit* rayhits = (RTCRayHit*)ALIGNED_MALLOC( SCRWIDTH * SCRHEIGHT * 16 * sizeof( RTCRayHit ) );
	// copy our rays to Embree format
	for (int i = 0; i < N; i++)
	{
		rayhits[i].ray.org_x = rays[i].O.x, rayhits[i].ray.org_y = rays[i].O.y, rayhits[i].ray.org_z = rays[i].O.z;
		rayhits[i].ray.dir_x = rays[i].D.x, rayhits[i].ray.dir_y = rays[i].D.y, rayhits[i].ray.dir_z = rays[i].D.z;
		rayhits[i].ray.tnear = 0, rayhits[i].ray.tfar = rays[i].hit.t;
		rayhits[i].ray.mask = -1, rayhits[i].ray.flags = 0;
		rayhits[i].hit.geomID = RTC_INVALID_GEOMETRY_ID;
		rayhits[i].hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	}
	t.reset();
	for (int pass = 0; pass < 3; pass++)
		for (int i = 0; i < N; i++) rtcIntersect1( embreeScene, rayhits + i );
	float traceTimeEmbree = t.elapsed() / 3.0f;
	// retrieve intersection results
	for (int i = 0; i < N; i++)
	{
		rays[i].hit.t = rayhits[i].ray.tfar;
		rays[i].hit.u = rayhits[i].hit.u, rays[i].hit.u = rayhits[i].hit.v;
		rays[i].hit.prim = rayhits[i].hit.primID;
	}
	mrays = (float)N / traceTimeEmbree;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeEmbree * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

#ifdef TRAVERSE_2WAY_MT_DIVERGENT

	// shuffle rays for the next experiment - TODO: replace by random bounce
	for (int i = 0; i < N; i++)
	{
		int j = (unsigned)(i + 17 * rand()) % N;
		Ray t = rays[i];
		rays[i] = rays[j];
		rays[j] = t;
	}

	// trace all rays three times to estimate average performance
	// - divergent distribution, multi-core
	printf( "- CPU, incoherent, basic 2-way layout, MT: " );
	t.reset();
	for (int j = 0; j < 3; j++)
	{
		const int batchCount = N / 10000;
	#pragma omp parallel for schedule(dynamic)
		for (int batch = 0; batch < batchCount; batch++)
		{
			const int batchStart = batch * 10000;
			for (int i = 0; i < 10000; i++) bvh.Intersect( rays[batchStart + i] );
		}
	}
	float traceTimeMTI = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeMTI;
	printf( "%8.1fms for %6.2fM rays => %6.2fMRay/s\n", traceTimeMTI * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

	printf( "all done." );
	return 0;
}
