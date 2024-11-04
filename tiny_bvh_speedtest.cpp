#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#ifdef _MSC_VER
#include "stdio.h"		// for printf
#include "stdlib.h"		// for rand
#else
#include <cstdio>
#endif

// 'screen resolution': see tiny_bvh_fenster.cpp; this program traces the
// same rays, but without visualization - just performance statistics.
#define SCRWIDTH	800
#define SCRHEIGHT	600

using namespace tinybvh;

bvhvec4 triangles[259 /* level 3 */ * 6 * 2 * 49 * 3]{};
int verts = 0;
BVH bvh;

float uniform_rand() { return (float)rand() / (float)RAND_MAX; }

#include <chrono>
struct Timer
{
	Timer() { reset(); }
	float elapsed() const
	{
		auto t2 = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::duration<float>>(t2 - start).count();
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
	// generate a sphere flake scene
	sphere_flake( 0, 0, 0, 1.5f );

	// setup view pyramid for a pinhole camera: 
	// eye, p1 (top-left), p2 (top-right) and p3 (bottom-left)
	bvhvec3 eye( -3.5f, -1.5f, -6.5f ), view = normalize( bvhvec3( 3, 1.5f, 5 ) );
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) );
	bvhvec3 up = 0.8f * cross( view, right ), C = eye + 2 * view;
	bvhvec3 p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;

	// generate primary rays in a cacheline-aligned buffer - and, for data locality:
	// organized in 4x4 pixel tiles, 16 samples per pixel, so 256 rays per tile.
	int N = 0;
	Ray* rays = (Ray*)ALIGNED_MALLOC( SCRWIDTH * SCRHEIGHT * 16 * sizeof( Ray ) );
	for( int ty = 0; ty < SCRHEIGHT / 4; ty++ ) for( int tx = 0; tx < SCRWIDTH / 4; tx++ )
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

	int minor = TINY_BVH_VERSION_MINOR, major = TINY_BVH_VERSION_MAJOR;
	printf( "tiny_bvh version %i.%i performance statistics\n", major, minor );
	printf( "----------------------------------------------------------------\n" );

	Timer t;
	float mrays;

	// measure single-core bvh construction time - warming caches
	printf( "BVH construction speed\n" );
	printf( "warming caches...\n" );
	bvh.Build( (bvhvec4*)triangles, verts / 3 );

#if 1

	// measure single-core bvh construction time - reference builder
	t.reset();
	printf( "- reference builder: " );
	for (int pass = 0; pass < 3; pass++)
		bvh.Build( (bvhvec4*)triangles, verts / 3 );
	float buildTime = t.elapsed() / 3.0f;
	printf( "%.2fms for %i triangles ", buildTime * 1000.0f, verts / 3 );
	printf( "- %i nodes, SAH=%.2f\n", bvh.newNodePtr, bvh.SAHCost() );

#ifdef BVH_USEAVX
	// measure single-core bvh construction time - AVX builder
	t.reset();
	printf( "- fast AVX builder:  " );
	for (int pass = 0; pass < 3; pass++) bvh.BuildAVX( (bvhvec4*)triangles, verts / 3 );
	float buildTimeAVX = t.elapsed() / 3.0f;
	printf( "%.2fms for %i triangles ", buildTimeAVX * 1000.0f, verts / 3 );
	printf( "- %i nodes, SAH=%.2f\n", bvh.newNodePtr, bvh.SAHCost() );
#endif

	// trace all rays once to warm the caches
	printf( "BVH traversal speed\n" );
	printf( "warming caches...\n" );
	for (int i = 0; i < N; i++) bvh.Intersect( rays[i] );

	// trace all rays three times to estimate average performance
	// - single core version
	t.reset();
	printf( "- CPU, coherent, basic 2-way layout, ST: " );
	for (int pass = 0; pass < 3; pass++)
		for (int i = 0; i < N; i++) bvh.Intersect( rays[i] );
	float traceTimeST = t.elapsed() / 3.0f;
	mrays = (float)N / traceTimeST;
	printf( "%.2fms for %.2fM rays (%.2fMRays/s)\n", traceTimeST * 1000, (float)N * 1e-6f, mrays * 1e-6f );

	// trace all rays three times to estimate average performance
	// - multi-core version (using OpenMP and batches of 10,000 rays)
	t.reset();
	printf( "- CPU, coherent, basic 2-way layout, MT:  " );
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
	printf( "%.2fms for %.2fM rays (%.2fMRays/s)\n", traceTimeMT * 1000, (float)N * 1e-6f, mrays * 1e-6f );

#endif

	// trace all rays three times to estimate average performance
	// - coherent distribution, multi-core, packet traversal
	t.reset();
	printf( "- CPU, coherent, basic 2-way layout, MT, packets:  " );
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
	printf( "%.2fms for %.2fM rays (%.2fMRays/s)\n", traceTimeMTP * 1000, (float)N * 1e-6f, mrays * 1e-6f );

	// shuffle rays for the next experiment - TODO: replace by random bounce
	for( int i = 0; i < N; i++ )
	{
		int j = (i + 17 * rand()) % N;
		Ray t = rays[i];
		rays[i] = rays[j];
		rays[j] = t;
	}

	// trace all rays three times to estimate average performance
	// - divergent distribution, multi-core
	t.reset();
	printf( "- CPU, incoherent, basic 2-way layout, MT:  " );
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
	printf( "%.2fms for %.2fM rays (%.2fMRays/s)\n", traceTimeMTI * 1000, (float)N * 1e-6f, mrays * 1e-6f );

	// all done.
	return 0;
}