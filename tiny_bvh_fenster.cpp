#include "external/fenster.h" // https://github.com/zserge/fenster

// #define USE_NANORT // enable to verify correct implementation

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#ifdef USE_NANORT
#include "external/nanort.h"
static nanort::BVHAccel<float> accel;
static float* nanort_verts = 0;
static unsigned int* nanort_faces = 0;
#else
BVH bvh;
#endif

using namespace tinybvh;

bvhvec4 triangles[259 /* level 3 */ * 6 * 2 * 49 * 3]{};
int verts = 0;

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

void Init()
{
	// generate a sphere flake scene
	sphere_flake( 0, 0, 0, 1.5f );

#ifndef USE_NANORT

	// build a BVH over the scene
	bvh.Build( (bvhvec4*)triangles, verts / 3 );

#else

	// convert data to correct format for NanoRT and build a BVH
	// https://github.com/lighttransport/nanort
	nanort_verts = new float[verts * 3];
	nanort_faces = new unsigned int[verts];
	for (int i = 0; i < verts; i++)
		nanort_verts[i * 3 + 0] = triangles[i].x, nanort_verts[i * 3 + 1] = triangles[i].y,
		nanort_verts[i * 3 + 2] = triangles[i].z, nanort_faces[i] = i; // Note: not using shared vertices.
	nanort::TriangleMesh<float> triangle_mesh( nanort_verts, nanort_faces, sizeof( float ) * 3 );
	nanort::TriangleSAHPred<float> triangle_pred( nanort_verts, nanort_faces, sizeof( float ) * 3 );
	nanort::BVHBuildOptions<float> build_options; // BVH build option(optional)
	accel.Build( verts / 3, triangle_mesh, triangle_pred, build_options );

#endif

}

void Tick( uint32_t* buf )
{
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

	// trace primary rays
#ifndef USE_NANORT
#if 0
	const int packetCount = N / 256;
	for (int i = 0; i < packetCount; i++) bvh.Intersect256Rays( rays + i * 256 );
#else
	for (int i = 0; i < N; i++) bvh.Intersect( rays[i] );
#endif
#else
	nanort::Ray<float> ray;
	nanort::BVHTraceOptions trace_options; // optional
	nanort::TriangleIntersector<float> triangle_intersector( nanort_verts, nanort_faces, sizeof( float ) * 3 );
	for (int i = 0; i < N; i += 16)
	{
		ray.org[0] = rays[i].O.x, ray.org[1] = rays[i].O.y, ray.org[2] = rays[i].O.z;
		ray.dir[0] = rays[i].D.x, ray.dir[1] = rays[i].D.y, ray.dir[2] = rays[i].D.z;
		ray.min_t = 0, ray.max_t = rays[i].hit.t;
		nanort::TriangleIntersection<float> isect;
		bool hit = accel.Traverse( ray, triangle_intersector, &isect, trace_options );
		if (hit)
			rays[i].hit.t = isect.t,
			rays[i].hit.u = isect.u, rays[i].hit.v = isect.v,
			rays[i].hit.prim = isect.prim_id;
	}
#endif

	// visualize result
	for (int i = 0, ty = 0; ty < SCRHEIGHT / 4; ty++) for (int tx = 0; tx < SCRWIDTH / 4; tx++)
	{
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
		{
			int pixel_x = tx * 4 + x;
			int pixel_y = ty * 4 + y;
			float avg = 0;
			for (int s = 0; s < 16; s++, i++) if (rays[i].hit.t < 1000)
			{
				int primIdx = rays[i].hit.prim;
				bvhvec3 v0 = triangles[primIdx * 3 + 0];
				bvhvec3 v1 = triangles[primIdx * 3 + 1];
				bvhvec3 v2 = triangles[primIdx * 3 + 2];
				bvhvec3 N = normalize( cross( v1 - v0, v2 - v0 ) );
				avg += fabs( dot( N, normalize( bvhvec3( 1, 2, 3 ) ) ) );
			}
		#ifndef USE_NANORT
			int c = (int)(15.9f * avg);
		#else
			int c = (int)(255.9f * avg); // we trace only every 16th ray with NanoRT
		#endif
			buf[pixel_x + pixel_y * SCRWIDTH] = c + (c << 8) + (c << 16);
		}
	}
	ALIGNED_FREE( rays );
}
