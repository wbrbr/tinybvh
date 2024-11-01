#include "external/fenster.h" // https://github.com/zserge/fenster

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#ifdef _MSC_VER
#include "stdlib.h"		// for rand
#include "stdio.h"		// for printf
#else
#include <cstdlib>
#include <cstdio>
#endif

using namespace tinybvh;

#define SPHERE_COUNT	259

struct TriVertex {
	TriVertex() = default;
	TriVertex( bvhvec3 v ) : x( v.x ), y( v.y ), z( v.z ), dummy( 0 ) {}
	float x, y, z, dummy;
};
TriVertex triangles[SPHERE_COUNT * 6 * 2 * 49 * 3]{};
int verts = 0, spheres = 0;
BVH bvh;

void create_sphere( float x, float y, float z, float s )
{
#define P(F,a,b,c) p[i+F*64]={(float)a ,(float)b,(float)c}
	bvhvec3 p[384], pos( x, y, z ), ofs( 3.5 );
	for (int i = 0, u = 0; u < 8; u++) for (int v = 0; v < 8; v++, i++)
		P( 0, u, v, 0 ), P( 1, u, 0, v ), P( 2, 0, u, v ),
		P( 3, u, v, 7 ), P( 4, u, 7, v ), P( 5, 7, u, v );
	for (int i = 0; i < 384; i++) p[i] = normalize( p[i] - ofs ) * s + pos;
	for ( int i = 0, side = 0; side < 6; side++, i += 8 )
		for (int u = 0; u < 7; u++, i++) for (int v = 0; v < 7; v++, i++)
			triangles[verts++] = p[i], triangles[verts++] = p[i + 8],
			triangles[verts++] = p[i + 1], triangles[verts++] = p[i + 1],
			triangles[verts++] = p[i + 9], triangles[verts++] = p[i + 8];
}

void sphere_flake( float x, float y, float z, float s, int d = 0 )
{
	spheres++;
	create_sphere( x, y, z, s * 0.5f );
	if (d < 3) sphere_flake( x + s * 0.75f, y, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x - s * 0.75f, y, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, y + s * 0.75f, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, x - s * 0.75f, z, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, y, z + s * 0.75f, s * 0.5f, d + 1 );
	if (d < 3) sphere_flake( x, y, z - s * 0.75f, s * 0.5f, d + 1 );
}

void Init()
{
	// generate a sphere flake scene
	sphere_flake( 0, 0, 0, 3 );

	// build a BVH over the scene
	bvh.Build( (bvhvec4*)triangles, verts / 3 );
}

void Tick( uint32_t* buf )
{
	// trace primary rays
	bvhvec3 eye( -3.5f, -1.5f, -6.5f ), view = normalize( bvhvec3( 3, 1.5f, 5 ) );
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) );
	bvhvec3 up = 0.8f * cross( view, right ), C = eye + 2 * view;
	bvhvec3 p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;
	static int s, x, y = -1;
	if (y < SCRHEIGHT - 1) for( y++, x = 0; x < SCRWIDTH; x++ )
	{
		float u = (float)x / SCRWIDTH;
		float v = (float)y / SCRHEIGHT;
		bvhvec3 P = p1 + u * (p2 - p1) + v * (p3 - p1);
		Ray ray( eye, normalize( P - eye ) );
		bvh.Intersect( ray );
		int c = 0;
		if (ray.hit.t < 1000)
		{
			int primIdx = ray.hit.prim;
			bvhvec3 v0 = *(bvhvec3*)&triangles[primIdx * 3 + 0];
			bvhvec3 v1 = *(bvhvec3*)&triangles[primIdx * 3 + 1];
			bvhvec3 v2 = *(bvhvec3*)&triangles[primIdx * 3 + 2];
			bvhvec3 N = normalize( cross( v1 - v0, v2 - v0 ) );
			c = (int)( 200.0f * fabs( dot( N, normalize( bvhvec3( 1, 2, 3 ) ) ) ) ) + 55;
		}
		buf[x + y * SCRWIDTH] = c + (c << 8) + (c << 16); 
	}
	else y = 0;
}