#include "external/fenster.h" // https://github.com/zserge/fenster

// #define USE_EMBREE // enable to verify correct implementation, win64 only for now.
// #define LOADSPONZA

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

using namespace tinybvh;

#if defined(USE_EMBREE)
#include "embree4/rtcore.h"
static RTCScene embreeScene;
void embreeError( void* userPtr, enum RTCError error, const char* str )
{
	printf( "error %d: %s\n", error, str );
}
#else
BVH bvh;
#endif

#ifdef LOADSPONZA
bvhvec4* triangles = 0;
#include <fstream>
#else
ALIGNED( 16 ) bvhvec4 triangles[259 /* level 3 */ * 6 * 2 * 49 * 3]{};
#endif
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
#ifdef LOADSPONZA
	// load raw vertex data for Crytek's Sponza
	std::string filename{ "../testdata/cryteksponza.bin" };
	std::fstream s{ filename, s.binary | s.in };
	if (!s.is_open())
	{
		// try again, look in .\testdata
		std::string filename{ "./testdata/cryteksponza.bin" };
		s = std::fstream{ filename, s.binary | s.in };
		assert( s.is_open() );
	}
	s.seekp( 0 );
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)default_malloc( verts * 16 );
	s.read( (char*)triangles, verts * 16 );
#else
	// generate a sphere flake scene
	sphere_flake( 0, 0, 0, 1.5f );
#endif

#if defined USE_EMBREE

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
	rtcCommitGeometry( embreeGeom );
	rtcAttachGeometry( embreeScene, embreeGeom );
	rtcReleaseGeometry( embreeGeom );
	rtcCommitScene( embreeScene );

#else

	// build a BVH over the scene
#if defined(BVH_USEAVX)
	bvh.BuildAVX( triangles, verts / 3 );
#elif defined(BVH_USENEON)
	bvh.BuildNEON( triangles, verts / 3 );
#else
	// bvh.Build( triangles, verts / 3 );
#endif

#endif

}

void Tick( uint32_t* buf )
{
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
	Ray* rays = (Ray*)default_malloc( SCRWIDTH * SCRHEIGHT * 16 * sizeof( Ray ) );
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
#if defined(USE_EMBREE)
	struct RTCRayHit rayhit;
	for (int i = 0; i < N; i++)
	{
		rayhit.ray.org_x = rays[i].O.x, rayhit.ray.org_y = rays[i].O.y, rayhit.ray.org_z = rays[i].O.z;
		rayhit.ray.dir_x = rays[i].D.x, rayhit.ray.dir_y = rays[i].D.y, rayhit.ray.dir_z = rays[i].D.z;
		rayhit.ray.tnear = 0, rayhit.ray.tfar = rays[i].hit.t, rayhit.ray.mask = -1, rayhit.ray.flags = 0;
		rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID, rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
		rtcIntersect1( embreeScene, &rayhit );
		rays[i].hit.u = rayhit.hit.u, rays[i].hit.u = rayhit.hit.v;
		rays[i].hit.prim = rayhit.hit.primID, rays[i].hit.t = rayhit.ray.tfar;
	}
#else
	for (int i = 0; i < N; i++) bvh.Intersect( rays[i] );
#endif

	// visualize result
	for (int i = 0, ty = 0; ty < SCRHEIGHT / 4; ty++) for (int tx = 0; tx < SCRWIDTH / 4; tx++)
	{
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
		{
			int pixel_x = tx * 4 + x;
			int pixel_y = ty * 4 + y;
			float avg = 0;
			for (int s = 0; s < 16; s++, i++) if (rays[i].hit.t < 10000)
			{
				int primIdx = rays[i].hit.prim;
				bvhvec3 v0 = triangles[primIdx * 3 + 0];
				bvhvec3 v1 = triangles[primIdx * 3 + 1];
				bvhvec3 v2 = triangles[primIdx * 3 + 2];
				bvhvec3 N = normalize( cross( v1 - v0, v2 - v0 ) );
				avg += fabs( dot( N, normalize( bvhvec3( 1, 2, 3 ) ) ) );
			}
			int c = (int)(15.9f * avg);
			buf[pixel_x + pixel_y * SCRWIDTH] = c + (c << 8) + (c << 16);
		}
	}
	default_free( rays );
}