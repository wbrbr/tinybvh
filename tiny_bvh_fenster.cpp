#include "external/fenster.h" // https://github.com/zserge/fenster
#include <chrono>

#define SCRWIDTH 800
#define SCRHEIGHT 600


struct Timer
{
	Timer() { reset(); }
	float elapsed() const
	{
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - start);
		return (float)time_span.count();
	}
	void reset() { start = std::chrono::high_resolution_clock::now(); }
	std::chrono::high_resolution_clock::time_point start;
};


void Init();
void Tick(float delta_time_s, fenster& f, uint32_t* buf);
void Shutdown();


// #define USE_EMBREE // enable to verify correct implementation, win64 only for now.
#define LOADSCENE

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <fstream>

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

#ifdef LOADSCENE
bvhvec4* triangles = 0;
const char scene[] = "cryteksponza.bin";
#else
ALIGNED( 16 ) bvhvec4 triangles[259 /* level 3 */ * 6 * 2 * 49 * 3]{};
#endif
int verts = 0;

// setup view pyramid for a pinhole camera: 
// eye, p1 (top-left), p2 (top-right) and p3 (bottom-left)
#ifdef LOADSCENE
static bvhvec3 eye( 0, 30, 0 ), p1, p2, p3;
static bvhvec3 view = normalize( bvhvec3( -8, 2, -1.7f ) );
#else
static bvhvec3 eye( -3.5f, -1.5f, -6.5f ), p1, p2, p3;
static bvhvec3 view = normalize( bvhvec3( 3, 1.5f, 5 ) );
#endif

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
#ifdef LOADSCENE
	// load raw vertex data for Crytek's Sponza
	std::string filename{ "../testdata/" };
	filename += scene;
	std::fstream s{ filename, s.binary | s.in };
	if (!s.is_open())
	{
		// try again, look in .\testdata
		std::string filename{ "./testdata/" };
		filename += scene;
		s = std::fstream{ filename, s.binary | s.in };
		assert( s.is_open() );
	}
	s.seekp( 0 );
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)malloc64( verts * 16 );
	s.read( (char*)triangles, verts * 16 );
	s.close();
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
	bvh.BuildHQ( triangles, verts / 3 );
	bvh.Convert( BVH::WALD_32BYTE, BVH::BASIC_BVH4 );
	bvh.Convert( BVH::BASIC_BVH4, BVH::BVH4_AFRA );
#elif defined(BVH_USENEON)
	bvh.BuildNEON( triangles, verts / 3 );
#else
	// bvh.Build( triangles, verts / 3 );
#endif

#endif

	// load camera position / direction from file
	std::fstream t = std::fstream{ "camera.bin", t.binary | t.in };
	if (!t.is_open()) return;
	t.seekp( 0 );
	t.read( (char*)&eye, sizeof( eye ) );
	t.read( (char*)&view, sizeof( view ) );
	t.close();
}


void UpdateCamera(float delta_time_s, fenster& f)
{
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) );
	bvhvec3 up = 0.8f * cross( view, right );
	int64_t new_fenster_time = fenster_time();

	// get camera controls.

	if (f.keys['A']) eye += right * -1.0f * delta_time_s * 10;
	if (f.keys['D']) eye += right * delta_time_s * 10;
	if (f.keys['W']) eye += view * delta_time_s * 10;
	if (f.keys['S']) eye += view * -1.0f * delta_time_s * 10;
	if (f.keys['R']) eye += up * delta_time_s * 10;
	if (f.keys['F']) eye += up * -1.0f * delta_time_s * 10;

	// recalculate right, up
	right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) );
	up = 0.8f * cross( view, right );
	bvhvec3 C = eye + 2 * view;
	p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;
}

void Tick(float delta_time_s, fenster & f, uint32_t* buf)
{
	// handle user input and update camera
	UpdateCamera(delta_time_s, f);

	// clear the screen with a debug-friendly color
	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++) buf[i] = 0xff00ff;

	// generate primary rays in a cacheline-aligned buffer - and, for data locality:
	// organized in 4x4 pixel tiles, 16 samples per pixel, so 256 rays per tile.
	int N = 0;
	Ray* rays = (Ray*)tinybvh::malloc64( SCRWIDTH * SCRHEIGHT * 16 * sizeof( Ray ) );
	int * depths = (int *)tinybvh::malloc64(SCRWIDTH * SCRHEIGHT * sizeof (int));
	for (int ty = 0; ty < SCRHEIGHT; ty += 4) for (int tx = 0; tx < SCRWIDTH; tx += 4 )
	{
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
		{
			float u = (float)(tx + x) / SCRWIDTH, v = (float)(ty + y) / SCRHEIGHT;
			bvhvec3 D = normalize( p1 + u * (p2 - p1) + v * (p3 - p1) - eye );
			rays[N++] = Ray( eye, D, 1e30f );
		}
	}

	// trace primary rays
#if !defined USE_EMBREE
	for (int i = 0; i < N; i++) depths[i] = bvh.Intersect( rays[i], BVH::BVH4_AFRA );
#else
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
#endif

	// visualize result
	const bvhvec3 L = normalize( bvhvec3( 1, 2, 3 ) );
	for (int i = 0, ty = 0; ty < SCRHEIGHT / 4; ty++) for (int tx = 0; tx < SCRWIDTH / 4; tx++)
	{
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++, i++) if (rays[i].hit.t < 10000)
		{
			int pixel_x = tx * 4 + x, pixel_y = ty * 4 + y, primIdx = rays[i].hit.prim;
			bvhvec3 v0 = triangles[primIdx * 3 + 0];
			bvhvec3 v1 = triangles[primIdx * 3 + 1];
			bvhvec3 v2 = triangles[primIdx * 3 + 2];
			bvhvec3 N = normalize( cross( v1 - v0, v2 - v0 ) );
			int c = (int)(255.9f * fabs( dot( N, L ) ));
			buf[pixel_x + pixel_y * SCRWIDTH] = c + (c << 8) + (c << 16);
			//buf[pixel_x + pixel_y * SCRWIDTH] = (primIdx * 0xdeece66d + 0xb) & 0xFFFFFF;
			buf[pixel_x + pixel_y * SCRWIDTH] = c + (c << 8) + depths[i] << 18;//
		}
	}
	tinybvh::free64( rays );
}

void Shutdown()
{
	// save camera position / direction to file
	std::fstream s = std::fstream{ "camera.bin", s.binary | s.out };
	s.seekp( 0 );
	s.write( (char*)&eye, sizeof( eye ) );
	s.write( (char*)&view, sizeof( view ) );
	s.close();
}



int run()
{
	uint32_t* buf = new uint32_t[SCRWIDTH * SCRHEIGHT];
	struct fenster f = { .title = "tiny_bvh", .width = SCRWIDTH, .height = SCRHEIGHT, .buf = buf, };
	
	fenster_open(&f);
	Timer t;
	Init();
	t.reset();
	while (fenster_loop(&f) == 0) {
		float elapsed = t.elapsed();
		t.reset();
		Tick(elapsed, f, buf);
		if (f.keys[27]) break;
	}
	Shutdown();
	fenster_close(&f);
	delete[] buf;
	return 0;
}

#if defined(_WIN32)
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine,
	int nCmdShow) {
	(void)hInstance, (void)hPrevInstance, (void)pCmdLine, (void)nCmdShow;
	return run();
}
#else
int main() { return run(); }
#endif