#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include "stdlib.h"

#define TRIANGLE_COUNT	128

struct TriVertex { float x, y, z, dummy; };
TriVertex triangles[TRIANGLE_COUNT * 3];

float uniform_rand() { return (float)rand() / RAND_MAX; }

int main()
{
	// create a scene consisting of some random small triangles
	for( int i = 0; i < TRIANGLE_COUNT; i++ )
	{
		// create a random triangle
		TriVertex& v0 = triangles[i * 3 + 0];
		TriVertex& v1 = triangles[i * 3 + 1];
		TriVertex& v2 = triangles[i * 3 + 2];
		// triangle position, x/y/z = 0..1
		float x = uniform_rand();
		float y = uniform_rand();
		float z = uniform_rand();
		// set first vertex
		v0.x = x + 0.1f * uniform_rand();
		v0.y = y + 0.1f * uniform_rand();
		v0.z = z + 0.1f * uniform_rand();
		// set second vertex
		v1.x = x + 0.1f * uniform_rand();
		v1.y = y + 0.1f * uniform_rand();
		v1.z = z + 0.1f * uniform_rand();
		// set third vertex
		v2.x = x + 0.1f * uniform_rand();
		v2.y = y + 0.1f * uniform_rand();
		v2.z = z + 0.1f * uniform_rand();
	}
	// build a BVH over the scene
	tinybvh::BVH bvh;
	bvh.Build( (tinybvh::bvhvec4*)triangles, TRIANGLE_COUNT );
	// all done.
	return 0;
}