/*
The MIT License (MIT)

Copyright (c) 2024, Jacco Bikker / Breda University of Applied Sciences.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Nov 11, '24: version 0.5.0 : SBVH builder.
// Nov 10, '24: version 0.4.2 : BVH4/8, gpu-friendly BVH4.
// Nov 09, '24: version 0.4.0 : Layouts, BVH optimizer.
// Nov 08, '24: version 0.3.0
// Oct 30, '24: version 0.1.0 : Initial release.
// Oct 29, '24: version 0.0.1 : Establishing interface.
//

//
// Use this in *one* .c or .cpp
//   #define TINYBVH_IMPLEMENTATION
//   #include "tiny_bvh.h"
//

// How to use:
// See tiny_bvh_test.cpp for basic usage. In short:
// instantiate a BVH: tinybvh::BVH bvh;
// build it: bvh.Build( (tinybvh::bvhvec4*)triangleData, TRIANGLE_COUNT );
// ..where triangleData is an array of four-component float vectors:
// - For a single triangle, provide 3 vertices,
// - For each vertex provide x, y and z.
// The fourth float in each vertex is a dummy value and exists purely for
// a more efficient layout of the data in memory.

// More information about the BVH data structure:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics

// Further references:
// - Parallel Spatial Splits in Bounding Volume Hierarchies:
//   https://diglib.eg.org/items/f55715b1-9e56-4b40-af73-59d3dfba9fe7
// - Heuristics for ray tracing using space subdivision:
//   https://graphicsinterface.org/wp-content/uploads/gi1989-22.pdf
// - Heuristic Ray Shooting Algorithms:
//   https://dcgi.fel.cvut.cz/home/havran/DISSVH/phdthesis.html

// Author and contributors:
// Jacco Bikker: BVH code and examples
// Eddy L O Jansson: g++ / clang support
// Aras Pranckevičius: non-Intel architecture support
// Jefferson Amstutz: CMake surpport

#ifndef TINY_BVH_H_
#define TINY_BVH_H_

// binned BVH building: bin count
#define BVHBINS 8

// include fast AVX BVH builder
#if defined(__x86_64__) || defined(_M_X64)
#define BVH_USEAVX
#endif

// library version
#define TINY_BVH_VERSION_MAJOR	0
#define TINY_BVH_VERSION_MINOR	5
#define TINY_BVH_VERSION_SUB	0

// ============================================================================
//
//        P R E L I M I N A R I E S
// 
// ============================================================================

// aligned memory allocation
#ifdef _MSC_VER
// Visual Studio / C11
#include <malloc.h>
#include <math.h> // for sqrtf, fabs
#include <string.h> // for memset
#define ALIGNED( x ) __declspec( align( x ) )
#define ALIGNED_MALLOC( x ) ( ( x ) == 0 ? 0 : _aligned_malloc( ( x ), 64 ) )
#define ALIGNED_FREE( x ) _aligned_free( x )
#else
// gcc / clang
#include <cstdlib>
#include <cmath>
#include <cstring>
#define ALIGNED( x ) __attribute__( ( aligned( x ) ) )
#if defined(__x86_64__) || defined(_M_X64)
// https://stackoverflow.com/questions/32612881/why-use-mm-malloc-as-opposed-to-aligned-malloc-alligned-alloc-or-posix-mem
#include <xmmintrin.h>
#define ALIGNED_MALLOC( x ) ( ( x ) == 0 ? 0 : _mm_malloc( ( x ), 64 ) )
#define ALIGNED_FREE( x ) _mm_free( x )
#else
#define ALIGNED_MALLOC( x ) ( ( x ) == 0 ? 0 : aligned_alloc( 64, ( x ) ) )
#define ALIGNED_FREE( x ) free( x )
#endif
#endif
// generic
#ifdef BVH_USEAVX
#include "immintrin.h" // for __m128 and __m256
#endif

namespace tinybvh {

#ifdef _MSC_VER
// Suppress a warning caused by the union of x,y,.. and cell[..] in vectors.
// We need this union to address vector components either by name or by index.
// The warning is re-enabled right after the definition of the data types.
#pragma warning ( push )
#pragma warning ( disable: 4201 /* nameless struct / union */ )
#endif

struct bvhvec3;
struct ALIGNED( 16 ) bvhvec4
{
	// vector naming is designed to not cause any name clashes.
	bvhvec4() = default;
	bvhvec4( const float a, const float b, const float c, const float d ) : x( a ), y( b ), z( c ), w( d ) {}
	bvhvec4( const float a ) : x( a ), y( a ), z( a ), w( a ) {}
	bvhvec4( const bvhvec3 & a );
	bvhvec4( const bvhvec3 & a, float b );
	float& operator [] ( const int i ) { return cell[i]; }
	union { struct { float x, y, z, w; }; float cell[4]; };
};

struct ALIGNED( 8 ) bvhvec2
{
	bvhvec2() = default;
	bvhvec2( const float a, const float b ) : x( a ), y( b ) {}
	bvhvec2( const float a ) : x( a ), y( a ) {}
	bvhvec2( const bvhvec4 a ) : x( a.x ), y( a.y ) {}
	float& operator [] ( const int i ) { return cell[i]; }
	union { struct { float x, y; }; float cell[2]; };
};

struct bvhvec3
{
	bvhvec3() = default;
	bvhvec3( const float a, const float b, const float c ) : x( a ), y( b ), z( c ) {}
	bvhvec3( const float a ) : x( a ), y( a ), z( a ) {}
	bvhvec3( const bvhvec4 a ) : x( a.x ), y( a.y ), z( a.z ) {}
	float halfArea() { return x < -1e30f ? 0 : (x * y + y * z + z * x); } // for SAH calculations
	float& operator [] ( const int i ) { return cell[i]; }
	union { struct { float x, y, z; }; float cell[3]; };
};
struct bvhint3
{
	bvhint3() = default;
	bvhint3( const int a, const int b, const int c ) : x( a ), y( b ), z( c ) {}
	bvhint3( const int a ) : x( a ), y( a ), z( a ) {}
	bvhint3( const bvhvec3& a ) { x = (int)a.x, y = (int)a.y, z = (int)a.z; }
	int& operator [] ( const int i ) { return cell[i]; }
	union { struct { int x, y, z; }; int cell[3]; };
};

#ifdef TINYBVH_IMPLEMENTATION
bvhvec4::bvhvec4( const bvhvec3& a ) { x = a.x; y = a.y; z = a.z; w = 0; }
bvhvec4::bvhvec4( const bvhvec3& a, float b ) { x = a.x; y = a.y; z = a.z; w = b; }
#endif

#ifdef _MSC_VER
#pragma warning ( pop )
#endif

// Math operations.
// Note: Since this header file is expected to be included in a source file
// of a separate project, the static keyword doesn't provide sufficient
// isolation; hence the tinybvh_ prefix.
inline float tinybvh_safercp( const float x ) { return x > 1e-12f ? (1.0f / x) : (x < -1e-12f ? (1.0f / x) : 1e30f); }
inline bvhvec3 tinybvh_safercp( const bvhvec3 a ) { return bvhvec3( tinybvh_safercp( a.x ), tinybvh_safercp( a.y ), tinybvh_safercp( a.z ) ); }
static inline float tinybvh_min( const float a, const float b ) { return a < b ? a : b; }
static inline float tinybvh_max( const float a, const float b ) { return a > b ? a : b; }
static inline int tinybvh_min( const int a, const int b ) { return a < b ? a : b; }
static inline int tinybvh_max( const int a, const int b ) { return a > b ? a : b; }
static inline bvhvec2 tinybvh_min( const bvhvec2& a, const bvhvec2& b ) { return bvhvec2( tinybvh_min( a.x, b.x ), tinybvh_min( a.y, b.y ) ); }
static inline bvhvec3 tinybvh_min( const bvhvec3& a, const bvhvec3& b ) { return bvhvec3( tinybvh_min( a.x, b.x ), tinybvh_min( a.y, b.y ), tinybvh_min( a.z, b.z ) ); }
static inline bvhvec4 tinybvh_min( const bvhvec4& a, const bvhvec4& b ) { return bvhvec4( tinybvh_min( a.x, b.x ), tinybvh_min( a.y, b.y ), tinybvh_min( a.z, b.z ), tinybvh_min( a.w, b.w ) ); }
static inline bvhvec2 tinybvh_max( const bvhvec2& a, const bvhvec2& b ) { return bvhvec2( tinybvh_max( a.x, b.x ), tinybvh_max( a.y, b.y ) ); }
static inline bvhvec3 tinybvh_max( const bvhvec3& a, const bvhvec3& b ) { return bvhvec3( tinybvh_max( a.x, b.x ), tinybvh_max( a.y, b.y ), tinybvh_max( a.z, b.z ) ); }
static inline bvhvec4 tinybvh_max( const bvhvec4& a, const bvhvec4& b ) { return bvhvec4( tinybvh_max( a.x, b.x ), tinybvh_max( a.y, b.y ), tinybvh_max( a.z, b.z ), tinybvh_max( a.w, b.w ) ); }
static inline float tinybvh_clamp( const float x, const float a, const float b ) { return x < a ? a : (x > b ? b : x); }
static inline int tinybvh_clamp( const int x, const int a, const int b ) { return x < a ? a : (x > b ? b : x); }
template <class T> inline static void tinybvh_swap( T& a, T& b ) { T t = a; a = b; b = t; }

// Operator overloads.
// Only a minimal set is provided.
inline bvhvec2 operator-( const bvhvec2& a ) { return bvhvec2( -a.x, -a.y ); }
inline bvhvec3 operator-( const bvhvec3& a ) { return bvhvec3( -a.x, -a.y, -a.z ); }
inline bvhvec4 operator-( const bvhvec4& a ) { return bvhvec4( -a.x, -a.y, -a.z, -a.w ); }
inline bvhvec2 operator+( const bvhvec2& a, const bvhvec2& b ) { return bvhvec2( a.x + b.x, a.y + b.y ); }
inline bvhvec3 operator+( const bvhvec3& a, const bvhvec3& b ) { return bvhvec3( a.x + b.x, a.y + b.y, a.z + b.z ); }
inline bvhvec4 operator+( const bvhvec4& a, const bvhvec4& b ) { return bvhvec4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
inline bvhvec2 operator-( const bvhvec2& a, const bvhvec2& b ) { return bvhvec2( a.x - b.x, a.y - b.y ); }
inline bvhvec3 operator-( const bvhvec3& a, const bvhvec3& b ) { return bvhvec3( a.x - b.x, a.y - b.y, a.z - b.z ); }
inline bvhvec4 operator-( const bvhvec4& a, const bvhvec4& b ) { return bvhvec4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
inline void operator+=( bvhvec2& a, const bvhvec2& b ) { a.x += b.x;	a.y += b.y; }
inline void operator+=( bvhvec3& a, const bvhvec3& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z; }
inline void operator+=( bvhvec4& a, const bvhvec4& b ) { a.x += b.x;	a.y += b.y;	a.z += b.z;	a.w += b.w; }
inline bvhvec2 operator*( const bvhvec2& a, const bvhvec2& b ) { return bvhvec2( a.x * b.x, a.y * b.y ); }
inline bvhvec3 operator*( const bvhvec3& a, const bvhvec3& b ) { return bvhvec3( a.x * b.x, a.y * b.y, a.z * b.z ); }
inline bvhvec4 operator*( const bvhvec4& a, const bvhvec4& b ) { return bvhvec4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
inline bvhvec2 operator*( const bvhvec2& a, float b ) { return bvhvec2( a.x * b, a.y * b ); }
inline bvhvec2 operator*( float b, const bvhvec2& a ) { return bvhvec2( b * a.x, b * a.y ); }
inline bvhvec3 operator*( const bvhvec3& a, float b ) { return bvhvec3( a.x * b, a.y * b, a.z * b ); }
inline bvhvec3 operator*( float b, const bvhvec3& a ) { return bvhvec3( b * a.x, b * a.y, b * a.z ); }
inline bvhvec4 operator*( const bvhvec4& a, float b ) { return bvhvec4( a.x * b, a.y * b, a.z * b, a.w * b ); }
inline bvhvec4 operator*( float b, const bvhvec4& a ) { return bvhvec4( b * a.x, b * a.y, b * a.z, b * a.w ); }
inline bvhvec2 operator/( float b, const bvhvec2& a ) { return bvhvec2( b / a.x, b / a.y ); }
inline bvhvec3 operator/( float b, const bvhvec3& a ) { return bvhvec3( b / a.x, b / a.y, b / a.z ); }
inline bvhvec4 operator/( float b, const bvhvec4& a ) { return bvhvec4( b / a.x, b / a.y, b / a.z, b / a.w ); }
inline bvhvec3 operator*=( bvhvec3& a, const float b ) { return bvhvec3( a.x * b, a.y * b, a.z * b ); }

// Vector math: cross and dot.
static inline bvhvec3 cross( const bvhvec3& a, const bvhvec3& b )
{
	return bvhvec3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}
static inline float dot( const bvhvec2& a, const bvhvec2& b ) { return a.x * b.x + a.y * b.y; }
static inline float dot( const bvhvec3& a, const bvhvec3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float dot( const bvhvec4& a, const bvhvec4& b ) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Vector math: common operations.
static float length( const bvhvec3& a ) { return sqrtf( a.x * a.x + a.y * a.y + a.z * a.z ); }
static bvhvec3 normalize( const bvhvec3& a )
{
	float l = length( a ), rl = l == 0 ? 0 : (1.0f / l);
	return a * rl;
}

// SIMD typedef, helps keeping the interface generic
#ifdef BVH_USEAVX
typedef __m128 SIMDVEC4;
#define SIMD_SETVEC(a,b,c,d) _mm_set_ps( a, b, c, d )
#define SIMD_SETRVEC(a,b,c,d) _mm_set_ps( d, c, b, a )
#else
typedef bvhvec4 SIMDVEC4;
#define SIMD_SETVEC(a,b,c,d) bvhvec4( d, c, b, a )
#define SIMD_SETRVEC(a,b,c,d) bvhvec4( a, b, c, d )
#endif

// ============================================================================
//
//        R A Y   T R A C I N G   S T R U C T S  /  C L A S S E S
// 
// ============================================================================

struct Intersection
{
	// An intersection result is designed to fit in no more than
	// four 32-bit values. This allows efficient storage of a result in
	// GPU code. The obvious missing result is an instance id; consider
	// squeezing this in the 'prim' field in some way.
	// Using this data and the original triangle data, all other info for
	// shading (such as normal, texture color etc.) can be reconstructed.
	float t, u, v;		// distance along ray & barycentric coordinates of the intersection
	unsigned int prim;	// primitive index
};

struct Ray
{
	// Basic ray class. Note: For single blas traversal it is expected
	// that Ray::rD is properly initialized. For tlas/blas traversal this
	// field is typically updated for each blas.
	Ray() = default;
	Ray( bvhvec3 origin, bvhvec3 direction, float t = 1e30f )
	{
		memset( this, 0, sizeof( Ray ) );
		O = origin, D = normalize( direction ), rD = tinybvh_safercp( D );
		hit.t = t;
	}
	ALIGNED( 16 ) bvhvec3 O; unsigned int dummy1;
	ALIGNED( 16 ) bvhvec3 D; unsigned int dummy2;
	ALIGNED( 16 ) bvhvec3 rD; unsigned int dummy3;
	ALIGNED( 16 ) Intersection hit;
};

class BVH
{
public:
	enum BVHLayout { WALD_32BYTE = 1, AILA_LAINE, ALT_SOA, VERBOSE, BASIC_BVH4, BVH4_GPU, BASIC_BVH8 };
	struct BVHNode
	{
		// 'Traditional' 32-byte BVH node layout, as proposed by Ingo Wald.
		// When aligned to a cache line boundary, two of these fit together.
		bvhvec3 aabbMin; unsigned int leftFirst; // 16 bytes
		bvhvec3 aabbMax; unsigned int triCount;	// 16 bytes, total: 32 bytes
		bool isLeaf() const { return triCount > 0; /* empty BVH leaves do not exist */ }
		float Intersect( const Ray& ray ) const { return BVH::IntersectAABB( ray, aabbMin, aabbMax ); }
		float SurfaceArea() const { return BVH::SA( aabbMin, aabbMax ); }
		float CalculateNodeCost() const { return SurfaceArea() * triCount; }
	};
	struct BVHNodeAlt
	{
		// Alternative 64-byte BVH node layout, which specifies the bounds of
		// the children rather than the node itself. This layout is used by
		// Aila and Laine in their seminal GPU ray tracing paper.
		bvhvec3 lmin; unsigned int left;
		bvhvec3 lmax; unsigned int right;
		bvhvec3 rmin; unsigned int triCount;
		bvhvec3 rmax; unsigned int firstTri; // total: 64 bytes
		bool isLeaf() const { return triCount > 0; }
	};
	struct BVHNodeAlt2
	{
		// Second alternative 64-byte BVH node layout, same as BVHNodeAlt but
		// with child AABBs stored in SoA order.
		SIMDVEC4 xxxx, yyyy, zzzz;
		unsigned int left, right, triCount, firstTri; // total: 64 bytes
		bool isLeaf() const { return triCount > 0; }
	};
	struct BVHNodeVerbose
	{
		// This node layout has some extra data per node: It stores left and right
		// child node indices explicitly, and stores the index of the parent node.
		// This format exists primarily for the BVH optimizer.
		bvhvec3 aabbMin; unsigned int left;
		bvhvec3 aabbMax; unsigned int right;
		unsigned int triCount, firstTri, parent, sibling;
		bool isLeaf() const { return triCount > 0; }
	};
	struct BVHNode4
	{
		// 4-wide (aka 'shallow') BVH layout.  
		bvhvec3 aabbMin; unsigned int firstTri;
		bvhvec3 aabbMax; unsigned int triCount;
		unsigned int child[4];
		unsigned int childCount, dummy1, dummy2, dummy3; // dummies are for alignment.
		bool isLeaf() const { return triCount > 0; }
	};
	struct BVHNode4Alt
	{
		// 4-way BVH node, optimized for GPU rendering
		struct aabb8 { unsigned char xmin, ymin, zmin, xmax, ymax, zmax; }; // quantized
		bvhvec3 aabbMin; unsigned int c0Info;			// 16
		bvhvec3 aabbExt; unsigned int c1Info;			// 16
		aabb8 c0bounds, c1bounds; unsigned int c2Info;	// 16
		aabb8 c2bounds, c3bounds; unsigned int c3Info;	// 16; total: 64 bytes
		// childInfo, 32bit:
		// msb:        0=interior, 1=leaf
		// leaf:       16 bits: relative start of triangle data, 15 bits: triangle count.
		// interior:   31 bits: child node address, in float4s from BVH data start.
		// Triangle data: directly follows nodes with leaves. Per tri:
		// - bvhvec4 vert0, vert1, vert2
		// - uint vert0.w stores original triangle index.
		// We can make the node smaller by storing child nodes sequentially, but
		// there is no way we can shave off a full 16 bytes, unless aabbExt is stored
		// as chars as well, as in CWBVH.
	};
	struct BVHNode8
	{
		// 8-wide (aka 'shallow') BVH layout.  
		bvhvec3 aabbMin; unsigned int firstTri;
		bvhvec3 aabbMax; unsigned int triCount;
		unsigned int child[8];
		unsigned int childCount, dummy1, dummy2, dummy3; // dummies are for alignment.
		bool isLeaf() const { return triCount > 0; }
	};
	struct Fragment
	{
		// A fragment stores the bounds of an input primitive. The name 'Fragment' is from
		// "Parallel Spatial Splits in Bounding Volume Hierarchies", 2016, Fuetterling et al.,
		// and refers to the potential splitting of these boxes for SBVH construction.
		bvhvec3 bmin;				// AABB min x, y and z
		unsigned int primIdx;		// index of the original primitive
		bvhvec3 bmax;				// AABB max x, y and z
		unsigned int clipped = 0;	// Fragment is the result of clipping if > 0.
		bool validBox() { return bmin.x < 1e30f; }
	};
	BVH() = default;
	~BVH()
	{
		ALIGNED_FREE( bvhNode );
		ALIGNED_FREE( altNode ); // Note: no action if pointer is null
		ALIGNED_FREE( alt2Node );
		ALIGNED_FREE( verbose );
		ALIGNED_FREE( bvh4Node );
		ALIGNED_FREE( bvh4Alt );
		ALIGNED_FREE( bvh8Node );
		ALIGNED_FREE( triIdx );
		ALIGNED_FREE( fragment );
		bvhNode = 0, triIdx = 0, fragment = 0;
		allocatedBVHNodes = 0;
		allocatedAltNodes = 0;
		allocatedAlt2Nodes = 0;
		allocatedVerbose = 0;
		allocatedBVH4Nodes = 0;
		allocatedAlt4Blocks = 0;
		allocatedBVH8Nodes = 0;
	}
	float SAHCost( const unsigned int nodeIdx = 0 ) const
	{
		// Determine the SAH cost of the tree. This provides an indication
		// of the quality of the BVH: Lower is better.
		const BVHNode& n = bvhNode[nodeIdx];
		if (n.isLeaf()) return 2.0f * n.SurfaceArea() * n.triCount;
		float cost = 3.0f * n.SurfaceArea() + SAHCost( n.leftFirst ) + SAHCost( n.leftFirst + 1 );
		return nodeIdx == 0 ? (cost / n.SurfaceArea()) : cost;
	}
	int NodeCount( const unsigned int nodeIdx = 0 ) const
	{
		// Determine the number of nodes in the tree. Typically the result should
		// be usedBVHNodes - 1 (second node is always unused), but some builders may 
		// have unused nodes besides node 1.
		// TODO: Implement for other layouts.
		const BVHNode& n = bvhNode[nodeIdx];
		unsigned int retVal = 1;
		if (!n.isLeaf()) retVal += NodeCount( n.leftFirst ) + NodeCount( n.leftFirst + 1 );
		return retVal;
	}
	void Build( const bvhvec4* vertices, const unsigned int primCount );
	void BuildHQ( const bvhvec4* vertices, const unsigned int primCount );
	void BuildAVX( const bvhvec4* vertices, const unsigned int primCount );
	void Convert( BVHLayout from, BVHLayout to, bool deleteOriginal = false );
	void Optimize();
	void Refit();
	int Intersect( Ray& ray, BVHLayout layout = WALD_32BYTE ) const;
	void Intersect256Rays( Ray* first ) const;
	void Intersect256RaysSSE( Ray* packet ) const; // requires BVH_USEAVX
private:
	int Intersect_Wald32Byte( Ray& ray ) const;
	int Intersect_AilaLaine( Ray& ray ) const;
	int Intersect_BasicBVH4( Ray& ray ) const;
	int Intersect_BasicBVH8( Ray& ray ) const;
	int Intersect_AltSoA( Ray& ray ) const; // requires BVH_USEAVX
	void IntersectTri( Ray& ray, const unsigned int triIdx ) const;
	static float IntersectAABB( const Ray& ray, const bvhvec3& aabbMin, const bvhvec3& aabbMax );
	static float SA( const bvhvec3& aabbMin, const bvhvec3& aabbMax )
	{
		bvhvec3 e = aabbMax - aabbMin; // extent of the node
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
	bool ClipFrag( const Fragment& orig, Fragment& newFrag, bvhvec3 bmin, bvhvec3 bmax, bvhvec3 minDim );
	void RefitUpVerbose( unsigned int nodeIdx );
	unsigned int FindBestNewPosition( const unsigned int Lid );
	void ReinsertNodeVerbose( const unsigned int Lid, const unsigned int Nid, const unsigned int origin );
public:
	bvhvec4* verts = 0;				// pointer to input primitive array: 3x16 bytes per tri
	unsigned int triCount = 0;		// number of primitives in tris
	Fragment* fragment = 0;			// input primitive bounding boxes
	unsigned int* triIdx = 0;		// primitive index array
	unsigned int idxCount = 0;		// number of indices in triIdx. May exceed triCount * 3 for SBVH.
	BVHNode* bvhNode = 0;			// BVH node pool, Wald 32-byte format. Root is always in node 0.
	BVHNodeAlt* altNode = 0;		// BVH node in Aila & Laine format.
	BVHNodeAlt2* alt2Node = 0;		// BVH node in Aila & Laine (SoA version) format.
	BVHNodeVerbose* verbose = 0;	// BVH node with additional info, for BVH optimizer.
	BVHNode4* bvh4Node = 0;			// BVH node for 4-wide BVH.
	bvhvec4* bvh4Alt = 0;			// 64-byte 4-wide BVH node for efficient GPU rendering.
	BVHNode8* bvh8Node = 0;			// BVH node for 8-wide BVH.
	bool rebuildable = true;		// rebuilds are safe only if a tree has not been converted.
	bool refittable = true;			// refits are safe only if the tree has no spatial splits.
	// keep track of allocated buffer size to avoid 
	// repeated allocation during layout conversion.
	unsigned allocatedBVHNodes = 0;
	unsigned allocatedAltNodes = 0;
	unsigned allocatedAlt2Nodes = 0;
	unsigned allocatedVerbose = 0;
	unsigned allocatedBVH4Nodes = 0;
	unsigned allocatedAlt4Blocks = 0;
	unsigned allocatedBVH8Nodes = 0;
	unsigned usedBVHNodes = 0;
	unsigned usedAltNodes = 0;
	unsigned usedAlt2Nodes = 0;
	unsigned usedVerboseNodes = 0;
	unsigned usedBVH4Nodes = 0;
	unsigned usedAlt4Blocks = 0;
	unsigned usedBVH8Nodes = 0;
};

} // namespace tinybvh

// ============================================================================
//
//        I M P L E M E N T A T I O N
// 
// ============================================================================

#ifdef TINYBVH_IMPLEMENTATION

#include <assert.h>			// for assert

namespace tinybvh {

// Basic single-function binned-SAH-builder. 
// This is the reference builder; it yields a decent tree suitable for ray 
// tracing on the CPU. This code uses no SIMD instructions. 
// Faster code, using SSE/AVX, is available for x64 CPUs.
// For GPU rendering: The resulting BVH should be converted to a more optimal
// format after construction.
void BVH::Build( const bvhvec4* vertices, const unsigned int primCount )
{
	// allocate on first build
	const unsigned int spaceNeeded = primCount * 2; // upper limit
	if (allocatedBVHNodes < spaceNeeded)
	{
		ALIGNED_FREE( bvhNode );
		ALIGNED_FREE( triIdx );
		ALIGNED_FREE( fragment );
		bvhNode = (BVHNode*)ALIGNED_MALLOC( spaceNeeded * sizeof( BVHNode ) );
		allocatedBVHNodes = spaceNeeded;
		memset( &bvhNode[1], 0, 32 );	// node 1 remains unused, for cache line alignment.
		triIdx = (unsigned int*)ALIGNED_MALLOC( primCount * sizeof( unsigned int ) );
		verts = (bvhvec4*)vertices;		// note: we're not copying this data; don't delete.
		fragment = (Fragment*)ALIGNED_MALLOC( primCount * sizeof( Fragment ) );
	}
	else assert( rebuildable == true );
	idxCount = triCount = primCount;
	// reset node pool
	unsigned int newNodePtr = 2;
	// assign all triangles to the root node
	BVHNode& root = bvhNode[0];
	root.leftFirst = 0, root.triCount = triCount, root.aabbMin = bvhvec3( 1e30f ), root.aabbMax = bvhvec3( -1e30f );
	// initialize fragments and initialize root node bounds
	for (unsigned int i = 0; i < triCount; i++)
	{
		fragment[i].bmin = tinybvh_min( tinybvh_min( verts[i * 3], verts[i * 3 + 1] ), verts[i * 3 + 2] );
		fragment[i].bmax = tinybvh_max( tinybvh_max( verts[i * 3], verts[i * 3 + 1] ), verts[i * 3 + 2] );
		root.aabbMin = tinybvh_min( root.aabbMin, fragment[i].bmin );
		root.aabbMax = tinybvh_max( root.aabbMax, fragment[i].bmax ), triIdx[i] = i;
	}
	// subdivide recursively
	unsigned int task[256], taskCount = 0, nodeIdx = 0;
	bvhvec3 minDim = (root.aabbMax - root.aabbMin) * 1e-20f, bestLMin = 0, bestLMax = 0, bestRMin = 0, bestRMax = 0;
	while (1)
	{
		while (1)
		{
			BVHNode& node = bvhNode[nodeIdx];
			// find optimal object split
			bvhvec3 binMin[3][BVHBINS], binMax[3][BVHBINS];
			for (unsigned int a = 0; a < 3; a++) for (unsigned int i = 0; i < BVHBINS; i++) binMin[a][i] = 1e30f, binMax[a][i] = -1e30f;
			unsigned int count[3][BVHBINS];
			memset( count, 0, BVHBINS * 3 * sizeof( unsigned int ) );
			const bvhvec3 rpd3 = bvhvec3( BVHBINS / (node.aabbMax - node.aabbMin) ), nmin3 = node.aabbMin;
			for (unsigned int i = 0; i < node.triCount; i++) // process all tris for x,y and z at once
			{
				const unsigned int fi = triIdx[node.leftFirst + i];
				bvhint3 bi = bvhint3( ((fragment[fi].bmin + fragment[fi].bmax) * 0.5f - nmin3) * rpd3 );
				bi.x = tinybvh_clamp( bi.x, 0, BVHBINS - 1 );
				bi.y = tinybvh_clamp( bi.y, 0, BVHBINS - 1 );
				bi.z = tinybvh_clamp( bi.z, 0, BVHBINS - 1 );
				binMin[0][bi.x] = tinybvh_min( binMin[0][bi.x], fragment[fi].bmin );
				binMax[0][bi.x] = tinybvh_max( binMax[0][bi.x], fragment[fi].bmax ), count[0][bi.x]++;
				binMin[1][bi.y] = tinybvh_min( binMin[1][bi.y], fragment[fi].bmin );
				binMax[1][bi.y] = tinybvh_max( binMax[1][bi.y], fragment[fi].bmax ), count[1][bi.y]++;
				binMin[2][bi.z] = tinybvh_min( binMin[2][bi.z], fragment[fi].bmin );
				binMax[2][bi.z] = tinybvh_max( binMax[2][bi.z], fragment[fi].bmax ), count[2][bi.z]++;
			}
			// calculate per-split totals
			float splitCost = 1e30f;
			unsigned int bestAxis = 0, bestPos = 0;
			for (int a = 0; a < 3; a++) if ((node.aabbMax[a] - node.aabbMin[a]) > minDim[a])
			{
				bvhvec3 lBMin[BVHBINS - 1], rBMin[BVHBINS - 1], l1 = 1e30f, l2 = -1e30f;
				bvhvec3 lBMax[BVHBINS - 1], rBMax[BVHBINS - 1], r1 = 1e30f, r2 = -1e30f;
				float ANL[BVHBINS - 1], ANR[BVHBINS - 1];
				for (unsigned int lN = 0, rN = 0, i = 0; i < BVHBINS - 1; i++)
				{
					lBMin[i] = l1 = tinybvh_min( l1, binMin[a][i] );
					rBMin[BVHBINS - 2 - i] = r1 = tinybvh_min( r1, binMin[a][BVHBINS - 1 - i] );
					lBMax[i] = l2 = tinybvh_max( l2, binMax[a][i] );
					rBMax[BVHBINS - 2 - i] = r2 = tinybvh_max( r2, binMax[a][BVHBINS - 1 - i] );
					lN += count[a][i], rN += count[a][BVHBINS - 1 - i];
					ANL[i] = lN == 0 ? 1e30f : ((l2 - l1).halfArea() * (float)lN);
					ANR[BVHBINS - 2 - i] = rN == 0 ? 1e30f : ((r2 - r1).halfArea() * (float)rN);
				}
				// evaluate bin totals to find best position for object split
				for (unsigned int i = 0; i < BVHBINS - 1; i++)
				{
					const float C = ANL[i] + ANR[i];
					if (C < splitCost)
					{
						splitCost = C, bestAxis = a, bestPos = i;
						bestLMin = lBMin[i], bestRMin = rBMin[i], bestLMax = lBMax[i], bestRMax = rBMax[i];
					}
				}
			}
			if (splitCost >= node.CalculateNodeCost()) break; // not splitting is better.
			// in-place partition
			unsigned int j = node.leftFirst + node.triCount, src = node.leftFirst;
			const float rpd = rpd3.cell[bestAxis], nmin = nmin3.cell[bestAxis];
			for (unsigned int i = 0; i < node.triCount; i++)
			{
				const unsigned int fi = triIdx[src];
				int bi = (unsigned int)(((fragment[fi].bmin[bestAxis] + fragment[fi].bmax[bestAxis]) * 0.5f - nmin) * rpd);
				bi = tinybvh_clamp( bi, 0, BVHBINS - 1 );
				if ((unsigned int)bi <= bestPos) src++; else tinybvh_swap( triIdx[src], triIdx[--j] );
			}
			// create child nodes
			unsigned int leftCount = src - node.leftFirst, rightCount = node.triCount - leftCount;
			if (leftCount == 0 || rightCount == 0) break; // should not happen.
			const int lci = newNodePtr++, rci = newNodePtr++;
			bvhNode[lci].aabbMin = bestLMin, bvhNode[lci].aabbMax = bestLMax;
			bvhNode[lci].leftFirst = node.leftFirst, bvhNode[lci].triCount = leftCount;
			bvhNode[rci].aabbMin = bestRMin, bvhNode[rci].aabbMax = bestRMax;
			bvhNode[rci].leftFirst = j, bvhNode[rci].triCount = rightCount;
			node.leftFirst = lci, node.triCount = 0;
			// recurse
			task[taskCount++] = rci, nodeIdx = lci;
		}
		// fetch subdivision task from stack
		if (taskCount == 0) break; else nodeIdx = task[--taskCount];
	}
	usedBVHNodes = newNodePtr;
}

// SBVH builder.
// Besides the regular object splits used in the reference builder, the SBVH
// algorithm also considers spatial splits, where primitives may be cut in
// multiple parts. This increases primitive count but may reduce overlap of
// BVH nodes. The cost of each option is considered per split. 
// For typical geometry, SBVH yields a tree that can be traversed 25% faster. 
// This comes at greatly increased construction cost, making the SBVH 
// primarily useful for static geometry.
void BVH::BuildHQ( const bvhvec4* vertices, const unsigned int primCount )
{
	// allocate on first build
	const unsigned int slack = primCount >> 2; // for split prims
	const unsigned int spaceNeeded = primCount * 3;
	if (allocatedBVHNodes < spaceNeeded)
	{
		ALIGNED_FREE( bvhNode );
		ALIGNED_FREE( triIdx );
		ALIGNED_FREE( fragment );
		bvhNode = (BVHNode*)ALIGNED_MALLOC( spaceNeeded * sizeof( BVHNode ) );
		allocatedBVHNodes = spaceNeeded;
		memset( &bvhNode[1], 0, 32 );	// node 1 remains unused, for cache line alignment.
		triIdx = (unsigned int*)ALIGNED_MALLOC( (primCount + slack) * sizeof( unsigned int ) );
		verts = (bvhvec4*)vertices;		// note: we're not copying this data; don't delete.
		fragment = (Fragment*)ALIGNED_MALLOC( (primCount + slack) * sizeof( Fragment ) );
	}
	else assert( rebuildable == true );
	idxCount = primCount + slack;
	triCount = primCount;
	unsigned int* triIdxA = triIdx, * triIdxB = new unsigned int[triCount + slack];
	memset( triIdxA, 0, (triCount + slack) * 4 );
	memset( triIdxB, 0, (triCount + slack) * 4 );
	// reset node pool
	unsigned int newNodePtr = 2, nextFrag = triCount;
	// assign all triangles to the root node
	BVHNode& root = bvhNode[0];
	root.leftFirst = 0, root.triCount = triCount, root.aabbMin = bvhvec3( 1e30f ), root.aabbMax = bvhvec3( -1e30f );
	// initialize fragments and initialize root node bounds
	for (unsigned int i = 0; i < triCount; i++)
	{
		fragment[i].bmin = tinybvh_min( tinybvh_min( verts[i * 3], verts[i * 3 + 1] ), verts[i * 3 + 2] );
		fragment[i].bmax = tinybvh_max( tinybvh_max( verts[i * 3], verts[i * 3 + 1] ), verts[i * 3 + 2] );
		root.aabbMin = tinybvh_min( root.aabbMin, fragment[i].bmin );
		root.aabbMax = tinybvh_max( root.aabbMax, fragment[i].bmax ), triIdx[i] = i, fragment[i].primIdx = i;
	}
	const float rootArea = (root.aabbMax - root.aabbMin).halfArea();
	// subdivide recursively
	struct Task { unsigned int node, sliceStart, sliceEnd, dummy; };
	ALIGNED( 64 ) Task task[256];
	unsigned int taskCount = 0, nodeIdx = 0, sliceStart = 0, sliceEnd = triCount + slack;
	const bvhvec3 minDim = (root.aabbMax - root.aabbMin) * 1e-7f /* don't touch, carefully picked */;
	bvhvec3 bestLMin = 0, bestLMax = 0, bestRMin = 0, bestRMax = 0;
	while (1)
	{
		while (1)
		{
			BVHNode& node = bvhNode[nodeIdx];
			// find optimal object split
			bvhvec3 binMin[3][BVHBINS], binMax[3][BVHBINS];
			for (unsigned int a = 0; a < 3; a++) for (unsigned int i = 0; i < BVHBINS; i++) binMin[a][i] = 1e30f, binMax[a][i] = -1e30f;
			unsigned int count[3][BVHBINS];
			memset( count, 0, BVHBINS * 3 * sizeof( unsigned int ) );
			const bvhvec3 rpd3 = bvhvec3( BVHBINS / (node.aabbMax - node.aabbMin) ), nmin3 = node.aabbMin;
			for (unsigned int i = 0; i < node.triCount; i++) // process all tris for x,y and z at once
			{
				const unsigned int fi = triIdx[node.leftFirst + i];
				bvhint3 bi = bvhint3( ((fragment[fi].bmin + fragment[fi].bmax) * 0.5f - nmin3) * rpd3 );
				bi.x = tinybvh_clamp( bi.x, 0, BVHBINS - 1 );
				bi.y = tinybvh_clamp( bi.y, 0, BVHBINS - 1 );
				bi.z = tinybvh_clamp( bi.z, 0, BVHBINS - 1 );
				binMin[0][bi.x] = tinybvh_min( binMin[0][bi.x], fragment[fi].bmin );
				binMax[0][bi.x] = tinybvh_max( binMax[0][bi.x], fragment[fi].bmax ), count[0][bi.x]++;
				binMin[1][bi.y] = tinybvh_min( binMin[1][bi.y], fragment[fi].bmin );
				binMax[1][bi.y] = tinybvh_max( binMax[1][bi.y], fragment[fi].bmax ), count[1][bi.y]++;
				binMin[2][bi.z] = tinybvh_min( binMin[2][bi.z], fragment[fi].bmin );
				binMax[2][bi.z] = tinybvh_max( binMax[2][bi.z], fragment[fi].bmax ), count[2][bi.z]++;
			}
			// calculate per-split totals
			float splitCost = 1e30f;
			unsigned int bestAxis = 0, bestPos = 0;
			for (int a = 0; a < 3; a++) if ((node.aabbMax[a] - node.aabbMin[a]) > minDim.cell[a])
			{
				bvhvec3 lBMin[BVHBINS - 1], rBMin[BVHBINS - 1], l1 = 1e30f, l2 = -1e30f;
				bvhvec3 lBMax[BVHBINS - 1], rBMax[BVHBINS - 1], r1 = 1e30f, r2 = -1e30f;
				float ANL[BVHBINS - 1], ANR[BVHBINS - 1];
				for (unsigned int lN = 0, rN = 0, i = 0; i < BVHBINS - 1; i++)
				{
					lBMin[i] = l1 = tinybvh_min( l1, binMin[a][i] );
					rBMin[BVHBINS - 2 - i] = r1 = tinybvh_min( r1, binMin[a][BVHBINS - 1 - i] );
					lBMax[i] = l2 = tinybvh_max( l2, binMax[a][i] );
					rBMax[BVHBINS - 2 - i] = r2 = tinybvh_max( r2, binMax[a][BVHBINS - 1 - i] );
					lN += count[a][i], rN += count[a][BVHBINS - 1 - i];
					ANL[i] = lN == 0 ? 1e30f : ((l2 - l1).halfArea() * (float)lN);
					ANR[BVHBINS - 2 - i] = rN == 0 ? 1e30f : ((r2 - r1).halfArea() * (float)rN);
				}
				// evaluate bin totals to find best position for object split
				for (unsigned int i = 0; i < BVHBINS - 1; i++)
				{
					const float C = ANL[i] + ANR[i];
					if (C < splitCost)
					{
						splitCost = C, bestAxis = a, bestPos = i;
						bestLMin = lBMin[i], bestRMin = rBMin[i], bestLMax = lBMax[i], bestRMax = rBMax[i];
					}
				}
			}
			// consider a spatial split
			bool spatial = false;
			unsigned int NL[BVHBINS - 1], NR[BVHBINS - 1], budget = sliceEnd - sliceStart;
			bvhvec3 spatialUnion = bestLMax - bestRMin;
			float spatialOverlap = (spatialUnion.halfArea()) / rootArea;
			if (budget > node.triCount && splitCost < 1e30f && spatialOverlap > 1e-5f)
			{
				for (unsigned int a = 0; a < 3; a++) if ((node.aabbMax[a] - node.aabbMin[a]) > minDim.cell[a])
				{
					// setup bins
					bvhvec3 binMin[BVHBINS], binMax[BVHBINS];
					for (unsigned int i = 0; i < BVHBINS; i++) binMin[i] = 1e30f, binMax[i] = -1e30f;
					unsigned int countIn[BVHBINS] = { 0 }, countOut[BVHBINS] = { 0 };
					// populate bins with clipped fragments
					const float planeDist = (node.aabbMax[a] - node.aabbMin[a]) / (BVHBINS * 0.9999f);
					const float rPlaneDist = 1.0f / planeDist, nodeMin = node.aabbMin[a];
					for (unsigned int i = 0; i < node.triCount; i++)
					{
						const unsigned int fragIdx = triIdxA[node.leftFirst + i];
						const int bin1 = tinybvh_clamp( (int)((fragment[fragIdx].bmin[a] - nodeMin) * rPlaneDist), 0, BVHBINS - 1 );
						const int bin2 = tinybvh_clamp( (int)((fragment[fragIdx].bmax[a] - nodeMin) * rPlaneDist), 0, BVHBINS - 1 );
						countIn[bin1]++, countOut[bin2]++;
						if (bin2 == bin1)
						{
							// fragment fits in a single bin
							binMin[bin1] = tinybvh_min( binMin[bin1], fragment[fragIdx].bmin );
							binMax[bin1] = tinybvh_max( binMax[bin1], fragment[fragIdx].bmax );
						}
						else for (int j = bin1; j <= bin2; j++)
						{
							// clip fragment to each bin it overlaps
							bvhvec3 bmin = node.aabbMin, bmax = node.aabbMax;
							bmin[a] = nodeMin + planeDist * j;
							bmax[a] = j == 6 ? node.aabbMax[a] : (bmin[a] + planeDist);
							Fragment orig = fragment[fragIdx];
							Fragment tmpFrag;
							if (!ClipFrag( orig, tmpFrag, bmin, bmax, minDim )) continue;
							binMin[j] = tinybvh_min( binMin[j], tmpFrag.bmin );
							binMax[j] = tinybvh_max( binMax[j], tmpFrag.bmax );
						}
					}
					// evaluate split candidates
					bvhvec3 lBMin[BVHBINS - 1], rBMin[BVHBINS - 1], l1 = 1e30f, l2 = -1e30f;
					bvhvec3 lBMax[BVHBINS - 1], rBMax[BVHBINS - 1], r1 = 1e30f, r2 = -1e30f;
					float ANL[BVHBINS], ANR[BVHBINS];
					for (unsigned int lN = 0, rN = 0, i = 0; i < BVHBINS - 1; i++)
					{
						lBMin[i] = l1 = tinybvh_min( l1, binMin[i] ), rBMin[BVHBINS - 2 - i] = r1 = tinybvh_min( r1, binMin[BVHBINS - 1 - i] );
						lBMax[i] = l2 = tinybvh_max( l2, binMax[i] ), rBMax[BVHBINS - 2 - i] = r2 = tinybvh_max( r2, binMax[BVHBINS - 1 - i] );
						lN += countIn[i], rN += countOut[BVHBINS - 1 - i], NL[i] = lN, NR[BVHBINS - 2 - i] = rN;
						ANL[i] = lN == 0 ? 1e30f : ((l2 - l1).halfArea() * (float)lN);
						ANR[BVHBINS - 2 - i] = rN == 0 ? 1e30f : ((r2 - r1).halfArea() * (float)rN);
					}
					// find best position for spatial split
					for (unsigned int i = 0; i < BVHBINS - 1; i++) if (ANL[i] + ANR[i] < splitCost && NL[i] + NR[i] < budget)
					{
						spatial = true, splitCost = ANL[i] + ANR[i], bestAxis = a, bestPos = i;
						bestLMin = lBMin[i], bestLMax = lBMax[i], bestRMin = rBMin[i], bestRMax = rBMax[i];
						bestLMax[a] = bestRMin[a]; // accurate
					}
				}
			}
			// terminate recursion
			if (splitCost >= node.CalculateNodeCost()) break;
			// double-buffered partition
			unsigned int A = sliceStart, B = sliceEnd, src = node.leftFirst;
			if (spatial)
			{
				const float planeDist = (node.aabbMax[bestAxis] - node.aabbMin[bestAxis]) / (BVHBINS * 0.9999f);
				const float rPlaneDist = 1.0f / planeDist, nodeMin = node.aabbMin[bestAxis];
				for (unsigned int i = 0; i < node.triCount; i++)
				{
					const unsigned int fragIdx = triIdxA[src++];
					const unsigned int bin1 = (unsigned int)((fragment[fragIdx].bmin[bestAxis] - nodeMin) * rPlaneDist);
					const unsigned int bin2 = (unsigned int)((fragment[fragIdx].bmax[bestAxis] - nodeMin) * rPlaneDist);
					if (bin2 <= bestPos) triIdxB[A++] = fragIdx; else if (bin1 > bestPos) triIdxB[--B] = fragIdx; else
					{
						// split straddler
						Fragment tmpFrag = fragment[fragIdx];
						Fragment newFrag;
						if (ClipFrag( tmpFrag, newFrag, tinybvh_max( bestRMin, node.aabbMin ), tinybvh_min( bestRMax, node.aabbMax ), minDim ))
							fragment[nextFrag] = newFrag, triIdxB[--B] = nextFrag++;
						if (ClipFrag( tmpFrag, fragment[fragIdx], tinybvh_max( bestLMin, node.aabbMin ), tinybvh_min( bestLMax, node.aabbMax ), minDim ))
							triIdxB[A++] = fragIdx;
					}
				}
			}
			else
			{
				// object partitioning
				const float rpd = rpd3.cell[bestAxis], nmin = nmin3.cell[bestAxis];
				for (unsigned int i = 0; i < node.triCount; i++)
				{
					const unsigned int fr = triIdx[src + i];
					int bi = (int)(((fragment[fr].bmin[bestAxis] + fragment[fr].bmax[bestAxis]) * 0.5f - nmin) * rpd);
					bi = tinybvh_clamp( bi, 0, BVHBINS - 1 );
					if (bi <= (int)bestPos) triIdxB[A++] = fr; else triIdxB[--B] = fr;
				}
			}
			// copy back slice data
			memcpy( triIdxA + sliceStart, triIdxB + sliceStart, (sliceEnd - sliceStart) * 4 );
			// create child nodes
			unsigned int leftCount = A - sliceStart, rightCount = sliceEnd - B;
			if (leftCount == 0 || rightCount == 0) break;
			int leftChildIdx = newNodePtr++, rightChildIdx = newNodePtr++;
			bvhNode[leftChildIdx].aabbMin = bestLMin, bvhNode[leftChildIdx].aabbMax = bestLMax;
			bvhNode[leftChildIdx].leftFirst = sliceStart, bvhNode[leftChildIdx].triCount = leftCount;
			bvhNode[rightChildIdx].aabbMin = bestRMin, bvhNode[rightChildIdx].aabbMax = bestRMax;
			bvhNode[rightChildIdx].leftFirst = B, bvhNode[rightChildIdx].triCount = rightCount;
			node.leftFirst = leftChildIdx, node.triCount = 0;
			// recurse
			task[taskCount].node = rightChildIdx;
			task[taskCount].sliceEnd = sliceEnd;
			task[taskCount++].sliceStart = sliceEnd = (A + B) >> 1;
			nodeIdx = leftChildIdx;
		}
		// fetch subdivision task from stack
		if (taskCount == 0) break; else
			nodeIdx = task[--taskCount].node,
			sliceStart = task[taskCount].sliceStart,
			sliceEnd = task[taskCount].sliceEnd;
	}
	// clean up
	for (unsigned int i = 0; i < triCount + slack; i++) triIdx[i] = fragment[triIdx[i]].primIdx;
	// Compact(); - TODO
	refittable = false; // can't refit an SBVH
	usedBVHNodes = newNodePtr;
}

bool BVH::ClipFrag( const Fragment& orig, Fragment& newFrag, bvhvec3 bmin, bvhvec3 bmax, bvhvec3 minDim )
{
	// find intersection of bmin/bmax and orig bmin/bmax
	bmin = tinybvh_max( bmin, orig.bmin );
	bmax = tinybvh_min( bmax, orig.bmax );
	const bvhvec3 extent = bmax - bmin;
	// Sutherland-Hodgeman against six bounding planes
	unsigned int Nin = 3, vidx = orig.primIdx * 3;
	bvhvec3 vin[10] = { verts[vidx], verts[vidx + 1], verts[vidx + 2] }, vout[10];
	for (unsigned int a = 0; a < 3; a++)
	{
		const float eps = minDim.cell[a];
		if (extent.cell[a] > eps)
		{
			unsigned int Nout = 0;
			const float l = bmin[a], r = bmax[a];
			for (unsigned int v = 0; v < Nin; v++)
			{
				bvhvec3 v0 = vin[v], v1 = vin[(v + 1) % Nin];
				const bool v0in = v0[a] >= l - eps, v1in = v1[a] >= l - eps;
				if (!(v0in || v1in)) continue; else if (v0in != v1in)
				{
					bvhvec3 C = v0 + (l - v0[a]) / (v1[a] - v0[a]) * (v1 - v0);
					C[a] = l /* accurate */, vout[Nout++] = C;
				}
				if (v1in) vout[Nout++] = v1;
			}
			Nin = 0;
			for (unsigned int v = 0; v < Nout; v++)
			{
				bvhvec3 v0 = vout[v], v1 = vout[(v + 1) % Nout];
				const bool v0in = v0[a] <= r + eps, v1in = v1[a] <= r + eps;
				if (!(v0in || v1in)) continue; else if (v0in != v1in)
				{
					bvhvec3 C = v0 + (r - v0[a]) / (v1[a] - v0[a]) * (v1 - v0);
					C[a] = r /* accurate */, vin[Nin++] = C;
				}
				if (v1in) vin[Nin++] = v1;
			}
		}
	}
	bvhvec3 mn( 1e30f ), mx( -1e30f );
	for (unsigned int i = 0; i < Nin; i++) mn = tinybvh_min( mn, vin[i] ), mx = tinybvh_max( mx, vin[i] );
	newFrag.primIdx = orig.primIdx;
	newFrag.bmin = tinybvh_max( mn, bmin ), newFrag.bmax = tinybvh_min( mx, bmax );
	newFrag.clipped = 1;
	return Nin > 0;
}

void BVH::Convert( BVHLayout from, BVHLayout to, bool deleteOriginal )
{
	if (from == WALD_32BYTE && to == AILA_LAINE)
	{
		// allocate space
		const unsigned int spaceNeeded = usedBVHNodes;
		if (allocatedAltNodes < spaceNeeded)
		{
			ALIGNED_FREE( altNode );
			altNode = (BVHNodeAlt*)ALIGNED_MALLOC( sizeof( BVHNodeAlt ) * spaceNeeded );
			allocatedAltNodes = spaceNeeded;
		}
		memset( altNode, 0, sizeof( BVHNodeAlt ) * spaceNeeded );
		// recursively convert nodes
		unsigned int newAltNode = 0, nodeIdx = 0, stack[128], stackPtr = 0;
		while (1)
		{
			const BVHNode& node = bvhNode[nodeIdx];
			const unsigned int idx = newAltNode++;
			if (node.isLeaf())
			{
				altNode[idx].triCount = node.triCount;
				altNode[idx].firstTri = node.leftFirst;
				if (!stackPtr) break;
				nodeIdx = stack[--stackPtr];
				unsigned int newNodeParent = stack[--stackPtr];
				altNode[newNodeParent].right = newAltNode;
			}
			else
			{
				const BVHNode& left = bvhNode[node.leftFirst];
				const BVHNode& right = bvhNode[node.leftFirst + 1];
				altNode[idx].lmin = left.aabbMin, altNode[idx].rmin = right.aabbMin;
				altNode[idx].lmax = left.aabbMax, altNode[idx].rmax = right.aabbMax;
				altNode[idx].left = newAltNode; // right will be filled when popped
				stack[stackPtr++] = idx;
				stack[stackPtr++] = node.leftFirst + 1;
				nodeIdx = node.leftFirst;
			}
		}
		usedAltNodes = newAltNode;
	}
	else if (from == WALD_32BYTE && to == ALT_SOA)
	{
		// allocate space
		const unsigned int spaceNeeded = usedBVHNodes;
		if (allocatedAlt2Nodes < spaceNeeded)
		{
			ALIGNED_FREE( alt2Node );
			alt2Node = (BVHNodeAlt2*)ALIGNED_MALLOC( sizeof( BVHNodeAlt2 ) * spaceNeeded );
			allocatedAlt2Nodes = spaceNeeded;
		}
		memset( alt2Node, 0, sizeof( BVHNodeAlt2 ) * spaceNeeded );
		// recursively convert nodes
		unsigned int newAlt2Node = 0, nodeIdx = 0, stack[128], stackPtr = 0;
		while (1)
		{
			const BVHNode& node = bvhNode[nodeIdx];
			const unsigned int idx = newAlt2Node++;
			if (node.isLeaf())
			{
				alt2Node[idx].triCount = node.triCount;
				alt2Node[idx].firstTri = node.leftFirst;
				if (!stackPtr) break;
				nodeIdx = stack[--stackPtr];
				unsigned int newNodeParent = stack[--stackPtr];
				alt2Node[newNodeParent].right = newAlt2Node;
			}
			else
			{
				const BVHNode& left = bvhNode[node.leftFirst];
				const BVHNode& right = bvhNode[node.leftFirst + 1];
				// This BVH layout requires BVH_USEAVX for traversal, but at least we
				// can convert to it without SSE/AVX/NEON support.
				alt2Node[idx].xxxx = SIMD_SETRVEC( left.aabbMin.x, left.aabbMax.x, right.aabbMin.x, right.aabbMax.x );
				alt2Node[idx].yyyy = SIMD_SETRVEC( left.aabbMin.y, left.aabbMax.y, right.aabbMin.y, right.aabbMax.y );
				alt2Node[idx].zzzz = SIMD_SETRVEC( left.aabbMin.z, left.aabbMax.z, right.aabbMin.z, right.aabbMax.z );
				alt2Node[idx].left = newAlt2Node; // right will be filled when popped
				stack[stackPtr++] = idx;
				stack[stackPtr++] = node.leftFirst + 1;
				nodeIdx = node.leftFirst;
			}
		}
		usedAlt2Nodes = newAlt2Node;
	}
	else if (from == WALD_32BYTE && to == VERBOSE)
	{
		// allocate space
		unsigned int spaceNeeded = usedBVHNodes;
		if (allocatedVerbose < spaceNeeded)
		{
			ALIGNED_FREE( verbose );
			verbose = (BVHNodeVerbose*)ALIGNED_MALLOC( sizeof( BVHNodeVerbose ) * spaceNeeded );
			allocatedVerbose = spaceNeeded;
		}
		memset( verbose, 0, sizeof( BVHNodeVerbose ) * spaceNeeded );
		verbose[0].parent = 0xffffffff; // root sentinel
		// convert
		unsigned int nodeIdx = 0, parent = 0xffffffff, stack[128], stackPtr = 0;
		while (1)
		{
			const BVHNode& node = bvhNode[nodeIdx];
			verbose[nodeIdx].aabbMin = node.aabbMin, verbose[nodeIdx].aabbMax = node.aabbMax;
			verbose[nodeIdx].triCount = node.triCount, verbose[nodeIdx].parent = parent;
			if (node.isLeaf())
			{
				verbose[nodeIdx].firstTri = node.leftFirst;
				if (stackPtr == 0) break;
				nodeIdx = stack[--stackPtr];
				parent = stack[--stackPtr];
			}
			else
			{
				verbose[nodeIdx].left = node.leftFirst;
				verbose[nodeIdx].right = node.leftFirst + 1;
				stack[stackPtr++] = nodeIdx;
				stack[stackPtr++] = node.leftFirst + 1;
				parent = nodeIdx;
				nodeIdx = node.leftFirst;
			}
		}
		usedVerboseNodes = usedBVHNodes;
	}
	else if (from == WALD_32BYTE && to == BASIC_BVH4)
	{
		// allocate space
		const unsigned int spaceNeeded = usedBVHNodes;
		if (allocatedBVH4Nodes < spaceNeeded)
		{
			ALIGNED_FREE( bvh4Node );
			bvh4Node = (BVHNode4*)ALIGNED_MALLOC( spaceNeeded * sizeof( BVHNode4 ) );
			allocatedBVH4Nodes = spaceNeeded;
		}
		memset( bvh4Node, 0, sizeof( BVHNode4 ) * spaceNeeded );
		// create an mbvh node for each bvh2 node
		for (unsigned int i = 0; i < usedBVHNodes; i++) if (i != 1)
		{
			BVHNode& orig = bvhNode[i];
			BVHNode4& node4 = bvh4Node[i];
			node4.aabbMin = orig.aabbMin, node4.aabbMax = orig.aabbMax;
			if (orig.isLeaf()) node4.triCount = orig.triCount, node4.firstTri = orig.leftFirst;
			else node4.child[0] = orig.leftFirst, node4.child[1] = orig.leftFirst + 1, node4.childCount = 2;
		}
		// collapse
		unsigned int stack[128], stackPtr = 1, nodeIdx = stack[0] = 0; // i.e., root node
		while (1)
		{
			BVHNode4& node = bvh4Node[nodeIdx];
			while (node.childCount < 4)
			{
				int bestChild = -1;
				float bestChildSA = 0;
				for (unsigned int i = 0; i < node.childCount; i++)
				{
					// see if we can adopt child i
					const BVHNode4& child = bvh4Node[node.child[i]];
					if (!child.isLeaf() && node.childCount - 1 + child.childCount <= 4)
					{
						const float childSA = SA( child.aabbMin, child.aabbMax );
						if (childSA > bestChildSA) bestChild = i, bestChildSA = childSA;
					}
				}
				if (bestChild == -1) break; // could not adopt
				const BVHNode4& child = bvh4Node[node.child[bestChild]];
				node.child[bestChild] = child.child[0];
				for (unsigned int i = 1; i < child.childCount; i++)
					node.child[node.childCount++] = child.child[i];
			}
			// we're done with the node; proceed with the children
			for (unsigned int i = 0; i < node.childCount; i++)
			{
				const unsigned int childIdx = node.child[i];
				const BVHNode4& child = bvh4Node[childIdx];
				if (!child.isLeaf()) stack[stackPtr++] = childIdx;
			}
			if (stackPtr == 0) break;
			nodeIdx = stack[--stackPtr];
		}
		usedBVH4Nodes = usedBVHNodes; // there will be gaps / unused nodes though.
	}
	else if (from == BASIC_BVH4 && to == BVH4_GPU)
	{
		// Convert a 4-wide BVH to a format suitable for GPU traversal. Layout:
		// offs 0:   aabbMin (12 bytes), 4x quantized child xmin (4 bytes)
		// offs 16:  aabbMax (12 bytes), 4x quantized child xmax (4 bytes)
		// offs 32:  4x child ymin, then ymax, zmax, zmax (total 16 bytes)
		// offs 48:  4x child node info: leaf if MSB set.
		//           Leaf: 15 bits for tri count, 16 for offset
		//           Interior: 32 bits for position of child node.
		// Triangle data ('by value') immediately follows each leaf node.
		unsigned int blocksNeeded = usedBVHNodes * 4; // here, 'block' is 16 bytes.
		blocksNeeded += 6 * triCount; // this layout stores tris in the same buffer.
		if (allocatedAlt4Blocks < blocksNeeded)
		{
			ALIGNED_FREE( bvh4Alt );
			assert( sizeof( BVHNode4Alt ) == 64 );
			assert( bvh4Node != 0 );
			bvh4Alt = (bvhvec4*)ALIGNED_MALLOC( blocksNeeded * 16 );
			allocatedAlt4Blocks = blocksNeeded;
		}
		memset( bvh4Alt, 0, 16 * blocksNeeded );
		// start conversion
	#ifdef __GNUC__
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wstrict-aliasing"
	#endif
		unsigned int nodeIdx = 0, newAlt4Ptr = 0, stack[128], stackPtr = 0, retValPos = 0;
		while (1)
		{
			const BVHNode4& node = bvh4Node[nodeIdx];
			// convert BVH4 node - must be an interior node.
			assert( !bvh4Node[nodeIdx].isLeaf() );
			bvhvec4* nodeBase = bvh4Alt + newAlt4Ptr;
			unsigned int baseAlt4Ptr = newAlt4Ptr;
			newAlt4Ptr += 4;
			nodeBase[0] = bvhvec4( node.aabbMin, 0 );
			nodeBase[1] = bvhvec4( (node.aabbMax - node.aabbMin) * (1.0f / 255.0f), 0 );
			BVHNode4* childNode[4] = {
				&bvh4Node[node.child[0]], &bvh4Node[node.child[1]],
				&bvh4Node[node.child[2]], &bvh4Node[node.child[3]]
			};
			// start with leaf child node conversion
			unsigned int childInfo[4] = { 0, 0, 0, 0 }; // will store in final fields later
			for (int i = 0; i < 4; i++) if (childNode[i]->isLeaf())
			{
				childInfo[i] = newAlt4Ptr - baseAlt4Ptr;
				childInfo[i] |= childNode[i]->triCount << 16;
				childInfo[i] |= 0x80000000;
				for (unsigned int j = 0; j < childNode[i]->triCount; j++)
				{
					unsigned int t = triIdx[childNode[i]->firstTri + j];
					bvhvec4 v0 = verts[t * 3 + 0];
					v0.w = *(float*)&t; // as_float
					bvh4Alt[newAlt4Ptr++] = v0;
					bvh4Alt[newAlt4Ptr++] = verts[t * 3 + 1];
					bvh4Alt[newAlt4Ptr++] = verts[t * 3 + 2];
				}
			}
			// process interior nodes
			for (int i = 0; i < 4; i++) if (!childNode[i]->isLeaf())
			{
				// childInfo[i] = node.child[i] == 0 ? 0 : GPUFormatBVH4( node.child[i] );
				if (node.child[i] == 0) childInfo[i] = 0; else
				{
					stack[stackPtr++] = (unsigned)(((float*)&nodeBase[3] + i) - (float*)bvh4Alt);
					stack[stackPtr++] = node.child[i];
				}
			}
			// store child node bounds, quantized
			const bvhvec3 extent = node.aabbMax - node.aabbMin;
			bvhvec3 scale;
			scale.x = extent.x > 1e-10f ? (254.999f / extent.x) : 0;
			scale.y = extent.y > 1e-10f ? (254.999f / extent.y) : 0;
			scale.z = extent.z > 1e-10f ? (254.999f / extent.z) : 0;
			unsigned char* slot0 = (unsigned char*)&nodeBase[0] + 12;	// 4 chars
			unsigned char* slot1 = (unsigned char*)&nodeBase[1] + 12;	// 4 chars
			unsigned char* slot2 = (unsigned char*)&nodeBase[2];		// 16 chars
			if (node.child[0])
			{
				const bvhvec3 relBMin = childNode[0]->aabbMin - node.aabbMin, relBMax = childNode[0]->aabbMax - node.aabbMin;
				slot0[0] = (unsigned char)floorf( relBMin.x * scale.x ), slot1[0] = (unsigned char)ceilf( relBMax.x * scale.x );
				slot2[0] = (unsigned char)floorf( relBMin.y * scale.y ), slot2[4] = (unsigned char)ceilf( relBMax.y * scale.y );
				slot2[8] = (unsigned char)floorf( relBMin.z * scale.z ), slot2[12] = (unsigned char)ceilf( relBMax.z * scale.z );
			}
			if (node.child[1])
			{
				const bvhvec3 relBMin = childNode[1]->aabbMin - node.aabbMin, relBMax = childNode[1]->aabbMax - node.aabbMin;
				slot0[1] = (unsigned char)floorf( relBMin.x * scale.x ), slot1[1] = (unsigned char)ceilf( relBMax.x * scale.x );
				slot2[1] = (unsigned char)floorf( relBMin.y * scale.y ), slot2[5] = (unsigned char)ceilf( relBMax.y * scale.y );
				slot2[9] = (unsigned char)floorf( relBMin.z * scale.z ), slot2[13] = (unsigned char)ceilf( relBMax.z * scale.z );
			}
			if (node.child[2])
			{
				const bvhvec3 relBMin = childNode[2]->aabbMin - node.aabbMin, relBMax = childNode[2]->aabbMax - node.aabbMin;
				slot0[2] = (unsigned char)floorf( relBMin.x * scale.x ), slot1[2] = (unsigned char)ceilf( relBMax.x * scale.x );
				slot2[2] = (unsigned char)floorf( relBMin.y * scale.y ), slot2[6] = (unsigned char)ceilf( relBMax.y * scale.y );
				slot2[10] = (unsigned char)floorf( relBMin.z * scale.z ), slot2[14] = (unsigned char)ceilf( relBMax.z * scale.z );
			}
			if (node.child[3])
			{
				const bvhvec3 relBMin = childNode[3]->aabbMin - node.aabbMin, relBMax = childNode[3]->aabbMax - node.aabbMin;
				slot0[3] = (unsigned char)floorf( relBMin.x * scale.x ), slot1[3] = (unsigned char)ceilf( relBMax.x * scale.x );
				slot2[3] = (unsigned char)floorf( relBMin.y * scale.y ), slot2[7] = (unsigned char)ceilf( relBMax.y * scale.y );
				slot2[11] = (unsigned char)floorf( relBMin.z * scale.z ), slot2[15] = (unsigned char)ceilf( relBMax.z * scale.z );
			}
			// finalize node
			nodeBase[3] = bvhvec4(
				*(float*)&childInfo[0], *(float*)&childInfo[1],
				*(float*)&childInfo[2], *(float*)&childInfo[3]
			);
			// pop new work from the stack
			if (retValPos > 0) ((unsigned int*)bvh4Alt)[retValPos] = baseAlt4Ptr;
			if (stackPtr == 0) break;
			nodeIdx = stack[--stackPtr];
			retValPos = stack[--stackPtr];
		}
	#ifdef __GNUC__
	#pragma GCC diagnostic pop
	#endif
		usedAlt4Blocks = newAlt4Ptr;
	}
	else if (from == WALD_32BYTE && to == BASIC_BVH8)
	{
		// allocate space
		const unsigned int spaceNeeded = usedBVHNodes;
		if (allocatedBVH8Nodes < spaceNeeded)
		{
			ALIGNED_FREE( bvh8Node );
			bvh8Node = (BVHNode8*)ALIGNED_MALLOC( spaceNeeded * sizeof( BVHNode8 ) );
			allocatedBVH8Nodes = spaceNeeded;
		}
		memset( bvh8Node, 0, sizeof( BVHNode4 ) * spaceNeeded );
		// create an mbvh node for each bvh2 node
		for (unsigned int i = 0; i < usedBVHNodes; i++) if (i != 1)
		{
			BVHNode& orig = bvhNode[i];
			BVHNode8& node8 = bvh8Node[i];
			node8.aabbMin = orig.aabbMin, node8.aabbMax = orig.aabbMax;
			if (orig.isLeaf()) node8.triCount = orig.triCount, node8.firstTri = orig.leftFirst;
			else node8.child[0] = orig.leftFirst, node8.child[1] = orig.leftFirst + 1, node8.childCount = 2;
		}
		// collapse
		unsigned int stack[128], stackPtr = 1, nodeIdx = stack[0] = 0; // i.e., root node
		while (1)
		{
			BVHNode8& node = bvh8Node[nodeIdx];
			while (node.childCount < 8)
			{
				int bestChild = -1;
				float bestChildSA = 0;
				for (unsigned int i = 0; i < node.childCount; i++)
				{
					// see if we can adopt child i
					const BVHNode8& child = bvh8Node[node.child[i]];
					if ((!child.isLeaf()) && (node.childCount - 1 + child.childCount) <= 8)
					{
						const float childSA = SA( child.aabbMin, child.aabbMax );
						if (childSA > bestChildSA) bestChild = i, bestChildSA = childSA;
					}
				}
				if (bestChild == -1) break; // could not adopt
				const BVHNode8& child = bvh8Node[node.child[bestChild]];
				node.child[bestChild] = child.child[0];
				for (unsigned int i = 1; i < child.childCount; i++)
					node.child[node.childCount++] = child.child[i];
			}
			// we're done with the node; proceed with the children
			for (unsigned int i = 0; i < node.childCount; i++)
			{
				const unsigned int childIdx = node.child[i];
				const BVHNode8& child = bvh8Node[childIdx];
				if (!child.isLeaf()) stack[stackPtr++] = childIdx;
			}
			if (stackPtr == 0) break;
			nodeIdx = stack[--stackPtr];
		}
		usedBVH8Nodes = usedBVHNodes; // there will be gaps / unused nodes though.
	}
	else if (from == VERBOSE && to == WALD_32BYTE)
	{
		// allocate space
		const unsigned int spaceNeeded = usedVerboseNodes;
		if (allocatedBVHNodes < spaceNeeded)
		{
			ALIGNED_FREE( bvhNode );
			bvhNode = (BVHNode*)ALIGNED_MALLOC( triCount * 2 * sizeof( BVHNode ) );
			allocatedBVHNodes = spaceNeeded;
		}
		memset( bvhNode, 0, sizeof( BVHNode ) * spaceNeeded );
		// start conversion
		unsigned int srcNodeIdx = 0, dstNodeIdx = 0, newNodePtr = 2;
		unsigned int srcStack[64], dstStack[64], stackPtr = 0;
		while (1)
		{
			const BVHNodeVerbose& srcNode = verbose[srcNodeIdx];
			bvhNode[dstNodeIdx].aabbMin = srcNode.aabbMin;
			bvhNode[dstNodeIdx].aabbMax = srcNode.aabbMax;
			if (srcNode.isLeaf())
			{
				bvhNode[dstNodeIdx].triCount = srcNode.triCount;
				bvhNode[dstNodeIdx].leftFirst = srcNode.firstTri;
				if (stackPtr == 0) break;
				srcNodeIdx = srcStack[--stackPtr];
				dstNodeIdx = dstStack[stackPtr];
			}
			else
			{
				bvhNode[dstNodeIdx].leftFirst = newNodePtr;
				unsigned int srcRightIdx = srcNode.right;
				srcNodeIdx = srcNode.left, dstNodeIdx = newNodePtr++;
				srcStack[stackPtr] = srcRightIdx;
				dstStack[stackPtr++] = newNodePtr++;
			}
		}
		usedBVHNodes = usedVerboseNodes;
	}
	else
	{
		// For now all other conversions are invalid.
		assert( false );
	}
	rebuildable = false; // hard to guarantee safe rebuilds after a layout conversion.
}

// Refitting: For animated meshes, where the topology remains intact. This
// includes trees waving in the wind, or subsequent frames for skinned
// animations. Repeated refitting tends to lead to deteriorated BVHs and
// slower ray tracing. Rebuild when this happens.
void BVH::Refit()
{
	for (int i = usedBVHNodes - 1; i >= 0; i--)
	{
		BVHNode& node = bvhNode[i];
		if (node.isLeaf()) // leaf: adjust to current triangle vertex positions
		{
			bvhvec4 aabbMin( 1e30f ), aabbMax( -1e30f );
			for (unsigned int first = node.leftFirst, j = 0; j < node.triCount; j++)
			{
				const unsigned int vertIdx = triIdx[first + j] * 3;
				aabbMin = tinybvh_min( aabbMin, verts[vertIdx] ), aabbMax = tinybvh_max( aabbMax, verts[vertIdx] );
				aabbMin = tinybvh_min( aabbMin, verts[vertIdx + 1] ), aabbMax = tinybvh_max( aabbMax, verts[vertIdx + 1] );
				aabbMin = tinybvh_min( aabbMin, verts[vertIdx + 2] ), aabbMax = tinybvh_max( aabbMax, verts[vertIdx + 2] );
			}
			node.aabbMin = aabbMin, node.aabbMax = aabbMax;
			continue;
		}
		// interior node: adjust to child bounds
		const BVHNode& left = bvhNode[node.leftFirst], & right = bvhNode[node.leftFirst + 1];
		node.aabbMin = tinybvh_min( left.aabbMin, right.aabbMin );
		node.aabbMax = tinybvh_max( left.aabbMax, right.aabbMax );
	}
}

//  Optimizing a BVH: BVH must be in 'verbose' format.
//  Implements "Fast Insertion-Based Optimization of Bounding Volume Hierarchies",
void BVH::Optimize()
{
	// optimize by reinserting a random subtree - call repeatedly for best results.
	unsigned int Nid, valid = 0;
	do
	{
		static unsigned int seed = 0x12345678;
		seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5; // xor32
		valid = 1, Nid = 2 + seed % (usedVerboseNodes - 2);
		if (verbose[Nid].parent == 0 || verbose[Nid].isLeaf()) valid = 0;
		if (valid) if (verbose[verbose[Nid].parent].parent == 0) valid = 0;
	} while (valid == 0);
	// snip it loose
	const BVHNodeVerbose& N = verbose[Nid];
	const BVHNodeVerbose& P = verbose[N.parent];
	const unsigned int Pid = N.parent, X1 = P.parent;
	const unsigned int X2 = P.left == Nid ? P.right : P.left;
	if (verbose[X1].left == Pid) verbose[X1].left = X2;
	else /* verbose[X1].right == Pid */ verbose[X1].right = X2;
	verbose[X2].parent = X1;
	unsigned int L = N.left, R = N.right;
	// fix affected node bounds
	RefitUpVerbose( X1 );
	ReinsertNodeVerbose( L, Pid, X1 );
	ReinsertNodeVerbose( R, Nid, X1 );
}

// RefitUpVerbose: Update bounding boxes of ancestors of the specified node.
void BVH::RefitUpVerbose( unsigned int nodeIdx )
{
	while (nodeIdx != 0xffffffff)
	{
		BVHNodeVerbose& node = verbose[nodeIdx];
		BVHNodeVerbose& left = verbose[node.left];
		BVHNodeVerbose& right = verbose[node.right];
		node.aabbMin = tinybvh_min( left.aabbMin, right.aabbMin );
		node.aabbMax = tinybvh_max( left.aabbMax, right.aabbMax );
		nodeIdx = node.parent;
	}
}

// FindBestNewPosition
// Part of "Fast Insertion-Based Optimization of Bounding Volume Hierarchies"
unsigned int BVH::FindBestNewPosition( const unsigned int Lid )
{
	BVHNodeVerbose& L = verbose[Lid];
	float SA_L = SA( L.aabbMin, L.aabbMax );
	// reinsert L into BVH
	unsigned int taskNode[512], tasks = 1, Xbest = 0;
	float taskCi[512], taskInvCi[512], Cbest = 1e30f, epsilon = 1e-10f;
	taskNode[0] = 0 /* root */, taskCi[0] = 0, taskInvCi[0] = 1 / epsilon;
	while (tasks > 0)
	{
		// 'pop' task with createst taskInvCi
		float maxInvCi = 0;
		unsigned int bestTask = 0;
		for (unsigned int j = 0; j < tasks; j++) if (taskInvCi[j] > maxInvCi) maxInvCi = taskInvCi[j], bestTask = j;
		unsigned int Xid = taskNode[bestTask];
		float CiLX = taskCi[bestTask];
		taskNode[bestTask] = taskNode[--tasks], taskCi[bestTask] = taskCi[tasks], taskInvCi[bestTask] = taskInvCi[tasks];
		// execute task
		BVHNodeVerbose& X = verbose[Xid];
		if (CiLX + SA_L >= Cbest) break;
		float CdLX = SA( tinybvh_min( L.aabbMin, X.aabbMin ), tinybvh_max( L.aabbMax, X.aabbMax ) );
		float CLX = CiLX + CdLX;
		if (CLX < Cbest) Cbest = CLX, Xbest = Xid;
		float Ci = CLX - SA( X.aabbMin, X.aabbMax );
		if (Ci + SA_L < Cbest) if (!X.isLeaf())
		{
			taskNode[tasks] = X.left, taskCi[tasks] = Ci, taskInvCi[tasks++] = 1.0f / (Ci + epsilon);
			taskNode[tasks] = X.right, taskCi[tasks] = Ci, taskInvCi[tasks++] = 1.0f / (Ci + epsilon);
		}
	}
	return Xbest;
}

// ReinsertNodeVerbose
// Part of "Fast Insertion-Based Optimization of Bounding Volume Hierarchies"
void BVH::ReinsertNodeVerbose( const unsigned int Lid, const unsigned int Nid, const unsigned int origin )
{
	unsigned int Xbest = FindBestNewPosition( Lid );
	if (verbose[Xbest].parent == 0) Xbest = origin;
	const unsigned int X1 = verbose[Xbest].parent;
	BVHNodeVerbose& N = verbose[Nid];
	N.left = Xbest, N.right = Lid;
	N.aabbMin = tinybvh_min( verbose[Xbest].aabbMin, verbose[Lid].aabbMin );
	N.aabbMax = tinybvh_max( verbose[Xbest].aabbMax, verbose[Lid].aabbMax );
	verbose[Nid].parent = X1;
	if (verbose[X1].left == Xbest) verbose[X1].left = Nid; else verbose[X1].right = Nid;
	verbose[Xbest].parent = verbose[Lid].parent = Nid;
	RefitUpVerbose( Nid );
}

// Intersect a BVH with a ray.
// This function returns the intersection details in Ray::hit. Additionally,
// the number of steps through the BVH is returned. Visualize this to get a
// visual impression of the structure of the BVH.
int BVH::Intersect( Ray& ray, BVHLayout layout ) const
{
	switch (layout)
	{
	case WALD_32BYTE:
		return Intersect_Wald32Byte( ray );
		break;
	case AILA_LAINE:
		return Intersect_AilaLaine( ray );
		break;
	case ALT_SOA:
		return Intersect_AltSoA( ray );
		break;
	case BASIC_BVH4:
		return Intersect_BasicBVH4( ray );
		break;
	case BASIC_BVH8:
		return Intersect_BasicBVH8( ray );
		break;
	default:
		assert( false );
	};
	return 0;
}

// Traverse the default BVH layout (WALD_32BYTE).
int BVH::Intersect_Wald32Byte( Ray& ray ) const
{
	assert( bvhNode != 0 );
	BVHNode* node = &bvhNode[0], * stack[64];
	unsigned int stackPtr = 0, steps = 0;
	while (1)
	{
		steps++;
		if (node->isLeaf())
		{
			for (unsigned int i = 0; i < node->triCount; i++) IntersectTri( ray, triIdx[node->leftFirst + i] );
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		float dist1 = child1->Intersect( ray ), dist2 = child2->Intersect( ray );
		if (dist1 > dist2) { tinybvh_swap( dist1, dist2 ); tinybvh_swap( child1, child2 ); }
		if (dist1 == 1e30f /* missed both child nodes */)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else /* hit at least one node */
		{
			node = child1; /* continue with the nearest */
			if (dist2 != 1e30f) stack[stackPtr++] = child2; /* push far child */
		}
	}
	return steps;
}

// Traverse the alternative BVH layout (AILA_LAINE).
int BVH::Intersect_AilaLaine( Ray& ray ) const
{
	assert( altNode != 0 );
	BVHNodeAlt* node = &altNode[0], * stack[64];
	unsigned int stackPtr = 0, steps = 0;
	while (1)
	{
		steps++;
		if (node->isLeaf())
		{
			for (unsigned int i = 0; i < node->triCount; i++) IntersectTri( ray, triIdx[node->firstTri + i] );
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		const bvhvec3 lmin = node->lmin - ray.O, lmax = node->lmax - ray.O;
		const bvhvec3 rmin = node->rmin - ray.O, rmax = node->rmax - ray.O;
		float dist1 = 1e30f, dist2 = 1e30f;
		const bvhvec3 t1a = lmin * ray.rD, t2a = lmax * ray.rD;
		const bvhvec3 t1b = rmin * ray.rD, t2b = rmax * ray.rD;
		const float tmina = tinybvh_max( tinybvh_max( tinybvh_min( t1a.x, t2a.x ), tinybvh_min( t1a.y, t2a.y ) ), tinybvh_min( t1a.z, t2a.z ) );
		const float tmaxa = tinybvh_min( tinybvh_min( tinybvh_max( t1a.x, t2a.x ), tinybvh_max( t1a.y, t2a.y ) ), tinybvh_max( t1a.z, t2a.z ) );
		const float tminb = tinybvh_max( tinybvh_max( tinybvh_min( t1b.x, t2b.x ), tinybvh_min( t1b.y, t2b.y ) ), tinybvh_min( t1b.z, t2b.z ) );
		const float tmaxb = tinybvh_min( tinybvh_min( tinybvh_max( t1b.x, t2b.x ), tinybvh_max( t1b.y, t2b.y ) ), tinybvh_max( t1b.z, t2b.z ) );
		if (tmaxa >= tmina && tmina < ray.hit.t && tmaxa >= 0) dist1 = tmina;
		if (tmaxb >= tminb && tminb < ray.hit.t && tmaxb >= 0) dist2 = tminb;
		unsigned int lidx = node->left, ridx = node->right;
		if (dist1 > dist2)
		{
			float t = dist1; dist1 = dist2; dist2 = t;
			unsigned int i = lidx; lidx = ridx; ridx = i;
		}
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = altNode + lidx;
			if (dist2 != 1e30f) stack[stackPtr++] = altNode + ridx;
		}
	}
	return steps;
}

//  Intersect4. For testing the converted data only; not efficient.
int BVH::Intersect_BasicBVH4( Ray& ray ) const
{
	BVHNode4* node = &bvh4Node[0], * stack[64];
	unsigned int stackPtr = 0, steps = 0;
	while (1)
	{
		steps++;
		if (node->isLeaf()) for (unsigned int i = 0; i < node->triCount; i++)
			IntersectTri( ray, triIdx[node->firstTri + i] );
		else for (unsigned int i = 0; i < node->childCount; i++)
		{
			BVHNode4* child = bvh4Node + node->child[i];
			float dist = IntersectAABB( ray, child->aabbMin, child->aabbMax );
			if (dist < 1e30f) stack[stackPtr++] = child;
		}
		if (stackPtr == 0) break; else node = stack[--stackPtr];
	}
	return steps;
}

//  Intersect4. For testing the converted data only; not efficient.
int BVH::Intersect_BasicBVH8( Ray& ray ) const
{
	BVHNode8* node = &bvh8Node[0], * stack[128];
	unsigned int stackPtr = 0, steps = 0;
	while (1)
	{
		steps++;
		if (node->isLeaf()) for (unsigned int i = 0; i < node->triCount; i++)
			IntersectTri( ray, triIdx[node->firstTri + i] );
		else for (unsigned int i = 0; i < node->childCount; i++)
		{
			BVHNode8* child = bvh8Node + node->child[i];
			float dist = IntersectAABB( ray, child->aabbMin, child->aabbMax );
			if (dist < 1e30f) stack[stackPtr++] = child;
		}
		if (stackPtr == 0) break; else node = stack[--stackPtr];
	}
	return steps;
}

// Intersect a WALD_32BYTE BVH with a ray packet.
// The 256 rays travel together to better utilize the caches and to amortize the cost 
// of memory transfers over the rays in the bundle.
// Note that this basic implementation assumes a specific layout of the rays. Provided
// as 'proof of concept', should not be used in production code.
// Based on Large Ray Packets for Real-time Whitted Ray Tracing, Overbeck et al., 2008,
// extended with sorted traversal and reduced stack traffic.
void BVH::Intersect256Rays( Ray* packet ) const
{
	// convenience macro
#define CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( r ) const bvhvec3 rD = packet[r].rD, t1 = o1 * rD, t2 = o2 * rD; \
	const float tmin = tinybvh_max( tinybvh_max( tinybvh_min( t1.x, t2.x ), tinybvh_min( t1.y, t2.y ) ), tinybvh_min( t1.z, t2.z ) ); \
	const float tmax = tinybvh_min( tinybvh_min( tinybvh_max( t1.x, t2.x ), tinybvh_max( t1.y, t2.y ) ), tinybvh_max( t1.z, t2.z ) );
	// Corner rays are: 0, 51, 204 and 255
	// Construct the bounding planes, with normals pointing outwards
	const bvhvec3 O = packet[0].O; // same for all rays in this case
	const bvhvec3 p0 = packet[0].O + packet[0].D; // top-left
	const bvhvec3 p1 = packet[51].O + packet[51].D; // top-right
	const bvhvec3 p2 = packet[204].O + packet[204].D; // bottom-left
	const bvhvec3 p3 = packet[255].O + packet[255].D; // bottom-right
	const bvhvec3 plane0 = normalize( cross( p0 - O, p0 - p2 ) ); // left plane
	const bvhvec3 plane1 = normalize( cross( p3 - O, p3 - p1 ) ); // right plane
	const bvhvec3 plane2 = normalize( cross( p1 - O, p1 - p0 ) ); // top plane
	const bvhvec3 plane3 = normalize( cross( p2 - O, p2 - p3 ) ); // bottom plane
	const int sign0x = plane0.x < 0 ? 4 : 0, sign0y = plane0.y < 0 ? 5 : 1, sign0z = plane0.z < 0 ? 6 : 2;
	const int sign1x = plane1.x < 0 ? 4 : 0, sign1y = plane1.y < 0 ? 5 : 1, sign1z = plane1.z < 0 ? 6 : 2;
	const int sign2x = plane2.x < 0 ? 4 : 0, sign2y = plane2.y < 0 ? 5 : 1, sign2z = plane2.z < 0 ? 6 : 2;
	const int sign3x = plane3.x < 0 ? 4 : 0, sign3y = plane3.y < 0 ? 5 : 1, sign3z = plane3.z < 0 ? 6 : 2;
	const float d0 = dot( O, plane0 ), d1 = dot( O, plane1 );
	const float d2 = dot( O, plane2 ), d3 = dot( O, plane3 );
	// Traverse the tree with the packet
	int first = 0, last = 255; // first and last active ray in the packet
	const BVHNode* node = &bvhNode[0];
	ALIGNED( 64 ) unsigned int stack[64], stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			// handle leaf node
			for (unsigned int j = 0; j < node->triCount; j++)
			{
				const unsigned int idx = triIdx[node->leftFirst + j], vid = idx * 3;
				const bvhvec3 edge1 = verts[vid + 1] - verts[vid], edge2 = verts[vid + 2] - verts[vid];
				const bvhvec3 s = O - bvhvec3( verts[vid] );
				for (int i = first; i <= last; i++)
				{
					Ray& ray = packet[i];
					const bvhvec3 h = cross( ray.D, edge2 );
					const float a = dot( edge1, h );
					if (fabs( a ) < 0.0000001f) continue; // ray parallel to triangle
					const float f = 1 / a, u = f * dot( s, h );
					if (u < 0 || u > 1) continue;
					const bvhvec3 q = cross( s, edge1 );
					const float v = f * dot( ray.D, q );
					if (v < 0 || u + v > 1) continue;
					const float t = f * dot( edge2, q );
					if (t <= 0 || t >= ray.hit.t) continue;
					ray.hit.t = t, ray.hit.u = u, ray.hit.v = v, ray.hit.prim = idx;
				}
			}
			if (stackPtr == 0) break; else // pop
				last = stack[--stackPtr], node = bvhNode + stack[--stackPtr],
				first = last >> 8, last &= 255;
		}
		else
		{
			// fetch pointers to child nodes
			const BVHNode* left = bvhNode + node->leftFirst;
			const BVHNode* right = bvhNode + node->leftFirst + 1;
			bool visitLeft = true, visitRight = true;
			int leftFirst = first, leftLast = last, rightFirst = first, rightLast = last;
			float distLeft, distRight;
			{
				// see if we want to intersect the left child
				const bvhvec3 o1( left->aabbMin.x - O.x, left->aabbMin.y - O.y, left->aabbMin.z - O.z );
				const bvhvec3 o2( left->aabbMax.x - O.x, left->aabbMax.y - O.y, left->aabbMax.z - O.z );
				// 1. Early-in test: if first ray hits the node, the packet visits the node
				CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( first );
				const bool earlyHit = (tmax >= tmin && tmin < packet[first].hit.t && tmax >= 0);
				distLeft = tmin;
				// 2. Early-out test: if the node aabb is outside the four planes, we skip the node
				if (!earlyHit)
				{
					float* minmax = (float*)left;
					bvhvec3 p0( minmax[sign0x], minmax[sign0y], minmax[sign0z] );
					bvhvec3 p1( minmax[sign1x], minmax[sign1y], minmax[sign1z] );
					bvhvec3 p2( minmax[sign2x], minmax[sign2y], minmax[sign2z] );
					bvhvec3 p3( minmax[sign3x], minmax[sign3y], minmax[sign3z] );
					if (dot( p0, plane0 ) > d0 || dot( p1, plane1 ) > d1 || dot( p2, plane2 ) > d2 || dot( p3, plane3 ) > d3)
						visitLeft = false;
					else
					{
						// 3. Last resort: update first and last, stay in node if first > last
						for (; leftFirst <= leftLast; leftFirst++)
						{
							CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( leftFirst );
							if (tmax >= tmin && tmin < packet[leftFirst].hit.t && tmax >= 0) { distLeft = tmin; break; }
						}
						for (; leftLast >= leftFirst; leftLast--)
						{
							CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( leftLast );
							if (tmax >= tmin && tmin < packet[leftLast].hit.t && tmax >= 0) break;
						}
						visitLeft = leftLast >= leftFirst;
					}
				}
			}
			{
				// see if we want to intersect the right child
				const bvhvec3 o1( right->aabbMin.x - O.x, right->aabbMin.y - O.y, right->aabbMin.z - O.z );
				const bvhvec3 o2( right->aabbMax.x - O.x, right->aabbMax.y - O.y, right->aabbMax.z - O.z );
				// 1. Early-in test: if first ray hits the node, the packet visits the node
				CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( first );
				const bool earlyHit = (tmax >= tmin && tmin < packet[first].hit.t && tmax >= 0);
				distRight = tmin;
				// 2. Early-out test: if the node aabb is outside the four planes, we skip the node
				if (!earlyHit)
				{
					float* minmax = (float*)right;
					bvhvec3 p0( minmax[sign0x], minmax[sign0y], minmax[sign0z] );
					bvhvec3 p1( minmax[sign1x], minmax[sign1y], minmax[sign1z] );
					bvhvec3 p2( minmax[sign2x], minmax[sign2y], minmax[sign2z] );
					bvhvec3 p3( minmax[sign3x], minmax[sign3y], minmax[sign3z] );
					if (dot( p0, plane0 ) > d0 || dot( p1, plane1 ) > d1 || dot( p2, plane2 ) > d2 || dot( p3, plane3 ) > d3)
						visitRight = false;
					else
					{
						// 3. Last resort: update first and last, stay in node if first > last
						for (; rightFirst <= rightLast; rightFirst++)
						{
							CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( rightFirst );
							if (tmax >= tmin && tmin < packet[rightFirst].hit.t && tmax >= 0) { distRight = tmin; break; }
						}
						for (; rightLast >= first; rightLast--)
						{
							CALC_TMIN_TMAX_WITH_SLABTEST_ON_RAY( rightLast );
							if (tmax >= tmin && tmin < packet[rightLast].hit.t && tmax >= 0) break;
						}
						visitRight = rightLast >= rightFirst;
					}
				}
			}
			// process intersection result
			if (visitLeft && visitRight)
			{
				if (distLeft < distRight)
				{
					// push right, continue with left
					stack[stackPtr++] = node->leftFirst + 1;
					stack[stackPtr++] = (rightFirst << 8) + rightLast;
					node = left, first = leftFirst, last = leftLast;
				}
				else
				{
					// push left, continue with right
					stack[stackPtr++] = node->leftFirst;
					stack[stackPtr++] = (leftFirst << 8) + leftLast;
					node = right, first = rightFirst, last = rightLast;
				}
			}
			else if (visitLeft) // continue with left
				node = left, first = leftFirst, last = leftLast;
			else if (visitRight) // continue with right
				node = right, first = rightFirst, last = rightLast;
			else if (stackPtr == 0) break; else // pop
				last = stack[--stackPtr], node = bvhNode + stack[--stackPtr],
				first = last >> 8, last &= 255;
		}
	}
}

// IntersectTri
void BVH::IntersectTri( Ray& ray, const unsigned int idx ) const
{
	// Moeller-Trumbore ray/triangle intersection algorithm
	const unsigned int vertIdx = idx * 3;
	const bvhvec3 edge1 = verts[vertIdx + 1] - verts[vertIdx];
	const bvhvec3 edge2 = verts[vertIdx + 2] - verts[vertIdx];
	const bvhvec3 h = cross( ray.D, edge2 );
	const float a = dot( edge1, h );
	if (fabs( a ) < 0.0000001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const bvhvec3 s = ray.O - bvhvec3( verts[vertIdx] );
	const float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	const bvhvec3 q = cross( s, edge1 );
	const float v = f * dot( ray.D, q );
	if (v < 0 || u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0 && t < ray.hit.t)
	{
		// register a hit: ray is shortened to t
		ray.hit.t = t, ray.hit.u = u, ray.hit.v = v, ray.hit.prim = idx;
	}
}

// IntersectAABB
float BVH::IntersectAABB( const Ray& ray, const bvhvec3& aabbMin, const bvhvec3& aabbMax )
{
	// "slab test" ray/AABB intersection
	float tx1 = (aabbMin.x - ray.O.x) * ray.rD.x, tx2 = (aabbMax.x - ray.O.x) * ray.rD.x;
	float tmin = tinybvh_min( tx1, tx2 ), tmax = tinybvh_max( tx1, tx2 );
	float ty1 = (aabbMin.y - ray.O.y) * ray.rD.y, ty2 = (aabbMax.y - ray.O.y) * ray.rD.y;
	tmin = tinybvh_max( tmin, tinybvh_min( ty1, ty2 ) );
	tmax = tinybvh_min( tmax, tinybvh_max( ty1, ty2 ) );
	float tz1 = (aabbMin.z - ray.O.z) * ray.rD.z, tz2 = (aabbMax.z - ray.O.z) * ray.rD.z;
	tmin = tinybvh_max( tmin, tinybvh_min( tz1, tz2 ) );
	tmax = tinybvh_min( tmax, tinybvh_max( tz1, tz2 ) );
	if (tmax >= tmin && tmin < ray.hit.t && tmax >= 0) return tmin; else return 1e30f;
}

// ============================================================================
//
//        I M P L E M E N T A T I O N  -  S I M D  C O D E
// 
// ============================================================================

#ifdef BVH_USEAVX

// Ultra-fast single-threaded AVX binned-SAH-builder.
// This code produces BVHs nearly identical to reference, but much faster.
// On a 12th gen laptop i7 CPU, Sponza Crytek (~260k tris) is processed in 51ms.
// The code relies on the availability of AVX instructions. AVX2 is not needed.
#ifdef _MSC_VER
#define LANE(a,b) a.m128_f32[b]
#define LANE8(a,b) a.m256_f32[b]
// Not using clang/g++ method under MSCC; compiler may benefit from .m128_i32.
#define ILANE(a,b) a.m128i_i32[b]
#else
#define LANE(a,b) a[b]
#define LANE8(a,b) a[b]
// Below method reduces to a single instruction.
#define ILANE(a,b) _mm_cvtsi128_si32(_mm_castps_si128( _mm_shuffle_ps(_mm_castsi128_ps( a ), _mm_castsi128_ps( a ), b)))
#endif
inline float halfArea( const __m128 a /* a contains extent of aabb */ )
{
	return LANE( a, 0 ) * LANE( a, 1 ) + LANE( a, 1 ) * LANE( a, 2 ) + LANE( a, 2 ) * LANE( a, 3 );
}
inline float halfArea( const __m256& a /* a contains aabb itself, with min.xyz negated */ )
{
#ifndef _MSC_VER
	// g++ doesn't seem to like the faster construct
	float* c = (float*)&a;
	float ex = c[4] + c[0], ey = c[5] + c[1], ez = c[6] + c[2];
	return ex * ey + ey * ez + ez * ex;
#else
	const __m128 q = _mm256_castps256_ps128( _mm256_add_ps( _mm256_permute2f128_ps( a, a, 5 ), a ) );
	const __m128 v = _mm_mul_ps( q, _mm_shuffle_ps( q, q, 9 ) );
	return LANE( v, 0 ) + LANE( v, 1 ) + LANE( v, 2 );
#endif
}
#define PROCESS_PLANE( a, pos, ANLR, lN, rN, lb, rb ) if (lN * rN != 0) { \
	ANLR = halfArea( lb ) * (float)lN + halfArea( rb ) * (float)rN; if (ANLR < splitCost) \
	splitCost = ANLR, bestAxis = a, bestPos = pos, bestLBox = lb, bestRBox = rb; }
#if defined(_MSC_VER)
#pragma warning ( push )
#pragma warning( disable:4701 ) // "potentially uninitialized local variable 'bestLBox' used"
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
void BVH::BuildAVX( const bvhvec4* vertices, const unsigned int primCount )
{
	int test = BVHBINS;
	if (test != 8) assert( false ); // AVX builders require BVHBINS == 8.
	// aligned data
	ALIGNED( 64 ) __m256 binbox[3 * BVHBINS];				// 768 bytes
	ALIGNED( 64 ) __m256 binboxOrig[3 * BVHBINS];			// 768 bytes
	ALIGNED( 64 ) unsigned int count[3][BVHBINS]{};			// 96 bytes
	ALIGNED( 64 ) __m256 bestLBox, bestRBox;				// 64 bytes
	// some constants
	static const __m128 max4 = _mm_set1_ps( -1e30f ), half4 = _mm_set1_ps( 0.5f );
	static const __m128 two4 = _mm_set1_ps( 2.0f ), min1 = _mm_set1_ps( -1 );
	static const __m256 max8 = _mm256_set1_ps( -1e30f );
	static const __m256 signFlip8 = _mm256_setr_ps( -0.0f, -0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
	static const __m128 signFlip4 = _mm_setr_ps( -0.0f, -0.0f, -0.0f, 0.0f );
	static const __m128 mask3 = _mm_cmpeq_ps( _mm_setr_ps( 0, 0, 0, 1 ), _mm_setzero_ps() );
	static const __m128 binmul3 = _mm_set1_ps( BVHBINS * 0.49999f );
	for (unsigned int i = 0; i < 3 * BVHBINS; i++) binboxOrig[i] = max8; // binbox initialization template
	// reset node pool
	const unsigned int spaceNeeded = primCount * 2;
	if (allocatedBVHNodes < spaceNeeded)
	{
		ALIGNED_FREE( bvhNode );
		ALIGNED_FREE( triIdx );
		ALIGNED_FREE( fragment );
		verts = (bvhvec4*)vertices;
		triIdx = (unsigned int*)ALIGNED_MALLOC( primCount * sizeof( unsigned int ) );
		bvhNode = (BVHNode*)ALIGNED_MALLOC( spaceNeeded * sizeof( BVHNode ) );
		allocatedBVHNodes = spaceNeeded;
		memset( &bvhNode[1], 0, 32 ); // avoid crash in refit.
		fragment = (Fragment*)ALIGNED_MALLOC( primCount * sizeof( Fragment ) );
	}
	else assert( rebuildable == true );
	triCount = idxCount = primCount;
	unsigned int newNodePtr = 2;
	struct FragSSE { __m128 bmin4, bmax4; };
	FragSSE* frag4 = (FragSSE*)fragment;
	__m256* frag8 = (__m256*)fragment;
	const __m128* verts4 = (__m128*)verts;
	// assign all triangles to the root node
	BVHNode& root = bvhNode[0];
	root.leftFirst = 0, root.triCount = triCount;
	// initialize fragments and update root bounds
	__m128 rootMin = max4, rootMax = max4;
	for (unsigned int i = 0; i < triCount; i++)
	{
		const __m128 v1 = _mm_xor_ps( signFlip4, _mm_min_ps( _mm_min_ps( verts4[i * 3], verts4[i * 3 + 1] ), verts4[i * 3 + 2] ) );
		const __m128 v2 = _mm_max_ps( _mm_max_ps( verts4[i * 3], verts4[i * 3 + 1] ), verts4[i * 3 + 2] );
		frag4[i].bmin4 = v1, frag4[i].bmax4 = v2, rootMin = _mm_max_ps( rootMin, v1 ), rootMax = _mm_max_ps( rootMax, v2 ), triIdx[i] = i;
	}
	rootMin = _mm_xor_ps( rootMin, signFlip4 );
	root.aabbMin = *(bvhvec3*)&rootMin, root.aabbMax = *(bvhvec3*)&rootMax;
	// subdivide recursively
	ALIGNED( 64 ) unsigned int task[128], taskCount = 0, nodeIdx = 0;
	const bvhvec3 minDim = (root.aabbMax - root.aabbMin) * 1e-10f;
	while (1)
	{
		while (1)
		{
			BVHNode& node = bvhNode[nodeIdx];
			__m128* node4 = (__m128*) & bvhNode[nodeIdx];
			// find optimal object split
			const __m128 d4 = _mm_blendv_ps( min1, _mm_sub_ps( node4[1], node4[0] ), mask3 );
			const __m128 nmin4 = _mm_mul_ps( _mm_and_ps( node4[0], mask3 ), two4 );
			const __m128 rpd4 = _mm_and_ps( _mm_div_ps( binmul3, d4 ), _mm_cmpneq_ps( d4, _mm_setzero_ps() ) );
			// implementation of Section 4.1 of "Parallel Spatial Splits in Bounding Volume Hierarchies":
			// main loop operates on two fragments to minimize dependencies and maximize ILP.
			unsigned int fi = triIdx[node.leftFirst];
			memset( count, 0, sizeof( count ) );
			__m256 r0, r1, r2, f = frag8[fi];
			__m128i bi4 = _mm_cvtps_epi32( _mm_sub_ps( _mm_mul_ps( _mm_sub_ps( _mm_sub_ps( frag4[fi].bmax4, frag4[fi].bmin4 ), nmin4 ), rpd4 ), half4 ) );
			memcpy( binbox, binboxOrig, sizeof( binbox ) );
			unsigned int i0 = ILANE( bi4, 0 ), i1 = ILANE( bi4, 1 ), i2 = ILANE( bi4, 2 ), * ti = triIdx + node.leftFirst + 1;
			for (unsigned int i = 0; i < node.triCount - 1; i++)
			{
				unsigned int fid = *ti++;
				const __m256 b0 = binbox[i0], b1 = binbox[BVHBINS + i1], b2 = binbox[2 * BVHBINS + i2];
				const __m128 fmin = frag4[fid].bmin4, fmax = frag4[fid].bmax4;
				r0 = _mm256_max_ps( b0, f ), r1 = _mm256_max_ps( b1, f ), r2 = _mm256_max_ps( b2, f );
				const __m128i b4 = _mm_cvtps_epi32( _mm_sub_ps( _mm_mul_ps( _mm_sub_ps( _mm_sub_ps( fmax, fmin ), nmin4 ), rpd4 ), half4 ) );
			#if _MSC_VER < 1920 
				// VS2017 needs a random check on fid to produce correct code... The condition is never true. 
				if (fid > triCount) fid = triCount - 1;
			#endif
				f = frag8[fid], count[0][i0]++, count[1][i1]++, count[2][i2]++;
				binbox[i0] = r0, i0 = ILANE( b4, 0 );
				binbox[BVHBINS + i1] = r1, i1 = ILANE( b4, 1 );
				binbox[2 * BVHBINS + i2] = r2, i2 = ILANE( b4, 2 );
			}
			// final business for final fragment
			const __m256 b0 = binbox[i0], b1 = binbox[BVHBINS + i1], b2 = binbox[2 * BVHBINS + i2];
			count[0][i0]++, count[1][i1]++, count[2][i2]++;
			r0 = _mm256_max_ps( b0, f ), r1 = _mm256_max_ps( b1, f ), r2 = _mm256_max_ps( b2, f );
			binbox[i0] = r0, binbox[BVHBINS + i1] = r1, binbox[2 * BVHBINS + i2] = r2;
			// calculate per-split totals
			float splitCost = 1e30f;
			unsigned int bestAxis = 0, bestPos = 0, n = newNodePtr, j = node.leftFirst + node.triCount, src = node.leftFirst;
			const __m256* bb = binbox;
			for (int a = 0; a < 3; a++, bb += BVHBINS) if ((node.aabbMax[a] - node.aabbMin[a]) > minDim.cell[a])
			{
				// hardcoded bin processing for BVHBINS == 8
				assert( BVHBINS == 8 );
				const unsigned int lN0 = count[a][0], rN0 = count[a][7];
				const __m256 lb0 = bb[0], rb0 = bb[7];
				const unsigned int lN1 = lN0 + count[a][1], rN1 = rN0 + count[a][6], lN2 = lN1 + count[a][2];
				const unsigned int rN2 = rN1 + count[a][5], lN3 = lN2 + count[a][3], rN3 = rN2 + count[a][4];
				const __m256 lb1 = _mm256_max_ps( lb0, bb[1] ), rb1 = _mm256_max_ps( rb0, bb[6] );
				const __m256 lb2 = _mm256_max_ps( lb1, bb[2] ), rb2 = _mm256_max_ps( rb1, bb[5] );
				const __m256 lb3 = _mm256_max_ps( lb2, bb[3] ), rb3 = _mm256_max_ps( rb2, bb[4] );
				const unsigned int lN4 = lN3 + count[a][4], rN4 = rN3 + count[a][3], lN5 = lN4 + count[a][5];
				const unsigned int rN5 = rN4 + count[a][2], lN6 = lN5 + count[a][6], rN6 = rN5 + count[a][1];
				const __m256 lb4 = _mm256_max_ps( lb3, bb[4] ), rb4 = _mm256_max_ps( rb3, bb[3] );
				const __m256 lb5 = _mm256_max_ps( lb4, bb[5] ), rb5 = _mm256_max_ps( rb4, bb[2] );
				const __m256 lb6 = _mm256_max_ps( lb5, bb[6] ), rb6 = _mm256_max_ps( rb5, bb[1] );
				float ANLR3 = 1e30f; PROCESS_PLANE( a, 3, ANLR3, lN3, rN3, lb3, rb3 ); // most likely split
				float ANLR2 = 1e30f; PROCESS_PLANE( a, 2, ANLR2, lN2, rN4, lb2, rb4 );
				float ANLR4 = 1e30f; PROCESS_PLANE( a, 4, ANLR4, lN4, rN2, lb4, rb2 );
				float ANLR5 = 1e30f; PROCESS_PLANE( a, 5, ANLR5, lN5, rN1, lb5, rb1 );
				float ANLR1 = 1e30f; PROCESS_PLANE( a, 1, ANLR1, lN1, rN5, lb1, rb5 );
				float ANLR0 = 1e30f; PROCESS_PLANE( a, 0, ANLR0, lN0, rN6, lb0, rb6 );
				float ANLR6 = 1e30f; PROCESS_PLANE( a, 6, ANLR6, lN6, rN0, lb6, rb0 ); // least likely split
			}
			if (splitCost >= node.CalculateNodeCost()) break; // not splitting is better.
			// in-place partition
			const float rpd = (*(bvhvec3*)&rpd4)[bestAxis], nmin = (*(bvhvec3*)&nmin4)[bestAxis];
			unsigned int t, fr = triIdx[src];
			for (unsigned int i = 0; i < node.triCount; i++)
			{
				const unsigned int bi = (unsigned int)((fragment[fr].bmax[bestAxis] - fragment[fr].bmin[bestAxis] - nmin) * rpd);
				if (bi <= bestPos) fr = triIdx[++src]; else t = fr, fr = triIdx[src] = triIdx[--j], triIdx[j] = t;
			}
			// create child nodes and recurse
			const unsigned int leftCount = src - node.leftFirst, rightCount = node.triCount - leftCount;
			if (leftCount == 0 || rightCount == 0) break; // should not happen.
			*(__m256*)& bvhNode[n] = _mm256_xor_ps( bestLBox, signFlip8 );
			bvhNode[n].leftFirst = node.leftFirst, bvhNode[n].triCount = leftCount;
			node.leftFirst = n++, node.triCount = 0, newNodePtr += 2;
			*(__m256*)& bvhNode[n] = _mm256_xor_ps( bestRBox, signFlip8 );
			bvhNode[n].leftFirst = j, bvhNode[n].triCount = rightCount;
			task[taskCount++] = n, nodeIdx = n - 1;
		}
		// fetch subdivision task from stack
		if (taskCount == 0) break; else nodeIdx = task[--taskCount];
	}
	usedBVHNodes = newNodePtr;
}
#if defined(_MSC_VER)
#pragma warning ( pop ) // restore 4701
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop // restore -Wmaybe-uninitialized
#endif

// Intersect a BVH with a ray packet, basic SSE-optimized version.
// Note: This yields +10% on 10th gen Intel CPUs, but a small loss on
// more recent hardware. This function needs a full conversion to work
// with groups of 8 rays at a time - TODO.
void BVH::Intersect256RaysSSE( Ray* packet ) const
{
	// Corner rays are: 0, 51, 204 and 255
	// Construct the bounding planes, with normals pointing outwards
	bvhvec3 O = packet[0].O; // same for all rays in this case
	__m128 O4 = *(__m128*) & packet[0].O;
	__m128 mask4 = _mm_cmpeq_ps( _mm_setzero_ps(), _mm_set_ps( 1, 0, 0, 0 ) );
	bvhvec3 p0 = packet[0].O + packet[0].D; // top-left
	bvhvec3 p1 = packet[51].O + packet[51].D; // top-right
	bvhvec3 p2 = packet[204].O + packet[204].D; // bottom-left
	bvhvec3 p3 = packet[255].O + packet[255].D; // bottom-right
	bvhvec3 plane0 = normalize( cross( p0 - O, p0 - p2 ) ); // left plane
	bvhvec3 plane1 = normalize( cross( p3 - O, p3 - p1 ) ); // right plane
	bvhvec3 plane2 = normalize( cross( p1 - O, p1 - p0 ) ); // top plane
	bvhvec3 plane3 = normalize( cross( p2 - O, p2 - p3 ) ); // bottom plane
	int sign0x = plane0.x < 0 ? 4 : 0, sign0y = plane0.y < 0 ? 5 : 1, sign0z = plane0.z < 0 ? 6 : 2;
	int sign1x = plane1.x < 0 ? 4 : 0, sign1y = plane1.y < 0 ? 5 : 1, sign1z = plane1.z < 0 ? 6 : 2;
	int sign2x = plane2.x < 0 ? 4 : 0, sign2y = plane2.y < 0 ? 5 : 1, sign2z = plane2.z < 0 ? 6 : 2;
	int sign3x = plane3.x < 0 ? 4 : 0, sign3y = plane3.y < 0 ? 5 : 1, sign3z = plane3.z < 0 ? 6 : 2;
	float t0 = dot( O, plane0 ), t1 = dot( O, plane1 );
	float t2 = dot( O, plane2 ), t3 = dot( O, plane3 );
	// Traverse the tree with the packet
	int first = 0, last = 255; // first and last active ray in the packet
	BVHNode* node = &bvhNode[0];
	ALIGNED( 64 ) unsigned int stack[64], stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			// handle leaf node
			for (unsigned int j = 0; j < node->triCount; j++)
			{
				const unsigned int idx = triIdx[node->leftFirst + j], vid = idx * 3;
				const bvhvec3 edge1 = verts[vid + 1] - verts[vid], edge2 = verts[vid + 2] - verts[vid];
				const bvhvec3 s = O - bvhvec3( verts[vid] );
				for (int i = first; i <= last; i++)
				{
					Ray& ray = packet[i];
					const bvhvec3 h = cross( ray.D, edge2 );
					const float a = dot( edge1, h );
					if (fabs( a ) < 0.0000001f) continue; // ray parallel to triangle
					const float f = 1 / a, u = f * dot( s, h );
					if (u < 0 || u > 1) continue;
					const bvhvec3 q = cross( s, edge1 );
					const float v = f * dot( ray.D, q );
					if (v < 0 || u + v > 1) continue;
					const float t = f * dot( edge2, q );
					if (t <= 0 || t >= ray.hit.t) continue;
					ray.hit.t = t, ray.hit.u = u, ray.hit.v = v, ray.hit.prim = idx;
				}
			}
			if (stackPtr == 0) break; else // pop
				last = stack[--stackPtr], node = bvhNode + stack[--stackPtr],
				first = last >> 8, last &= 255;
		}
		else
		{
			// fetch pointers to child nodes
			BVHNode* left = bvhNode + node->leftFirst;
			BVHNode* right = bvhNode + node->leftFirst + 1;
			bool visitLeft = true, visitRight = true;
			int leftFirst = first, leftLast = last, rightFirst = first, rightLast = last;
			float distLeft, distRight;
			{
				// see if we want to intersect the left child
				const __m128 minO4 = _mm_sub_ps( *(__m128*) & left->aabbMin, O4 );
				const __m128 maxO4 = _mm_sub_ps( *(__m128*) & left->aabbMax, O4 );
				// 1. Early-in test: if first ray hits the node, the packet visits the node
				const __m128 rD4 = *(__m128*) & packet[first].rD;
				const __m128 st1 = _mm_mul_ps( _mm_and_ps( minO4, mask4 ), rD4 );
				const __m128 st2 = _mm_mul_ps( _mm_and_ps( maxO4, mask4 ), rD4 );
				const __m128 vmax4 = _mm_max_ps( st1, st2 ), vmin4 = _mm_min_ps( st1, st2 );
				const float tmax = tinybvh_min( LANE( vmax4, 0 ), tinybvh_min( LANE( vmax4, 1 ), LANE( vmax4, 2 ) ) );
				const float tmin = tinybvh_max( LANE( vmin4, 0 ), tinybvh_max( LANE( vmin4, 1 ), LANE( vmin4, 2 ) ) );
				const bool earlyHit = (tmax >= tmin && tmin < packet[first].hit.t && tmax >= 0);
				distLeft = tmin;
				// 2. Early-out test: if the node aabb is outside the four planes, we skip the node
				if (!earlyHit)
				{
					float* minmax = (float*)left;
					bvhvec3 p0( minmax[sign0x], minmax[sign0y], minmax[sign0z] );
					bvhvec3 p1( minmax[sign1x], minmax[sign1y], minmax[sign1z] );
					bvhvec3 p2( minmax[sign2x], minmax[sign2y], minmax[sign2z] );
					bvhvec3 p3( minmax[sign3x], minmax[sign3y], minmax[sign3z] );
					if (dot( p0, plane0 ) > t0 || dot( p1, plane1 ) > t1 || dot( p2, plane2 ) > t2 || dot( p3, plane3 ) > t3)
						visitLeft = false;
					else
					{
						// 3. Last resort: update first and last, stay in node if first > last
						for (; leftFirst <= leftLast; leftFirst++)
						{
							const __m128 rD4 = *(__m128*) & packet[leftFirst].rD;
							const __m128 st1 = _mm_mul_ps( _mm_and_ps( minO4, mask4 ), rD4 );
							const __m128 st2 = _mm_mul_ps( _mm_and_ps( maxO4, mask4 ), rD4 );
							const __m128 vmax4 = _mm_max_ps( st1, st2 ), vmin4 = _mm_min_ps( st1, st2 );
							const float tmax = tinybvh_min( LANE( vmax4, 0 ), tinybvh_min( LANE( vmax4, 1 ), LANE( vmax4, 2 ) ) );
							const float tmin = tinybvh_max( LANE( vmin4, 0 ), tinybvh_max( LANE( vmin4, 1 ), LANE( vmin4, 2 ) ) );
							if (tmax >= tmin && tmin < packet[leftFirst].hit.t && tmax >= 0) { distLeft = tmin; break; }
						}
						for (; leftLast >= leftFirst; leftLast--)
						{
							const __m128 rD4 = *(__m128*) & packet[leftLast].rD;
							const __m128 st1 = _mm_mul_ps( _mm_and_ps( minO4, mask4 ), rD4 );
							const __m128 st2 = _mm_mul_ps( _mm_and_ps( maxO4, mask4 ), rD4 );
							const __m128 vmax4 = _mm_max_ps( st1, st2 ), vmin4 = _mm_min_ps( st1, st2 );
							const float tmax = tinybvh_min( LANE( vmax4, 0 ), tinybvh_min( LANE( vmax4, 1 ), LANE( vmax4, 2 ) ) );
							const float tmin = tinybvh_max( LANE( vmin4, 0 ), tinybvh_max( LANE( vmin4, 1 ), LANE( vmin4, 2 ) ) );
							if (tmax >= tmin && tmin < packet[leftLast].hit.t && tmax >= 0) break;
						}
						visitLeft = leftLast >= leftFirst;
					}
				}
			}
			{
				// see if we want to intersect the right child
				const __m128 minO4 = _mm_sub_ps( *(__m128*) & right->aabbMin, O4 );
				const __m128 maxO4 = _mm_sub_ps( *(__m128*) & right->aabbMax, O4 );
				// 1. Early-in test: if first ray hits the node, the packet visits the node
				const __m128 rD4 = *(__m128*) & packet[first].rD;
				const __m128 st1 = _mm_mul_ps( minO4, rD4 ), st2 = _mm_mul_ps( maxO4, rD4 );
				const __m128 vmax4 = _mm_max_ps( st1, st2 ), vmin4 = _mm_min_ps( st1, st2 );
				const float tmax = tinybvh_min( LANE( vmax4, 0 ), tinybvh_min( LANE( vmax4, 1 ), LANE( vmax4, 2 ) ) );
				const float tmin = tinybvh_max( LANE( vmin4, 0 ), tinybvh_max( LANE( vmin4, 1 ), LANE( vmin4, 2 ) ) );
				const bool earlyHit = (tmax >= tmin && tmin < packet[first].hit.t && tmax >= 0);
				distRight = tmin;
				// 2. Early-out test: if the node aabb is outside the four planes, we skip the node
				if (!earlyHit)
				{
					float* minmax = (float*)right;
					bvhvec3 p0( minmax[sign0x], minmax[sign0y], minmax[sign0z] );
					bvhvec3 p1( minmax[sign1x], minmax[sign1y], minmax[sign1z] );
					bvhvec3 p2( minmax[sign2x], minmax[sign2y], minmax[sign2z] );
					bvhvec3 p3( minmax[sign3x], minmax[sign3y], minmax[sign3z] );
					if (dot( p0, plane0 ) > t0 || dot( p1, plane1 ) > t1 || dot( p2, plane2 ) > t2 || dot( p3, plane3 ) > t3)
						visitRight = false;
					else
					{
						// 3. Last resort: update first and last, stay in node if first > last
						for (; rightFirst <= rightLast; rightFirst++)
						{
							const __m128 rD4 = *(__m128*) & packet[rightFirst].rD;
							const __m128 st1 = _mm_mul_ps( _mm_and_ps( minO4, mask4 ), rD4 );
							const __m128 st2 = _mm_mul_ps( _mm_and_ps( maxO4, mask4 ), rD4 );
							const __m128 vmax4 = _mm_max_ps( st1, st2 ), vmin4 = _mm_min_ps( st1, st2 );
							const float tmax = tinybvh_min( LANE( vmax4, 0 ), tinybvh_min( LANE( vmax4, 1 ), LANE( vmax4, 2 ) ) );
							const float tmin = tinybvh_max( LANE( vmin4, 0 ), tinybvh_max( LANE( vmin4, 1 ), LANE( vmin4, 2 ) ) );
							if (tmax >= tmin && tmin < packet[rightFirst].hit.t && tmax >= 0) { distRight = tmin; break; }
						}
						for (; rightLast >= first; rightLast--)
						{
							const __m128 rD4 = *(__m128*) & packet[rightLast].rD;
							const __m128 st1 = _mm_mul_ps( _mm_and_ps( minO4, mask4 ), rD4 );
							const __m128 st2 = _mm_mul_ps( _mm_and_ps( maxO4, mask4 ), rD4 );
							const __m128 vmax4 = _mm_max_ps( st1, st2 ), vmin4 = _mm_min_ps( st1, st2 );
							const float tmax = tinybvh_min( LANE( vmax4, 0 ), tinybvh_min( LANE( vmax4, 1 ), LANE( vmax4, 2 ) ) );
							const float tmin = tinybvh_max( LANE( vmin4, 0 ), tinybvh_max( LANE( vmin4, 1 ), LANE( vmin4, 2 ) ) );
							if (tmax >= tmin && tmin < packet[rightLast].hit.t && tmax >= 0) break;
						}
						visitRight = rightLast >= rightFirst;
					}
				}
			}
			// process intersection result
			if (visitLeft && visitRight)
			{
				if (distLeft < distRight)
				{
					// push right, continue with left
					stack[stackPtr++] = node->leftFirst + 1;
					stack[stackPtr++] = (rightFirst << 8) + rightLast;
					node = left, first = leftFirst, last = leftLast;
				}
				else
				{
					// push left, continue with right
					stack[stackPtr++] = node->leftFirst;
					stack[stackPtr++] = (leftFirst << 8) + leftLast;
					node = right, first = rightFirst, last = rightLast;
				}
			}
			else if (visitLeft) // continue with left
				node = left, first = leftFirst, last = leftLast;
			else if (visitRight) // continue with right
				node = right, first = rightFirst, last = rightLast;
			else if (stackPtr == 0) break; else // pop
				last = stack[--stackPtr], node = bvhNode + stack[--stackPtr],
				first = last >> 8, last &= 255;
		}
	}
}

// Traverse the second alternative BVH layout (ALT_SOA).
int BVH::Intersect_AltSoA( Ray& ray ) const
{
	assert( alt2Node != 0 );
	BVHNodeAlt2* node = &alt2Node[0], * stack[64];
	unsigned int stackPtr = 0, steps = 0;
	const __m128 Ox4 = _mm_set1_ps( ray.O.x ), rDx4 = _mm_set1_ps( ray.rD.x );
	const __m128 Oy4 = _mm_set1_ps( ray.O.y ), rDy4 = _mm_set1_ps( ray.rD.y );
	const __m128 Oz4 = _mm_set1_ps( ray.O.z ), rDz4 = _mm_set1_ps( ray.rD.z );
	// const __m128 inf4 = _mm_set1_ps( 1e30f );
	while (1)
	{
		steps++;
		if (node->isLeaf())
		{
			for (unsigned int i = 0; i < node->triCount; i++)
			{
				const unsigned int tidx = triIdx[node->firstTri + i], vertIdx = tidx * 3;
				const bvhvec3 edge1 = verts[vertIdx + 1] - verts[vertIdx];
				const bvhvec3 edge2 = verts[vertIdx + 2] - verts[vertIdx];
				const bvhvec3 h = cross( ray.D, edge2 );
				const float a = dot( edge1, h );
				if (fabs( a ) < 0.0000001f) continue; // ray parallel to triangle
				const float f = 1 / a;
				const bvhvec3 s = ray.O - bvhvec3( verts[vertIdx] );
				const float u = f * dot( s, h );
				if (u < 0 || u > 1) continue;
				const bvhvec3 q = cross( s, edge1 );
				const float v = f * dot( ray.D, q );
				if (v < 0 || u + v > 1) continue;
				const float t = f * dot( edge2, q );
				if (t < 0 || t > ray.hit.t) continue;
				ray.hit.t = t, ray.hit.u = u, ray.hit.v = v, ray.hit.prim = tidx;
			}
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		__m128 x4 = _mm_mul_ps( _mm_sub_ps( node->xxxx, Ox4 ), rDx4 );
		__m128 y4 = _mm_mul_ps( _mm_sub_ps( node->yyyy, Oy4 ), rDy4 );
		__m128 z4 = _mm_mul_ps( _mm_sub_ps( node->zzzz, Oz4 ), rDz4 );
		// transpose
		__m128 t0 = _mm_unpacklo_ps( x4, y4 ), t2 = _mm_unpacklo_ps( z4, z4 );
		__m128 t1 = _mm_unpackhi_ps( x4, y4 ), t3 = _mm_unpackhi_ps( z4, z4 );
		__m128 xyzw1a = _mm_shuffle_ps( t0, t2, _MM_SHUFFLE( 1, 0, 1, 0 ) );
		__m128 xyzw2a = _mm_shuffle_ps( t0, t2, _MM_SHUFFLE( 3, 2, 3, 2 ) );
		__m128 xyzw1b = _mm_shuffle_ps( t1, t3, _MM_SHUFFLE( 1, 0, 1, 0 ) );
		__m128 xyzw2b = _mm_shuffle_ps( t1, t3, _MM_SHUFFLE( 3, 2, 3, 2 ) );
		// process
		__m128 tmina4 = _mm_min_ps( xyzw1a, xyzw2a ), tmaxa4 = _mm_max_ps( xyzw1a, xyzw2a );
		__m128 tminb4 = _mm_min_ps( xyzw1b, xyzw2b ), tmaxb4 = _mm_max_ps( xyzw1b, xyzw2b );
		// transpose back
		t0 = _mm_unpacklo_ps( tmina4, tmaxa4 ), t2 = _mm_unpacklo_ps( tminb4, tmaxb4 );
		t1 = _mm_unpackhi_ps( tmina4, tmaxa4 ), t3 = _mm_unpackhi_ps( tminb4, tmaxb4 );
		x4 = _mm_shuffle_ps( t0, t2, _MM_SHUFFLE( 1, 0, 1, 0 ) );
		y4 = _mm_shuffle_ps( t0, t2, _MM_SHUFFLE( 3, 2, 3, 2 ) );
		z4 = _mm_shuffle_ps( t1, t3, _MM_SHUFFLE( 1, 0, 1, 0 ) );
		unsigned int lidx = node->left, ridx = node->right;
		const __m128 min4 = _mm_max_ps( _mm_max_ps( _mm_max_ps( x4, y4 ), z4 ), _mm_setzero_ps() );
		const __m128 max4 = _mm_min_ps( _mm_min_ps( _mm_min_ps( x4, y4 ), z4 ), _mm_set1_ps( ray.hit.t ) );
	#if 0
		// TODO: why is this slower on gen14?
		const float tmina_0 = LANE( min4, 0 ), tmaxa_1 = LANE( max4, 1 );
		const float tminb_2 = LANE( min4, 2 ), tmaxb_3 = LANE( max4, 3 );
		t0 = _mm_shuffle_ps( max4, max4, _MM_SHUFFLE( 1, 3, 1, 3 ) );
		t1 = _mm_shuffle_ps( min4, min4, _MM_SHUFFLE( 0, 2, 0, 2 ) );
		t0 = _mm_blendv_ps( inf4, t1, _mm_cmpge_ps( t0, t1 ) );
		float dist1 = LANE( t0, 1 ), dist2 = LANE( t0, 0 );
	#else
		const float tmina_0 = LANE( min4, 0 ), tmaxa_1 = LANE( max4, 1 );
		const float tminb_2 = LANE( min4, 2 ), tmaxb_3 = LANE( max4, 3 );
		float dist1 = tmaxa_1 >= tmina_0 ? tmina_0 : 1e30f;
		float dist2 = tmaxb_3 >= tminb_2 ? tminb_2 : 1e30f;
	#endif
		if (dist1 > dist2)
		{
			float t = dist1; dist1 = dist2; dist2 = t;
			unsigned int i = lidx; lidx = ridx; ridx = i;
		}
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = alt2Node + lidx;
			if (dist2 != 1e30f) stack[stackPtr++] = alt2Node + ridx;
		}
	}
	return steps;
}

#else

int BVH::Intersect_AltSoA( Ray& ray ) const
{
	// not implemented for platforms that do not support SSE/AVX.
	assert( false );
	return 0;
}

#endif // BVH_USEAVX

} // namespace tinybvh

#endif // TINYBVH_IMPLEMENTATION

#endif // TINY_BVH_H_