// gpu-side code for ray traversal

struct Ray
{
	// data is defined here as 16-byte values to encourage the compilers
	// to fetch 16 bytes at a time: 12 (so, 8 + 4) will be slower.
	float4 O, D, rD; // 48 byte
	float4 hit; // 16 byte
};

struct BVHNodeAlt
{
	float4 lmin; // unsigned left in w
	float4 lmax; // unsigned right in w
	float4 rmin; // unsigned triCount in w
	float4 rmax; // unsigned firstTri in w
};

void kernel traverse_ailalaine( global struct BVHNodeAlt* altNode, global unsigned* idx, global float4* verts, global struct Ray* rayData )
{
	// fetch ray
	const unsigned threadId = get_global_id( 0 );
	const float3 O = rayData[threadId].O.xyz;
	const float3 D = rayData[threadId].D.xyz;
	const float3 rD = rayData[threadId].rD.xyz;
	float t = 1e30f; // ignoring value set in ray to spare one memory transaction.
	float4 hit;
	// traverse BVH
	unsigned node = 0, stack[64], stackPtr = 0;
	while (1)
	{
		// fetch the node
		const float4 lmin = altNode[node].lmin, lmax = altNode[node].lmax;
		const float4 rmin = altNode[node].rmin, rmax = altNode[node].rmax;
		const unsigned triCount = as_uint( rmin.w );
		if (triCount > 0)
		{
			// process leaf node
			const unsigned firstTri = as_uint( rmax.w );
			for (unsigned i = 0; i < triCount; i++)
			{
				const unsigned triIdx = idx[firstTri + i];
				const float4* tri = verts + 3 * triIdx;
				// triangle intersection - Möller-Trumbore
				const float4 edge1 = tri[1] - tri[0], edge2 = tri[2] - tri[0];
				const float3 h = cross( D, edge2.xyz );
				const float a = dot( edge1.xyz, h );
				if (fabs( a ) < 0.0000001f) continue;
				const float f = 1 / a;
				const float3 s = O - tri[0].xyz;
				const float u = f * dot( s, h );
				if (u < 0 && u > 1) continue;
				const float3 q = cross( s, edge1.xyz );
				const float v = f * dot( D, q );
				if (v < 0 && u + v > 1) continue;
				const float d = f * dot( edge2.xyz, q );
				if (d > 0.0f && d < t) hit = (float4)(t = d, u, v, as_float( triIdx ));
			}
			if (stackPtr == 0) break;
			node = stack[--stackPtr];
			continue;
		}
		unsigned left = as_uint( lmin.w ), right = as_uint( lmax.w );
		// child AABB intersection tests
		const float3 t1a = (lmin.xyz - O) * rD, t2a = (lmax.xyz - O) * rD;
		const float3 t1b = (rmin.xyz - O) * rD, t2b = (rmax.xyz - O) * rD;
		const float3 minta = fmin( t1a, t2a ), maxta = fmax( t1a, t2a );
		const float3 mintb = fmin( t1b, t2b ), maxtb = fmax( t1b, t2b );
		const float tmina = fmax( fmax( fmax( minta.x, minta.y ), minta.z ), 0 );
		const float tminb = fmax( fmax( fmax( mintb.x, mintb.y ), mintb.z ), 0 );
		const float tmaxa = fmin( fmin( fmin( maxta.x, maxta.y ), maxta.z ), t );
		const float tmaxb = fmin( fmin( fmin( maxtb.x, maxtb.y ), maxtb.z ), t );
		float dist1 = tmina > tmaxa ? 1e30f : tmina;
		float dist2 = tminb > tmaxb ? 1e30f : tminb;
		// traverse nearest child first
		if (dist1 > dist2)
		{
			float h = dist1; dist1 = dist2; dist2 = h;
			unsigned t = left; left = right; right = t;
		}
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = left;
			if (dist2 != 1e30f) stack[stackPtr++] = right;
		}
	}
	// write back intersection result
	rayData[threadId].hit = hit;
}

void kernel traverse_gpu4way( global float4* alt4Node, global struct Ray* rayData )
{
	// fetch ray
	const unsigned threadId = get_global_id( 0 );
	const float3 O = rayData[threadId].O.xyz;
	const float3 D = rayData[threadId].D.xyz;
	const float3 rD = rayData[threadId].rD.xyz;
	float4 hit;
	hit.x = 1e30f;
	// some local memory for storing leaf information
	local unsigned smem[64 * 4];
	// traverse the BVH
	const float4 zero4 = (float4)(0);
	unsigned offset = 0, stack[64], stackPtr = 0;
	const unsigned smBase = get_local_id( 0 ) * 4;
	while (1)
	{
		// vectorized 4-wide quantized aabb intersection
		const float4 data0 = alt4Node[offset];
		const float4 data1 = alt4Node[offset + 1];
		const float4 data2 = alt4Node[offset + 2];
		const float4 cminx4 = convert_float4( as_uchar4( data0.w ) );
		const float4 cmaxx4 = convert_float4( as_uchar4( data1.w ) );
		const float4 cminy4 = convert_float4( as_uchar4( data2.x ) );
		const float3 bminO = (O - data0.xyz) * rD, rDe = rD * data1.xyz;
		const float4 cmaxy4 = convert_float4( as_uchar4( data2.y ) );
		const float4 cminz4 = convert_float4( as_uchar4( data2.z ) );
		const float4 cmaxz4 = convert_float4( as_uchar4( data2.w ) );
		const float4 t1x4 = cminx4 * rDe.xxxx - bminO.xxxx, t2x4 = cmaxx4 * rDe.xxxx - bminO.xxxx;
		const float4 t1y4 = cminy4 * rDe.yyyy - bminO.yyyy, t2y4 = cmaxy4 * rDe.yyyy - bminO.yyyy;
		const float4 t1z4 = cminz4 * rDe.zzzz - bminO.zzzz, t2z4 = cmaxz4 * rDe.zzzz - bminO.zzzz;
		uint4 data3 = as_uint4( alt4Node[offset + 3] );
		const float4 mintx4 = fmin( t1x4, t2x4 ), maxtx4 = fmax( t1x4, t2x4 );
		const float4 minty4 = fmin( t1y4, t2y4 ), maxty4 = fmax( t1y4, t2y4 );
		const float4 mintz4 = fmin( t1z4, t2z4 ), maxtz4 = fmax( t1z4, t2z4 );
		const float4 maxxy4 = select( mintx4, minty4, isless( mintx4, minty4 ) );
		const float4 maxyz4 = select( maxxy4, mintz4, isless( maxxy4, mintz4 ) );
		float4 dst4 = select( maxyz4, zero4, isless( maxyz4, zero4 ) );
		const float4 minxy4 = select( maxtx4, maxty4, isgreater( maxtx4, maxty4 ) );
		const float4 minyz4 = select( minxy4, maxtz4, isgreater( minxy4, maxtz4 ) );
		const float4 tmax4 = select( minyz4, hit.xxxx, isgreater( minyz4, hit.xxxx ) );
		dst4 = select( dst4, (float4)(1e30f), isgreater( dst4, tmax4 ) );
		// sort intersection distances
		if (dst4.x < dst4.z) dst4 = dst4.zyxw, data3 = data3.zyxw; // bertdobbelaere.github.io/sorting_networks.html
		if (dst4.y < dst4.w) dst4 = dst4.xwzy, data3 = data3.xwzy;
		if (dst4.x < dst4.y) dst4 = dst4.yxzw, data3 = data3.yxzw;
		if (dst4.z < dst4.w) dst4 = dst4.xywz, data3 = data3.xywz;
		if (dst4.y < dst4.z) dst4 = dst4.xzyw, data3 = data3.xzyw;
		// process results, starting with farthest child, so nearest ends on top of stack
		unsigned nextNode = 0, leafs = 0;
		if (dst4.x < 1e30f) if (data3.x >> 31) smem[smBase + leafs++] = data3.x; else nextNode = data3.x;
		if (dst4.y < 1e30f) if (data3.y >> 31) smem[smBase + leafs++] = data3.y; else
		{
			if (nextNode) stack[stackPtr++] = nextNode;
			nextNode = data3.y;
		}
		if (dst4.z < 1e30f) if (data3.z >> 31) smem[smBase + leafs++] = data3.z; else
		{
			if (nextNode) stack[stackPtr++] = nextNode;
			nextNode = data3.z;
		}
		if (dst4.w < 1e30f) if (data3.w >> 31) smem[smBase + leafs++] = data3.w; else
		{
			if (nextNode) stack[stackPtr++] = nextNode;
			nextNode = data3.w;
		}
		// process encountered leaf primitives
		int leaf = 0, prim = 0;
		while (leaf < leafs)
		{
			const unsigned leafInfo = smem[smBase + leaf];
			unsigned thisTri = (leafInfo & 0xffff) + offset + prim * 3;
			const float4 v0 = alt4Node[thisTri];
			const float4 v1 = alt4Node[thisTri + 1];
			const float4 v2 = alt4Node[thisTri + 2];
			const unsigned triCount = (leafInfo >> 16) & 0x7fff;
			if (++prim == triCount) prim = 0, leaf++;
			const float4 edge1 = v1 - v0, edge2 = v2 - v0;
			const float3 h = cross( D, edge2.xyz );
			const float a = dot( edge1.xyz, h );
			if (fabs( a ) < 0.0000001f) continue;
			const float f = native_recip( a );
			const float3 s = O - v0.xyz;
			const float u = f * dot( s, h );
			if (u < 0 || u > 1) continue;
			const float3 q = cross( s, edge1.xyz );
			const float v = f * dot( D, q );
			if (v < 0 || u + v > 1) continue;
			const float d = f * dot( edge2.xyz, q );
			if (d <= 0.0f || d > hit.x) continue;
			hit = (float4)(d, u, v, v0.w);
		}
		// continue with nearest node or first node on the stack
		if (nextNode) offset = nextNode; else
		{
			if (!stackPtr) break;
			offset = stack[--stackPtr];
		}
	}
	rayData[threadId].hit = hit;
}