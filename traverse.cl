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

void kernel traverse( global struct BVHNodeAlt* altNode, global unsigned* idx, global float4* verts, global struct Ray* rayData )
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