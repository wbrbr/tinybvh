// gpu-side code for ray traversal

// Note: We are taking in nodes and rays as collection of floa4's here.
// You can use structs in OpenCL, which will be more convenient and
// clear here. Be careful though: float3 / int3 struct members are padded 
// to 16 bytes in OpenCL.

void kernel traverse( global float4* bvhNode, global unsigned* idx, global float4* verts, global float4* rayData )
{
	// fetch ray
	unsigned threadId = get_global_id( 0 );
	float3 O = rayData[threadId * 4 + 0].xyz;
	float3 D = rayData[threadId * 4 + 1].xyz;
	float3 rD = rayData[threadId * 4 + 2].xyz;
	float4 hit = rayData[threadId * 4 + 3];
	hit.x = 1e30f;
	// traverse BVH
	unsigned node = 0, stack[64], stackPtr = 0;
	while (1)
	{
		// fetch the node
		const float4 lmin = bvhNode[node * 4 + 0];
		const float4 lmax = bvhNode[node * 4 + 1];
		const float4 rmin = bvhNode[node * 4 + 2];
		const float4 rmax = bvhNode[node * 4 + 3];
		const unsigned triCount = as_uint( rmin.w );
		if (triCount > 0)
		{
			// process leaf node
			const unsigned firstTri = as_uint( rmax.w );
			for (unsigned i = 0; i < triCount; i++)
			{
				const unsigned triIdx = idx[firstTri + i];
				const float4* tri = verts + 3 * triIdx;
				// triangle intersection
				const float4 edge1 = tri[1] - tri[0], edge2 = tri[2] - tri[0];
				const float3 h = cross( D, edge2.xyz );
				const float a = dot( edge1.xyz, h );
				if (fabs( a ) >= 0.0000001f)
				{
					const float f = 1 / a;
					const float3 s = O - tri[0].xyz;
					const float u = f * dot( s, h );
					if (u >= 0 && u <= 1)
					{
						const float3 q = cross( s, edge1.xyz );
						const float v = f * dot( D, q );
						if (v >= 0 && u + v <= 1)
						{
							const float d = f * dot( edge2.xyz, q );
							if (d > 0.0f && d < hit.x /* i.e., ray.t */)
								hit = (float4)(d, u, v, as_float( triIdx ));
						}
					}
				}
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
		const float tmaxa = fmin( fmin( fmin( maxta.x, maxta.y ), maxta.z ), hit.x );
		const float tmaxb = fmin( fmin( fmin( maxtb.x, maxtb.y ), maxtb.z ), hit.x );
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
	rayData[threadId * 4 + 3] = hit;
}