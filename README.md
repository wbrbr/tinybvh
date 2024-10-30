# tinybvh
Single-header BVH construction and traversal library written as "Sane C++" (or "C with classes"). The library has no dependencies.

# BVH?
A Bounding Volume Hierarchy is a data structure used to quickly find intersections in a virtual scene; most commonly between a ray and a group of triangles. You can read more about this in a series of articles on the subject: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics .

# How To Use
The library is designed to be easy to use. Please have a look at tiny_bvh_test.cpp for an example.

# Version
The current version of the library is a 'prototype'; please expect changes to the interface. Once the interface has settled, more functionality will follow. Plans:
* Conversion to GPU-friendly format

# License
This library is made available under the MIT 2.0 license.
