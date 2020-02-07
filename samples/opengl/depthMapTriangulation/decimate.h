/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/openMeshWrapper.h"

#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <set>


namespace Saiga
{
class QuadricDecimater
{
   public:
    using MyMesh   = OpenTriangleMesh;
    using DeciHeap = OpenMesh::Decimater::DecimaterT<MyMesh>::DeciHeap;

    struct Settings
    {
        // max_decimations <= 0 means unlimited decimations are allowed
        int max_decimations = 0;
        // if there is no collapse left that would cause a smaller error than quadricMaxError, the decimation is done
        float quadricMaxError = 0.000001f;
        // if this is true, collapses that invert the orientation of one or more faces are prohibited
        bool check_self_intersections = true;
        // if this is true, collapses that influence faces which are more than 60 degree apart receive a penalty towards
        // their error
        bool check_folding_triangles    = true;
        float folding_triangle_constant = 0;

        // if this is true, border edges only get decimated if they are roughly parallel to the other influenced edges
        bool only_collapse_roughly_parallel_borders = true;
        // if this is true, collapses that create a face with an interior angle smaller than minimal_interior_angle_rad
        // receive a penalty towards their error
        bool check_interior_angles       = true;
        float minimal_interior_angle_rad = radians(13.0f);
        float interior_angle_constant    = 0;
    };

    QuadricDecimater(const Settings& s);

    // decimates the mesh as described in the paper here:
    // https://www.ri.cmu.edu/pub_files/pub2/garland_michael_1997_1/garland_michael_1997_1.pdf
    // One major difference is that only vertex-pairs sharing an edge are considered for decimation
    void decimate(MyMesh& mesh);

   private:
    MyMesh* current_mesh;
    Settings settings;

    std::unique_ptr<DeciHeap> collapseCandidates_heap;

    // --- Functions ---

    // checks whether any of the triangles after the collapse of collapse_edge would have an interior angle smaller than
    // the given min_angle calculations taken from https://www.calculator.net/triangle-calculator.html
    bool check_minimal_interior_angles_undershot(MyMesh::HalfedgeHandle collapse_edge);

    // checks whether any of the adjacent faces of a collapsing edge would get flipped
    // this assumes that a collapse moves the from-vertex to the to-vertex of the edge (no custom target locations
    // allowed)
    bool check_collapse_self_intersection(MyMesh::HalfedgeHandle collapse_edge);

    // checks several things for the collapse of a halfedge in mesh
    // inspired by OpenMesh::Decimator::BaseDecimatorT<class Mesh>::is_collapse_legal(...)
    // ----> also checks for valid, not deleted vertices, edges and a correct boundary
    // further description in the declaration
    bool custom_is_collapse_legal(MyMesh::HalfedgeHandle v0v1);

    // calculates the fundamental error matrix K for a face.
    // The error is also dependent on the area of the face
    mat4 calculate_fundamental_error_matrix(const MyMesh::FaceHandle fh);

    // gets an edge and checks if any of the adjacent faces have normals that are more than 60 degree off from each
    // other
    bool check_for_folding_triangles(const MyMesh::EdgeHandle edge);

    // gets three vertices which represent two edges and checks, whether those edges are roughly parallel to each other
    // v0 and v1 indicate the first edge, v1 and v2 the second one
    bool check_edge_parallelity(const OpenMesh::Vec3f v0, const OpenMesh::Vec3f v1, const OpenMesh::Vec3f v2);

    // calculates the error for collapsing the given half edge and sets new_vertex to the resulting vertex
    // The math part is taken from the paper and from samuel
    float calculate_collapse_error(const MyMesh::HalfedgeHandle candidat_edge);

    // finds the best collapse partner of a vertex
    // returns the error and writes the found collapse-partner to collapse_edge
    float find_collapse_partner(const MyMesh::VertexHandle vh, MyMesh::HalfedgeHandle& collapse_edge);

    // updates following parts of a vertex: h_collapseTarget, h_error
    // removes the old vertex entry from the heap and inserts a new one with the new error
    void update_vertex(const MyMesh::VertexHandle vh);

    // updates following parts of an edge: h_folding_triangles_edge:
    // h_collapse_self_intersection, h_interior_angles_undershot, h_folding_triangles_edge, h_parallel_border_edges
    void update_edge(const MyMesh::EdgeHandle eh);
    // this calls update_edge for all edges adjacent to the given vertex
    void update_edges(const MyMesh::VertexHandle vh);

    // --- Mesh properties ---

    // --- vertex properties ---
    // a property to store the error of the collapse using collapseTarget
    OpenMesh::VPropHandleT<float> h_error;
    // stores the vertex' position in the priority heap
    OpenMesh::VPropHandleT<int> h_heap_position;
    // stores the error matrix of the vertex
    OpenMesh::VPropHandleT<mat4> h_errorMatrix;
    // stores the halfEdge whose collapse causes the lowest error
    OpenMesh::VPropHandleT<MyMesh::HalfedgeHandle> h_collapseTarget;

    // --- edge properties ---
    // a boolean indicating whether there are adjacent faces to
    // the Edge with more than 60 degree difference in their normals
    OpenMesh::EPropHandleT<bool> h_folding_triangles_edge;

    // --- halfedge properties ---
    // a boolean indicating whether collapsing this halfedge
    // results in a flipped triangle
    OpenMesh::HPropHandleT<bool> h_collapse_self_intersection;
    // a boolean indicating whether collapsing this border halfedge results in a
    // relatively parallel border halfedge or not (non-borders have unspecified values)
    OpenMesh::HPropHandleT<bool> h_parallel_border_edges;
    // a boolean indicating whether collapsing this halfedge causes any of the adjacent
    // triangles to have one or more interior angles below minimal_interior_angle
    OpenMesh::HPropHandleT<bool> h_interior_angles_undershot;

    // --- face properties ---
    // a property to store the fundamental error matrices per face
    OpenMesh::FPropHandleT<mat4> h_fund_error_mat;
};
}  // namespace Saiga
