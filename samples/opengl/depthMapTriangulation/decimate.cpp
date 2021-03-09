/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "decimate.h"

namespace Saiga
{
// --- PUBLIC ---

QuadricDecimater::QuadricDecimater(const Settings& s) : settings(s) {}

void QuadricDecimater::decimate(MyMesh& mesh)
{
    // save a pointer to the mesh so all QuadricDecimater methods can access it easily
    // the pointer will only be used by this method and all the methods called within it
    current_mesh = &mesh;

    // https://www.ri.cmu.edu/pub_files/pub2/garland_michael_1997_1/garland_michael_1997_1.pdf

    // Pre-computation

    // add the relevant properties
    current_mesh->add_property(h_error);
    current_mesh->add_property(h_heap_position);
    current_mesh->add_property(h_errorMatrix);
    current_mesh->add_property(h_collapseTarget);
    if (settings.check_folding_triangles) current_mesh->add_property(h_folding_triangles_edge);
    if (settings.check_self_intersections) current_mesh->add_property(h_collapse_self_intersection);
    if (settings.only_collapse_roughly_parallel_borders) current_mesh->add_property(h_parallel_border_edges);
    if (settings.check_interior_angles) current_mesh->add_property(h_interior_angles_undershot);

    // calculate face normals
    current_mesh->request_face_normals();
    current_mesh->update_face_normals();

    // activate status so that items can be deleted
    current_mesh->request_face_status();
    current_mesh->request_edge_status();
    current_mesh->request_halfedge_status();
    current_mesh->request_vertex_status();

    // Main body of decimation

    // get all the information required about edges
    if (settings.check_self_intersections || settings.check_interior_angles || settings.check_folding_triangles ||
        settings.only_collapse_roughly_parallel_borders)
    {
        MyMesh::EdgeIter e_it, e_end(current_mesh->edges_end());
        for (e_it = current_mesh->edges_sbegin(); e_it != e_end; ++e_it)
        {
            update_edge(*e_it);
        }
    }

    // 1. Compute the Q matrices for all the initial vertices
    {
        // --- Calculation of fundamental error matrices per face ---

        // add error fundamental error matrix property to the faces
        current_mesh->add_property(h_fund_error_mat);

        // iterate through faces and calculate their fundamental error matrix
        MyMesh::FaceIter f_it, f_end(current_mesh->faces_end());
        MyMesh::FaceVertexCWIter fv_it;

        for (f_it = current_mesh->faces_sbegin(); f_it != f_end; ++f_it)
        {
            current_mesh->property(h_fund_error_mat, *f_it) = calculate_fundamental_error_matrix(*f_it);
        }

        // --- Calculation of Q matrices per vertex ---

        // iterate all vertices and calculate their error matrices
        MyMesh::VertexIter v_it, v_end(current_mesh->vertices_end());
        MyMesh::VertexFaceCWIter vf_it;
        for (v_it = current_mesh->vertices_sbegin(); v_it != v_end; ++v_it)
        {
            // circulate the faces of the vertex and add the matrices
            mat4 error_mat = mat4::Zero();

            vf_it = current_mesh->cvf_cwbegin(*v_it);
            for (; vf_it.is_valid(); ++vf_it)
            {
                MyMesh::FaceHandle f = *vf_it;
                error_mat += current_mesh->property(h_fund_error_mat, f);
            }

            // set the vertex error matrix
            current_mesh->property(h_errorMatrix, *v_it) = error_mat;
        }

        // remove fundamental error matrices from faces
        current_mesh->remove_property(h_fund_error_mat);
    }


    // 2. Find a collapse target and the corresponding error for every vertex

    // initialize heap
    OpenMesh::Decimater::DecimaterT<OpenTriangleMesh>::HeapInterface collapseCandidates_HI(*current_mesh, h_error,
                                                                                           h_heap_position);
    collapseCandidates_heap.reset(new DeciHeap(collapseCandidates_HI));
    collapseCandidates_heap->reserve(current_mesh->n_vertices());

    // do the decimation loop
    {
        // iterate all vertices and find their best decimation partner
        MyMesh::VertexIter v_it, v_end(current_mesh->vertices_end());
        for (v_it = current_mesh->vertices_begin(); v_it != v_end; ++v_it)
        {
            collapseCandidates_heap->reset_heap_position(*v_it);
            if (!current_mesh->status(*v_it).deleted()) update_vertex(*v_it);
        }
    }

    // 3. Iteratively remove the pair of least cost from the heap, contract this pair, and find new best contraction
    // partners for all neighbouring vertices
    {
        MyMesh::VertexHandle current_candidate;
        std::vector<MyMesh::VertexHandle> support;

        // initialize counter variables for in case a specific amount of decimations is requested
        int decimated_vertices = 0;
        int my_max_collapses = (settings.max_decimations <= 0) ? current_mesh->n_vertices() : settings.max_decimations;

        while (!collapseCandidates_heap->empty() && decimated_vertices < my_max_collapses)
        {
            current_candidate = collapseCandidates_heap->front();
            collapseCandidates_heap->pop_front();

            // collapse the edge
            MyMesh::HalfedgeHandle collapse_edge = current_mesh->property(h_collapseTarget, current_candidate);
            if (!custom_is_collapse_legal(collapse_edge))
            {
                // re-calculate the vertex error
                update_vertex(current_candidate);
                continue;
            }

            MyMesh::VertexHandle vh_base   = current_mesh->from_vertex_handle(collapse_edge);
            MyMesh::VertexHandle vh_target = current_mesh->to_vertex_handle(collapse_edge);
            mat4 new_error_mat =
                current_mesh->property(h_errorMatrix, vh_base) + current_mesh->property(h_errorMatrix, vh_target);
            current_mesh->property(h_errorMatrix, vh_target) = new_error_mat;

            // save all the vertices that will have to be updated
            support.clear();
            for (auto vv_it = current_mesh->vv_cwiter(vh_base); vv_it.is_valid(); ++vv_it)
            {
                support.push_back(*vv_it);
            }

            current_mesh->collapse(collapse_edge);
            ++decimated_vertices;

            // update the face normals surrounding the target vertex
            MyMesh::VertexFaceCWIter vf_it = current_mesh->vf_cwiter(vh_target);
            for (; vf_it.is_valid(); ++vf_it)
            {
                current_mesh->set_normal(*vf_it, current_mesh->calc_face_normal(*vf_it));
            }

            // update the information of surrounding edges
            update_edges(vh_target);

            // update the newly created vertex, its neighbours and their location in the heap
            for (MyMesh::VertexHandle vh : support)
            {
                SAIGA_ASSERT(!current_mesh->status(vh).deleted());
                update_vertex(vh);
            }
        }
    }

    // Post-computation
    collapseCandidates_heap.reset();
    current_mesh->delete_isolated_vertices();
    current_mesh->garbage_collection();

    // remove properties
    current_mesh->remove_property(h_error);
    current_mesh->remove_property(h_heap_position);
    current_mesh->remove_property(h_errorMatrix);
    current_mesh->remove_property(h_collapseTarget);
    if (settings.check_folding_triangles) current_mesh->remove_property(h_folding_triangles_edge);
    if (settings.check_self_intersections) current_mesh->remove_property(h_collapse_self_intersection);
    if (settings.only_collapse_roughly_parallel_borders) current_mesh->remove_property(h_parallel_border_edges);
    if (settings.check_interior_angles) current_mesh->remove_property(h_interior_angles_undershot);

    // deactivate status
    current_mesh->release_face_status();
    current_mesh->release_edge_status();
    current_mesh->release_vertex_status();
}

// --- PRIVATE ---

bool QuadricDecimater::check_minimal_interior_angles_undershot(MyMesh::HalfedgeHandle collapse_edge)
{
    MyMesh::FaceHandle collapse_face_1(current_mesh->face_handle(collapse_edge));
    MyMesh::FaceHandle collapse_face_2(current_mesh->opposite_face_handle(collapse_edge));

    MyMesh::VFCWIter vf_iter = current_mesh->vf_cwiter(current_mesh->from_vertex_handle(collapse_edge));
    for (; vf_iter.is_valid(); ++vf_iter)
    {
        // the faces that disappear in the collapse don't matter
        if (*vf_iter == collapse_face_1 || *vf_iter == collapse_face_2) continue;

        // find the vertices of the resulting face after a collapse
        MyMesh::FVCCWIter fv_iter = current_mesh->fv_ccwbegin(*vf_iter);
        MyMesh::VertexHandle vh1  = *fv_iter;
        ++fv_iter;
        MyMesh::VertexHandle vh2 = *fv_iter;
        ++fv_iter;
        MyMesh::VertexHandle vh3 = *fv_iter;

        if (vh1 == current_mesh->from_vertex_handle(collapse_edge))
            vh1 = current_mesh->to_vertex_handle(collapse_edge);
        else if (vh2 == current_mesh->from_vertex_handle(collapse_edge))
            vh2 = current_mesh->to_vertex_handle(collapse_edge);
        else if (vh3 == current_mesh->from_vertex_handle(collapse_edge))
            vh3 = current_mesh->to_vertex_handle(collapse_edge);

        OpenMesh::Vec3f v1 = current_mesh->point(vh1);
        OpenMesh::Vec3f v2 = current_mesh->point(vh2);
        OpenMesh::Vec3f v3 = current_mesh->point(vh3);

        // get the side lenght from the vertices
        float a = (v1 - v2).length();
        float b = (v2 - v3).length();
        float c = (v3 - v1).length();

        // calculate interior angles of the triangle
        if (acos((b * b + c * c - a * a) / (2 * b * c)) < settings.minimal_interior_angle_rad ||
            acos((a * a + c * c - b * b) / (2 * a * c)) < settings.minimal_interior_angle_rad ||
            acos((a * a + b * b - c * c) / (2 * a * b)) < settings.minimal_interior_angle_rad)
        {
            return true;
        }
    }
    return false;
}

bool QuadricDecimater::check_collapse_self_intersection(MyMesh::HalfedgeHandle collapse_edge)
{
    MyMesh::FaceHandle collapse_face_1(current_mesh->face_handle(collapse_edge));
    MyMesh::FaceHandle collapse_face_2(current_mesh->opposite_face_handle(collapse_edge));

    MyMesh::VFCCWIter vf_iter = current_mesh->vf_ccwiter(current_mesh->from_vertex_handle(collapse_edge));
    for (; vf_iter.is_valid(); ++vf_iter)
    {
        // the faces that disappear in the collapse don't matter
        if (*vf_iter == collapse_face_1 || *vf_iter == collapse_face_2) continue;

        OpenMesh::Vec3f pre_collapse_normal = current_mesh->normal(*vf_iter).normalized();

        // find the vertices of the resulting face after a collapse
        MyMesh::FVCCWIter fv_iter = current_mesh->fv_ccwbegin(*vf_iter);
        MyMesh::VertexHandle vh1  = *fv_iter;
        ++fv_iter;
        MyMesh::VertexHandle vh2 = *fv_iter;
        ++fv_iter;
        MyMesh::VertexHandle vh3 = *fv_iter;

        if (vh1 == current_mesh->from_vertex_handle(collapse_edge))
            vh1 = current_mesh->to_vertex_handle(collapse_edge);
        else if (vh2 == current_mesh->from_vertex_handle(collapse_edge))
            vh2 = current_mesh->to_vertex_handle(collapse_edge);
        else if (vh3 == current_mesh->from_vertex_handle(collapse_edge))
            vh3 = current_mesh->to_vertex_handle(collapse_edge);

        OpenMesh::Vec3f v1 = current_mesh->point(vh1);
        OpenMesh::Vec3f v2 = current_mesh->point(vh2);
        OpenMesh::Vec3f v3 = current_mesh->point(vh3);

        // get the normal from those vertices
        OpenMesh::Vec3f edge1                = v2 - v1;
        OpenMesh::Vec3f edge2                = v3 - v1;
        OpenMesh::Vec3f post_collapse_normal = cross(edge1, edge2).normalized();

        // if the angle between old and new normal is too great, the triangle probably gets flipped
        if (dot(pre_collapse_normal, post_collapse_normal) < 0.5)
        {  // approximately 60 degree
            return true;
        }
    }
    return false;
}

bool QuadricDecimater::custom_is_collapse_legal(MyMesh::HalfedgeHandle v0v1)
{
    /**
     *       vl
     *        *
     *       / \
     *      /   \
     *     / fl  \
     * v0 *------>* v1
     *     \ fr  /
     *      \   /
     *       \ /
     *        *
     *        vr
     **/

    // get the handles
    MyMesh::HalfedgeHandle v1v0(current_mesh->opposite_halfedge_handle(v0v1));  ///< Reverse halfedge
    MyMesh::VertexHandle v0(current_mesh->to_vertex_handle(v1v0));              ///< Vertex to be removed
    MyMesh::VertexHandle v1(current_mesh->to_vertex_handle(v0v1));              ///< Remaining vertex
    MyMesh::FaceHandle fl(current_mesh->face_handle(v0v1));                     ///< Left face
    MyMesh::FaceHandle fr(current_mesh->face_handle(v1v0));                     ///< Right face
    MyMesh::VertexHandle vl;                                                    ///< Left vertex
    MyMesh::VertexHandle vr;                                                    ///< Right vertex

    MyMesh::HalfedgeHandle vlv1, v0vl, vrv0, v1vr;  ///< Outer remaining halfedges

    // get vl
    if (fl.is_valid())
    {
        vlv1 = current_mesh->next_halfedge_handle(v0v1);
        v0vl = current_mesh->next_halfedge_handle(vlv1);
        vl   = current_mesh->to_vertex_handle(vlv1);
        vlv1 = current_mesh->opposite_halfedge_handle(vlv1);
        v0vl = current_mesh->opposite_halfedge_handle(v0vl);
    }

    // get vr
    if (fr.is_valid())
    {
        vrv0 = current_mesh->next_halfedge_handle(v1v0);
        v1vr = current_mesh->next_halfedge_handle(vrv0);
        vr   = current_mesh->to_vertex_handle(vrv0);
        vrv0 = current_mesh->opposite_halfedge_handle(vrv0);
        v1vr = current_mesh->opposite_halfedge_handle(v1vr);
    }

    // -------------------------------------
    // check if things are legal to collapse
    // -------------------------------------

    // was the vertex locked by someone?
    if (current_mesh->status(v0).locked()) return false;

    // this test checks:
    // is v0v1 deleted?
    // is v0 deleted?
    // is v1 deleted?
    // are both vlv0 and v1vl boundary edges?
    // are both v0vr and vrv1 boundary edges?
    // are vl and vr equal or both invalid?
    // one ring intersection test
    // edge between two boundary vertices should be a boundary edge
    if (!current_mesh->is_collapse_ok(v0v1)) return false;

    // check for self intersections after the collapse
    if (settings.check_self_intersections && current_mesh->property(h_collapse_self_intersection, v0v1)) return false;

    // check whether the edge is a border and roughly parallel to the other affected border edge
    if (settings.only_collapse_roughly_parallel_borders && current_mesh->is_boundary(current_mesh->edge_handle(v0v1)))
    {
        if (current_mesh->property(h_parallel_border_edges, v0v1) == false)
        {
            return false;
        }
    }

    if (vl.is_valid() && vr.is_valid() && current_mesh->find_halfedge(vl, vr).is_valid() &&
        current_mesh->valence(vl) == 3 && current_mesh->valence(vr) == 3)
    {
        return false;
    }
    //--- feature test ---

    if (current_mesh->status(v0).feature() && !current_mesh->status(current_mesh->edge_handle(v0v1)).feature())
        return false;

    //--- test boundary cases ---
    if (current_mesh->is_boundary(v0))
    {
        // don't collapse a boundary vertex to an inner one
        if (!current_mesh->is_boundary(v1)) return false;

        // only one one ring intersection
        if (vl.is_valid() && vr.is_valid()) return false;
    }

    // there have to be at least 2 incident faces at v0
    if (current_mesh->cw_rotated_halfedge_handle(current_mesh->cw_rotated_halfedge_handle(v0v1)) == v0v1) return false;

    // collapse passed all tests -> ok
    return true;
}

mat4 QuadricDecimater::calculate_fundamental_error_matrix(const MyMesh::FaceHandle fh)
{
    // https://en.wikipedia.org/wiki/Plane_(geometry)#Describing_a_plane_through_three_points

    MyMesh::FaceVertexCWIter fv_it = current_mesh->cfv_cwbegin(fh);
    MyMesh::VertexHandle vh0       = *fv_it;
    MyMesh::VertexHandle vh1       = *(++fv_it);
    MyMesh::VertexHandle vh2       = *(++fv_it);
    OpenMesh::Vec3f v0             = current_mesh->point(vh0);
    OpenMesh::Vec3f v1             = current_mesh->point(vh1);
    OpenMesh::Vec3f v2             = current_mesh->point(vh2);

    OpenMesh::Vec3f normal = OpenMesh::cross((v1 - v0), (v2 - v0));
    float area             = normal.norm();
    normal /= area;
    area *= 0.5;

    float d = -(normal[0] * v0[0]) - (normal[1] * v0[1]) - (normal[2] * v0[2]);

    mat4 k_p;
    k_p(0, 0) = normal[0] * normal[0] * area;
    k_p(0, 1) = normal[0] * normal[1] * area;
    k_p(0, 2) = normal[0] * normal[2] * area;
    k_p(0, 3) = normal[0] * d * area;

    k_p(1, 0) = k_p(0, 1);
    k_p(1, 1) = normal[1] * normal[1] * area;
    k_p(1, 2) = normal[1] * normal[2] * area;
    k_p(1, 3) = normal[1] * d * area;

    k_p(2, 0) = k_p(0, 2);
    k_p(2, 1) = k_p(1, 2);
    k_p(2, 2) = normal[2] * normal[2] * area;
    k_p(2, 3) = normal[2] * d * area;

    k_p(3, 0) = k_p(0, 3);
    k_p(3, 1) = k_p(1, 3);
    k_p(3, 2) = k_p(2, 3);
    k_p(3, 3) = d * d * area;

    return k_p;
}

bool QuadricDecimater::check_for_folding_triangles(const MyMesh::EdgeHandle edge)
{
    // check if any of the triangles that are altered by this collapse have normals with greater
    // angles than 60° to each other
    // --> loop through all faces of the vertices of the cendidat edge and check their normals against each other

    // collect all normals
    std::vector<OpenMesh::Vec3f> normals;
    MyMesh::VertexFaceCWIter fcw_it =
        current_mesh->cvf_cwbegin(current_mesh->to_vertex_handle(current_mesh->halfedge_handle(edge, 0)));
    MyMesh::VertexFaceCWIter fcw_it2 =
        current_mesh->cvf_cwbegin(current_mesh->from_vertex_handle(current_mesh->halfedge_handle(edge, 0)));

    while (fcw_it.is_valid())
    {
        if (current_mesh->status(*fcw_it).deleted()) continue;
        normals.push_back(current_mesh->normal(*fcw_it));
        ++fcw_it;
    }
    while (fcw_it2.is_valid())
    {
        if (current_mesh->status(*fcw_it2).deleted()) continue;
        normals.push_back(current_mesh->normal(*fcw_it2));
        ++fcw_it2;
    }

    // check them against each other
    for (int i = 0; i < (int)normals.size(); ++i)
    {
        OpenMesh::Vec3f normal_1 = normals[i];

        for (int j = i + 1; j < (int)normals.size(); ++j)
        {
            // dot(A,B) = |A| * |B| * cos(angle)
            // which can be rearranged to
            // angle = arccos(dot(A, B) / (|A| * |B|)).
            float angle = OpenMesh::dot(normal_1, normals[j]);
            angle       = acos(angle);

            if (angle > radians(60.0f))
            {
                return true;
            }
        }
    }
    return false;
}

bool QuadricDecimater::check_edge_parallelity(const OpenMesh::Vec3f v0, const OpenMesh::Vec3f v1,
                                              const OpenMesh::Vec3f v2)
{
    OpenMesh::Vec3f edge0((v0 - v1).normalize()), edge1((v1 - v2).normalize());

    // check the edges against each other
    float dot = OpenMesh::dot(edge0, edge1);
    if (abs(dot) < 0.95) return false;

    return true;
}

float QuadricDecimater::calculate_collapse_error(const MyMesh::HalfedgeHandle candidat_edge)
{
    MyMesh::VertexHandle vh1 = current_mesh->from_vertex_handle(candidat_edge);
    MyMesh::VertexHandle vh2 = current_mesh->to_vertex_handle(candidat_edge);
    vec4 v2 = vec4(current_mesh->point(vh2)[0], current_mesh->point(vh2)[1], current_mesh->point(vh2)[2], 1.0f);

    float error =
        dot(v2, (current_mesh->property(h_errorMatrix, vh1) + current_mesh->property(h_errorMatrix, vh2)) * v2);

    if (settings.check_folding_triangles)
    {
        if (current_mesh->property(h_folding_triangles_edge, current_mesh->edge_handle(candidat_edge)))
        {
            // the collapse affects folding triangles
            // --> add a penalty to the calculated error
            error = error * 10.0f + settings.folding_triangle_constant;
        }
    }

    // undershooting interior angles gives a strong penalty
    if (settings.check_interior_angles)
    {
        if (current_mesh->property(h_interior_angles_undershot, candidat_edge))
        {
            // the collapse causes some acute triangles
            // (or rather triangles with at least one interior angle below settings.minimal_interior_angle_rad)
            // --> add a penalty to the calculated error
            error = error * 10.0f + settings.interior_angle_constant;
        }
    }

    // divide error by squared distance
    OpenMesh::Vec3f a             = current_mesh->point(vh1);
    OpenMesh::Vec3f b             = current_mesh->point(vh2);
    OpenMesh::Vec3f mid           = (a + b) / 2.0f;
    float edge_to_camera_distance = mid.length();
    error                         = error / (edge_to_camera_distance * edge_to_camera_distance);

    return error;
}

float QuadricDecimater::find_collapse_partner(const MyMesh::VertexHandle vh, MyMesh::HalfedgeHandle& collapse_edge)
{
    MyMesh::VertexOHalfedgeCWIter he_it = current_mesh->cvoh_cwbegin(vh);
    float error                         = std::numeric_limits<float>::max();

    float error_tmp;
    for (; he_it.is_valid(); ++he_it)
    {
        if (!custom_is_collapse_legal(*he_it))
        {
            continue;
        }
        else
        {
            error_tmp = calculate_collapse_error(*he_it);
        }

        if (error_tmp < error)
        {
            collapse_edge = *he_it;
            error         = error_tmp;
        }
    }
    return error;
}

void QuadricDecimater::update_vertex(const MyMesh::VertexHandle vh)
{
    // calculate new best fit
    MyMesh::HalfedgeHandle heh;
    float error = find_collapse_partner(vh, heh);

    // target found -> put vertex on heap
    if (heh.is_valid() && error <= settings.quadricMaxError)
    {
        // update the information in the vertex
        current_mesh->property(h_collapseTarget, vh) = heh;
        current_mesh->property(h_error, vh)          = error;

        if (collapseCandidates_heap->is_stored(vh))
            collapseCandidates_heap->update(vh);
        else
            collapseCandidates_heap->insert(vh);
    }
    // not valid -> remove from heap
    else
    {
        // remove from set if the error is too big
        if (collapseCandidates_heap->is_stored(vh)) collapseCandidates_heap->remove(vh);

        current_mesh->property(h_error, vh) = -1;
    }
}

void QuadricDecimater::update_edge(const MyMesh::EdgeHandle eh)
{
    MyMesh::HalfedgeHandle heh0 = current_mesh->halfedge_handle(eh, 0);
    MyMesh::HalfedgeHandle heh1 = current_mesh->halfedge_handle(eh, 1);

    if (settings.check_self_intersections)
    {
        current_mesh->property(h_collapse_self_intersection, heh0) = check_collapse_self_intersection(heh0);
        current_mesh->property(h_collapse_self_intersection, heh1) = check_collapse_self_intersection(heh1);
    }

    if (settings.check_interior_angles)
    {
        current_mesh->property(h_interior_angles_undershot, heh0) = check_minimal_interior_angles_undershot(heh0);
        current_mesh->property(h_interior_angles_undershot, heh1) = check_minimal_interior_angles_undershot(heh1);
    }

    if (settings.check_folding_triangles)
        current_mesh->property(h_folding_triangles_edge, eh) = check_for_folding_triangles(eh);

    if (settings.only_collapse_roughly_parallel_borders)
    {
        // check if it's a border
        if (current_mesh->is_boundary(eh))
        {
            // find the right halfedge
            bool border_at_1 = current_mesh->is_boundary(heh1);

            OpenMesh::Vec3f v0, v1, v2, v3;

            if (border_at_1)
            {
                v0 = current_mesh->point(current_mesh->from_vertex_handle(current_mesh->prev_halfedge_handle(heh1)));
                v1 = current_mesh->point(current_mesh->from_vertex_handle(heh1));
                v2 = current_mesh->point(current_mesh->to_vertex_handle(heh1));
                v3 = current_mesh->point(current_mesh->to_vertex_handle(current_mesh->next_halfedge_handle(heh1)));

                current_mesh->property(h_parallel_border_edges, heh0) = check_edge_parallelity(v3, v2, v1);
                current_mesh->property(h_parallel_border_edges, heh1) = check_edge_parallelity(v0, v1, v2);
            }
            else
            {
                v0 = current_mesh->point(current_mesh->from_vertex_handle(current_mesh->prev_halfedge_handle(heh0)));
                v1 = current_mesh->point(current_mesh->from_vertex_handle(heh0));
                v2 = current_mesh->point(current_mesh->to_vertex_handle(heh0));
                v3 = current_mesh->point(current_mesh->to_vertex_handle(current_mesh->next_halfedge_handle(heh0)));

                current_mesh->property(h_parallel_border_edges, heh0) = check_edge_parallelity(v0, v1, v2);
                current_mesh->property(h_parallel_border_edges, heh1) = check_edge_parallelity(v3, v2, v1);
            }
        }
    }
}

void QuadricDecimater::update_edges(const MyMesh::VertexHandle vh)
{
    for (auto it = current_mesh->ve_begin(vh); it.is_valid(); ++it)
    {
        update_edge(*it);
    }
}
}  // namespace Saiga
