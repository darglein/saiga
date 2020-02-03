/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "decimate.h"

// --- PUBLIC ---

Saiga::QuadricDecimater::QuadricDecimater(Settings const& s) : settings(s) {}

Saiga::QuadricDecimater::~QuadricDecimater() {}

void Saiga::QuadricDecimater::decimate_quadric(MyMesh& mesh_in)
{
    mesh = mesh_in;

    // https://www.ri.cmu.edu/pub_files/pub2/garland_michael_1997_1/garland_michael_1997_1.pdf

    // Pre-computation

    // add the relevant properties
    mesh.add_property(h_error);
    mesh.add_property(h_heap_position);
    mesh.add_property(h_errorMatrix);
    mesh.add_property(h_collapseTarget);
    if (settings.check_folding_triangles == true) mesh.add_property(h_folding_triangles_edge);
    if (settings.check_self_intersections == true) mesh.add_property(h_collapse_self_intersection);
    if (settings.only_collapse_roughly_parallel_borders == true) mesh.add_property(h_parallel_border_edges);
    if (settings.check_interior_angles == true) mesh.add_property(h_interior_angles_undershot);

    // calculate face normals
    mesh.request_face_normals();
    mesh.update_face_normals();

    // activate status so that items can be deleted
    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_halfedge_status();
    mesh.request_vertex_status();

    // Main body of decimation

    // get all the information required about edges
    if (settings.check_self_intersections || settings.check_interior_angles || settings.check_folding_triangles ||
        settings.only_collapse_roughly_parallel_borders)
    {
        MyMesh::EdgeIter e_it, e_end(mesh.edges_end());
        for (e_it = mesh.edges_sbegin(); e_it != e_end; ++e_it)
        {
            update_edge(*e_it);
        }
    }

    // 1. Compute the Q matrices for all the initial vertices
    {
        // --- Calculation of fundamental error matrices per face ---

        // add error fundamental error matrix property to the faces
        mesh.add_property(h_fund_error_mat);

        // iterate through faces and calculate their fundamental error matrix
        MyMesh::FaceIter f_it, f_end(mesh.faces_end());
        MyMesh::FaceVertexCWIter fv_it;

        for (f_it = mesh.faces_sbegin(); f_it != f_end; ++f_it)
        {
            mesh.property(h_fund_error_mat, *f_it) = calculate_fundamental_error_matrix(*f_it);
        }

        // --- Calculation of Q matrices per vertex ---

        // iterate all vertices and calculate their error matrices
        MyMesh::VertexIter v_it, v_end(mesh.vertices_end());
        MyMesh::VertexFaceCWIter vf_it;

        for (v_it = mesh.vertices_sbegin(); v_it != v_end; ++v_it)
        {
            // circulate the faces of the vertex and add the matrices
            mat4 error_mat = mat4::Zero();

            vf_it = mesh.cvf_cwbegin(*v_it);

            for (; vf_it.is_valid(); ++vf_it)
            {
                MyMesh::FaceHandle f = *vf_it;
                error_mat += mesh.property(h_fund_error_mat, f);
            }

            // set the vertex error matrix
            mesh.property(h_errorMatrix, *v_it) = error_mat;
        }

        // remove fundamental error matrices from faces
        mesh.remove_property(h_fund_error_mat);
    }


    // 2. Find a collapse target and the corresponding error for every vertex

    // initialize heap
    OpenMesh::Decimater::DecimaterT<OpenTriangleMesh>::HeapInterface collapseCandidates_HI(mesh, h_error,
                                                                                           h_heap_position);
    collapseCandidates_heap.reset(new DeciHeap(collapseCandidates_HI));
    collapseCandidates_heap->reserve(mesh.n_vertices());

    // do the decimation loop
    {
        // iterate all vertices and find their best decimation partner
        MyMesh::VertexIter v_it, v_end(mesh.vertices_end());

        for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
        {
            collapseCandidates_heap->reset_heap_position(*v_it);
            if (!mesh.status(*v_it).deleted()) update_vertex(*v_it);
        }
    }

    // 3. Iteratively remove the pair of least cost from the heap, contract this pair, and find new best contraction
    // partners for all neighbouring vertices
    {
        MyMesh::VertexHandle current_candidate;

        std::vector<MyMesh::VertexHandle> support;

        // initialize counter variables for in case a specific amount of decimations is requested
        int decimated_vertices = 0;
        int my_max_collapses   = (settings.max_decimations <= 0) ? mesh.n_vertices() : settings.max_decimations;

        while (!collapseCandidates_heap->empty() && decimated_vertices < my_max_collapses)
        {
            current_candidate = collapseCandidates_heap->front();
            collapseCandidates_heap->pop_front();

            // collapse the edge
            MyMesh::HalfedgeHandle collapse_edge = mesh.property(h_collapseTarget, current_candidate);
            if (!custom_is_collapse_legal(collapse_edge))
            {
                // re-calculate the vertex error
                update_vertex(current_candidate);
                continue;
            }

            MyMesh::VertexHandle vh_base   = mesh.from_vertex_handle(collapse_edge);
            MyMesh::VertexHandle vh_target = mesh.to_vertex_handle(collapse_edge);
            mat4 new_error_mat = mesh.property(h_errorMatrix, vh_base) + mesh.property(h_errorMatrix, vh_target);
            mesh.property(h_errorMatrix, vh_target) = new_error_mat;

            // save all the vertices that will have to be updated
            support.clear();
            for (auto vv_it = mesh.vv_cwiter(vh_base); vv_it.is_valid(); ++vv_it)
            {
                support.push_back(*vv_it);
            }

            mesh.collapse(collapse_edge);
            ++decimated_vertices;

            // update the face normals surrounding the target vertex
            MyMesh::VertexFaceCWIter vf_it = mesh.vf_cwiter(vh_target);
            for (; vf_it.is_valid(); ++vf_it)
            {
                mesh.set_normal(*vf_it, mesh.calc_face_normal(*vf_it));
            }

            // update the information of surrounding edges
            {
                update_edges(vh_target);
            }

            // update the newly created vertex, its neighbours and their location in the heap
            for (MyMesh::VertexHandle vh : support)
            {
                assert(!mesh.status(vh).deleted());
                update_vertex(vh);
            }
        }
    }

    // Post-computation

    collapseCandidates_heap.reset();

    mesh.delete_isolated_vertices();
    mesh.garbage_collection();

    // remove properties
    mesh.remove_property(h_error);
    mesh.remove_property(h_heap_position);
    mesh.remove_property(h_errorMatrix);
    mesh.remove_property(h_collapseTarget);
    if (settings.check_folding_triangles == true) mesh.remove_property(h_folding_triangles_edge);
    if (settings.check_self_intersections == true) mesh.remove_property(h_collapse_self_intersection);
    if (settings.only_collapse_roughly_parallel_borders == true) mesh.remove_property(h_parallel_border_edges);
    if (settings.check_interior_angles == true) mesh.remove_property(h_interior_angles_undershot);


    // deactivate status
    mesh.release_face_status();
    mesh.release_edge_status();
    mesh.release_vertex_status();

    mesh_in = mesh;
}

// --- PRIVATE ---

bool Saiga::QuadricDecimater::check_minimal_interior_angles_undershot(MyMesh::HalfedgeHandle collapse_edge)
{
    MyMesh::FaceHandle collapse_face_1(mesh.face_handle(collapse_edge));
    MyMesh::FaceHandle collapse_face_2(mesh.opposite_face_handle(collapse_edge));

    float a, b, c;  // side lengths

    MyMesh::VertexHandle vh1;
    MyMesh::VertexHandle vh2;
    MyMesh::VertexHandle vh3;
    OpenMesh::Vec3f v1;
    OpenMesh::Vec3f v2;
    OpenMesh::Vec3f v3;

    MyMesh::VFCWIter vf_iter = mesh.vf_cwiter(mesh.from_vertex_handle(collapse_edge));
    MyMesh::FVCCWIter fv_iter;

    for (; vf_iter.is_valid(); ++vf_iter)
    {
        // the faces that disappear in the collapse don't matter
        if (*vf_iter == collapse_face_1 || *vf_iter == collapse_face_2) continue;

        // find the vertices of the resulting face after a collapse
        fv_iter = mesh.fv_ccwbegin(*vf_iter);

        vh1 = *fv_iter;
        ++fv_iter;
        vh2 = *fv_iter;
        ++fv_iter;
        vh3 = *fv_iter;

        if (vh1 == mesh.from_vertex_handle(collapse_edge))
            vh1 = mesh.to_vertex_handle(collapse_edge);
        else if (vh2 == mesh.from_vertex_handle(collapse_edge))
            vh2 = mesh.to_vertex_handle(collapse_edge);
        else if (vh3 == mesh.from_vertex_handle(collapse_edge))
            vh3 = mesh.to_vertex_handle(collapse_edge);

        v1 = mesh.point(vh1);
        v2 = mesh.point(vh2);
        v3 = mesh.point(vh3);

        // get the side lenght from the vertices
        a = (v1 - v2).length();
        b = (v2 - v3).length();
        c = (v3 - v1).length();

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

bool Saiga::QuadricDecimater::check_collapse_self_intersection(MyMesh::HalfedgeHandle collapse_edge)
{
    MyMesh::FaceHandle collapse_face_1(mesh.face_handle(collapse_edge));
    MyMesh::FaceHandle collapse_face_2(mesh.opposite_face_handle(collapse_edge));

    OpenMesh::Vec3f pre_collapse_normal;
    OpenMesh::Vec3f post_collapse_normal;

    MyMesh::VertexHandle vh1;
    MyMesh::VertexHandle vh2;
    MyMesh::VertexHandle vh3;
    OpenMesh::Vec3f v1;
    OpenMesh::Vec3f v2;
    OpenMesh::Vec3f v3;

    MyMesh::VFCCWIter vf_iter = mesh.vf_ccwiter(mesh.from_vertex_handle(collapse_edge));
    MyMesh::FVCCWIter fv_iter;

    for (; vf_iter.is_valid(); ++vf_iter)
    {
        // the faces that disappear in the collapse don't matter
        if (*vf_iter == collapse_face_1 || *vf_iter == collapse_face_2) continue;

        pre_collapse_normal = mesh.normal(*vf_iter).normalized();

        // find the vertices of the resulting face after a collapse
        fv_iter = mesh.fv_ccwbegin(*vf_iter);

        vh1 = *fv_iter;
        ++fv_iter;
        vh2 = *fv_iter;
        ++fv_iter;
        vh3 = *fv_iter;

        if (vh1 == mesh.from_vertex_handle(collapse_edge))
            vh1 = mesh.to_vertex_handle(collapse_edge);
        else if (vh2 == mesh.from_vertex_handle(collapse_edge))
            vh2 = mesh.to_vertex_handle(collapse_edge);
        else if (vh3 == mesh.from_vertex_handle(collapse_edge))
            vh3 = mesh.to_vertex_handle(collapse_edge);

        v1 = mesh.point(vh1);
        v2 = mesh.point(vh2);
        v3 = mesh.point(vh3);

        // get the normal from those vertices
        OpenMesh::Vec3f edge1 = v2 - v1;
        OpenMesh::Vec3f edge2 = v3 - v1;
        post_collapse_normal  = cross(edge1, edge2).normalized();

        // if the angle between old and new normal is too great, the triangle probably gets flipped
        if (dot(pre_collapse_normal, post_collapse_normal) < 0.5)
        {  // approximately 60 degree
            return true;
        }
    }

    return false;
}

bool Saiga::QuadricDecimater::custom_is_collapse_legal(MyMesh::HalfedgeHandle v0v1)
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
    MyMesh::HalfedgeHandle v1v0(mesh.opposite_halfedge_handle(v0v1));  ///< Reverse halfedge
    MyMesh::VertexHandle v0(mesh.to_vertex_handle(v1v0));              ///< Vertex to be removed
    MyMesh::VertexHandle v1(mesh.to_vertex_handle(v0v1));              ///< Remaining vertex
    MyMesh::FaceHandle fl(mesh.face_handle(v0v1));                     ///< Left face
    MyMesh::FaceHandle fr(mesh.face_handle(v1v0));                     ///< Right face
    MyMesh::VertexHandle vl;                                           ///< Left vertex
    MyMesh::VertexHandle vr;                                           ///< Right vertex

    MyMesh::HalfedgeHandle vlv1, v0vl, vrv0, v1vr;  ///< Outer remaining halfedges

    // get vl
    if (fl.is_valid())
    {
        vlv1 = mesh.next_halfedge_handle(v0v1);
        v0vl = mesh.next_halfedge_handle(vlv1);
        vl   = mesh.to_vertex_handle(vlv1);
        vlv1 = mesh.opposite_halfedge_handle(vlv1);
        v0vl = mesh.opposite_halfedge_handle(v0vl);
    }

    // get vr
    if (fr.is_valid())
    {
        vrv0 = mesh.next_halfedge_handle(v1v0);
        v1vr = mesh.next_halfedge_handle(vrv0);
        vr   = mesh.to_vertex_handle(vrv0);
        vrv0 = mesh.opposite_halfedge_handle(vrv0);
        v1vr = mesh.opposite_halfedge_handle(v1vr);
    }

    // -------------------------------------
    // check if things are legal to collapse
    // -------------------------------------

    // locked ?
    if (mesh.status(v0).locked()) return false;

    // this test checks:
    // is v0v1 deleted?
    // is v0 deleted?
    // is v1 deleted?
    // are both vlv0 and v1vl boundary edges?
    // are both v0vr and vrv1 boundary edges?
    // are vl and vr equal or both invalid?
    // one ring intersection test
    // edge between two boundary vertices should be a boundary edge
    if (!mesh.is_collapse_ok(v0v1)) return false;

    // my modification
    if (settings.check_self_intersections == true && mesh.property(h_collapse_self_intersection, v0v1) == true)
        return false;

    // my modification
    if (settings.only_collapse_roughly_parallel_borders == true && mesh.is_boundary(mesh.edge_handle(v0v1)))
    {
        if (mesh.property(h_parallel_border_edges, v0v1) == false)
        {
            return false;
        }
    }

    if (vl.is_valid() && vr.is_valid() && mesh.find_halfedge(vl, vr).is_valid() && mesh.valence(vl) == 3 &&
        mesh.valence(vr) == 3)
    {
        return false;
    }
    //--- feature test ---

    if (mesh.status(v0).feature() && !mesh.status(mesh.edge_handle(v0v1)).feature()) return false;

    //--- test boundary cases ---
    if (mesh.is_boundary(v0))
    {
        // don't collapse a boundary vertex to an inner one
        if (!mesh.is_boundary(v1)) return false;

        // only one one ring intersection
        if (vl.is_valid() && vr.is_valid()) return false;
    }


    // there have to be at least 2 incident faces at v0
    if (mesh.cw_rotated_halfedge_handle(mesh.cw_rotated_halfedge_handle(v0v1)) == v0v1) return false;

    // collapse passed all tests -> ok
    return true;
}

Saiga::mat4 Saiga::QuadricDecimater::calculate_fundamental_error_matrix(const MyMesh::FaceHandle fh)
{
    // https://en.wikipedia.org/wiki/Plane_(geometry)#Describing_a_plane_through_three_points

    MyMesh::FaceVertexCWIter fv_it = mesh.cfv_cwbegin(fh);
    MyMesh::VertexHandle vh0       = *fv_it;
    MyMesh::VertexHandle vh1       = *(++fv_it);
    MyMesh::VertexHandle vh2       = *(++fv_it);
    OpenMesh::Vec3f v0             = mesh.point(vh0);
    OpenMesh::Vec3f v1             = mesh.point(vh1);
    OpenMesh::Vec3f v2             = mesh.point(vh2);

    OpenMesh::Vec3f normal = OpenMesh::cross((v1 - v0), (v2 - v0));

    float area = normal.norm();
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

bool Saiga::QuadricDecimater::check_for_folding_triangles(const MyMesh::EdgeHandle edge)
{
    // if any of the triangles that are altered by this collapse have normals with greater angles than 60° to each
    // other, don't consider it
    // --> loop through all faces of the vertices of the cendidat edge and check their normals against each other

    std::set<OpenMesh::Vec3f> normals;

    // collect all normals
    MyMesh::VertexFaceCWIter fcw_it  = mesh.cvf_cwbegin(mesh.to_vertex_handle(mesh.halfedge_handle(edge, 0)));
    MyMesh::VertexFaceCWIter fcw_it2 = mesh.cvf_cwbegin(mesh.from_vertex_handle(mesh.halfedge_handle(edge, 0)));

    while (fcw_it.is_valid())
    {
        if (mesh.status(*fcw_it).deleted()) continue;
        normals.insert(mesh.normal(*fcw_it));
        ++fcw_it;
    }
    while (fcw_it2.is_valid())
    {
        if (mesh.status(*fcw_it2).deleted()) continue;
        normals.insert(mesh.normal(*fcw_it2));
        ++fcw_it2;
    }

    // check them against each other
    std::vector<OpenMesh::Vec3f> normals_vec(normals.begin(), normals.end());

    for (int i = 0; i < normals_vec.size(); ++i)
    {
        OpenMesh::Vec3f normal_1 = normals_vec[i];

        for (int j = i + 1; j < normals_vec.size(); ++j)
        {
            // dot(A,B) = |A| * |B| * cos(angle)
            // which can be rearranged to
            // angle = arccos(dot(A, B) / (|A| * |B|)).
            float angle = OpenMesh::dot(normal_1, normals_vec[j]);
            angle       = acos(angle);

            // pi<float>() / 3.0f rad = 60 degree
            // pi<float>() / 3.0f = 1.0471975512f
            if (angle > 1.0471975512f)
            {
                return true;
            }
        }
    }

    return false;
}

bool Saiga::QuadricDecimater::check_edge_parallelity(const OpenMesh::Vec3f v0, const OpenMesh::Vec3f v1,
                                                     const OpenMesh::Vec3f v2)
{
    OpenMesh::Vec3f edge0((v0 - v1).normalize()), edge1((v1 - v2).normalize());

    // check the edges against each other
    float dot = OpenMesh::dot(edge0, edge1);

    if (abs(dot) < 0.95) return false;

    return true;
}

float Saiga::QuadricDecimater::calculate_collapse_error(const MyMesh::HalfedgeHandle candidat_edge)
{
    MyMesh::VertexHandle vh1 = mesh.from_vertex_handle(candidat_edge);
    MyMesh::VertexHandle vh2 = mesh.to_vertex_handle(candidat_edge);
    vec4 v2                  = vec4(mesh.point(vh2)[0], mesh.point(vh2)[1], mesh.point(vh2)[2], 1.0f);

    float error = dot(v2, (mesh.property(h_errorMatrix, vh1) + mesh.property(h_errorMatrix, vh2)) * v2);

    if (settings.check_folding_triangles == true)
    {
        if (mesh.property(h_folding_triangles_edge, mesh.edge_handle(candidat_edge)) == true)
        {
            error *= 10.0f;
        }
    }

    // undershooting interior angles gives a strong penalty
    if (settings.check_interior_angles == true)
    {
        if (mesh.property(h_interior_angles_undershot, candidat_edge) == true)
        {
            error *= 10.0f;
        }
    }

    // divide error by squared distance
    OpenMesh::Vec3f a             = mesh.point(vh1);
    OpenMesh::Vec3f b             = mesh.point(vh2);
    OpenMesh::Vec3f mid           = (a + b) / 2.0f;
    float edge_to_camera_distance = mid.length();
    error                         = error / pow(edge_to_camera_distance, 2);

    return error;
}

float Saiga::QuadricDecimater::find_collapse_partner(const MyMesh::VertexHandle vh,
                                                     MyMesh::HalfedgeHandle& collapse_edge)
{
    MyMesh::VertexOHalfedgeCWIter he_it = mesh.cvoh_cwbegin(vh);
    float error                         = std::numeric_limits<float>::max();

    float error_tmp;
    for (; he_it.is_valid(); ++he_it)
    {
        if (!custom_is_collapse_legal(*he_it))
        {
            error_tmp = settings.quadricMaxError + 999.9f;
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

void Saiga::QuadricDecimater::update_vertex(const MyMesh::VertexHandle vh)
{
    // calculate new best fit
    MyMesh::HalfedgeHandle heh;
    float error = find_collapse_partner(vh, heh);

    // target found -> put vertex on heap
    if (heh.is_valid() && error <= settings.quadricMaxError)
    {
        // update the information in the vertex
        mesh.property(h_collapseTarget, vh) = heh;
        mesh.property(h_error, vh)          = error;

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

        mesh.property(h_error, vh) = -1;
    }
}

void Saiga::QuadricDecimater::update_edge(const MyMesh::EdgeHandle eh)
{
    MyMesh::HalfedgeHandle heh0 = mesh.halfedge_handle(eh, 0);
    MyMesh::HalfedgeHandle heh1 = mesh.halfedge_handle(eh, 1);

    if (settings.check_self_intersections)
    {
        mesh.property(h_collapse_self_intersection, heh0) = check_collapse_self_intersection(heh0);
        mesh.property(h_collapse_self_intersection, heh1) = check_collapse_self_intersection(heh1);
    }

    if (settings.check_interior_angles)
    {
        mesh.property(h_interior_angles_undershot, heh0) = check_minimal_interior_angles_undershot(heh0);
        mesh.property(h_interior_angles_undershot, heh1) = check_minimal_interior_angles_undershot(heh1);
    }

    if (settings.check_folding_triangles) mesh.property(h_folding_triangles_edge, eh) = check_for_folding_triangles(eh);

    if (settings.only_collapse_roughly_parallel_borders)
    {
        // check if it's a border
        if (mesh.is_boundary(eh))
        {
            // find the right halfedge
            bool border_at_1 = mesh.is_boundary(heh1);

            OpenMesh::Vec3f v0, v1, v2, v3;

            if (border_at_1)
            {
                v0 = mesh.point(mesh.from_vertex_handle(mesh.prev_halfedge_handle(heh1)));
                v1 = mesh.point(mesh.from_vertex_handle(heh1));
                v2 = mesh.point(mesh.to_vertex_handle(heh1));
                v3 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(heh1)));

                mesh.property(h_parallel_border_edges, heh0) = check_edge_parallelity(v3, v2, v1);
                mesh.property(h_parallel_border_edges, heh1) = check_edge_parallelity(v0, v1, v2);
            }
            else
            {
                v0 = mesh.point(mesh.from_vertex_handle(mesh.prev_halfedge_handle(heh0)));
                v1 = mesh.point(mesh.from_vertex_handle(heh0));
                v2 = mesh.point(mesh.to_vertex_handle(heh0));
                v3 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(heh0)));

                mesh.property(h_parallel_border_edges, heh0) = check_edge_parallelity(v0, v1, v2);
                mesh.property(h_parallel_border_edges, heh1) = check_edge_parallelity(v3, v2, v1);
            }
        }
    }
}

void Saiga::QuadricDecimater::update_edges(const MyMesh::VertexHandle vh)
{
    auto it = mesh.ve_begin(vh);
    while (it.is_valid())
    {
        update_edge(*it);
        ++it;
    }
}
