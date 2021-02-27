/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "image_triangulator.h"

namespace Saiga
{
// ------------------- naive triangulation -------------------

// ----  PUBLIC  ----
SimpleTriangulator::SimpleTriangulator(const Settings& settings_in) : settings(settings_in) {}

void SimpleTriangulator::triangulate_image(ImageView<const float> depthImage, OpenTriangleMesh& mesh_out)
{
    // create an ImageView to hold the vertexHandles that are added to the mesh
    std::vector<OpenMesh::VertexHandle> pixel_vertexHandles(depthImage.height * depthImage.width);
    ImageView<OpenMesh::VertexHandle> pixel_vertexHandles_iV(depthImage.height, depthImage.width,
                                                             pixel_vertexHandles.data());

    // unproject and add all valid vertices to the mesh
    add_vertices_to_mesh(depthImage, pixel_vertexHandles_iV, mesh_out);

    // triangulate the added vertices
    completely_triangulate(mesh_out, pixel_vertexHandles_iV);

    // Post-computation

    // activate status so that items can be deleted
    mesh_out.request_face_status();
    mesh_out.request_edge_status();
    mesh_out.request_vertex_status();

    // delete unnecessarily added vertices
    mesh_out.delete_isolated_vertices();
    mesh_out.garbage_collection();

    // calculate normals for future use
    mesh_out.request_face_normals();
    mesh_out.update_face_normals();

    // release the status stuff
    mesh_out.release_face_status();
    mesh_out.release_edge_status();
    mesh_out.release_vertex_status();
}

// ----  PRIVATE  ----
void SimpleTriangulator::add_face_to_mesh(MyMesh& mesh, MyMesh::VertexHandle vh1, MyMesh::VertexHandle vh2,
                                          MyMesh::VertexHandle vh3)
{
    if (mesh.point(vh1)[2] == settings.broken_values || mesh.point(vh2)[2] == settings.broken_values ||
        mesh.point(vh3)[2] == settings.broken_values)
        return;

    mesh.add_face(vh1, vh2, vh3);
}

void SimpleTriangulator::add_vertices_to_mesh(ImageView<const float> depthImageView,
                                              ImageView<OpenMesh::VertexHandle> pixel_vertices, MyMesh& mesh)
{
    int height = depthImageView.height;
    int width  = depthImageView.width;

    SAIGA_ASSERT(pixel_vertices.height == height);
    SAIGA_ASSERT(pixel_vertices.width == width);

    DepthProcessor2::Settings ip_settings;
    ip_settings.cameraParameters = settings.cameraParameters;
    DepthProcessor2 ip(ip_settings);

    std::vector<vec3> unprojected_image_vector(height * width);
    ImageView<vec3> unprojected_image(height, width, unprojected_image_vector.data());
    ip.unproject_depth_image(depthImageView, unprojected_image);

    // iterate the pixels and add the not broken ones to the mesh
    OpenMesh::VertexHandle vh;
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            // add the vertex and color it
            if (depthImageView(h, w) != settings.broken_values)
            {
                vec3 p = unprojected_image(h, w);
                vh     = mesh.add_vertex(OpenMesh::Vec3f(p[0], p[1], p[2]));
                if (mesh.has_vertex_colors())
                {
                    mesh.set_color(vh,
                                   OpenMesh::Vec3f(float(h) / height, float(w) / width, depthImageView(h, w) / 5.0f));
                }
            }
            else
            {
                vh = OpenMesh::VertexHandle(-1);
            }

            pixel_vertices(h, w) = vh;
        }
    }
}

void SimpleTriangulator::completely_triangulate(MyMesh& mesh, ImageView<OpenMesh::VertexHandle> pixel_vertexHandles)
{
    // I add the faces on the lower right per vertex (the "4th quadrant" basically)
    for (int h = 0; h < pixel_vertexHandles.height - 1; ++h)
    {
        for (int w = 0; w < pixel_vertexHandles.width - 1; ++w)
        {
            OpenMesh::VertexHandle vh            = pixel_vertexHandles(h, w);
            OpenMesh::VertexHandle vh_right      = pixel_vertexHandles(h, w + 1);
            OpenMesh::VertexHandle vh_down       = pixel_vertexHandles(h + 1, w);
            OpenMesh::VertexHandle vh_right_down = pixel_vertexHandles(h + 1, w + 1);

            // check if vh is broken
            if (!vh.is_valid())
            {
                // yes it is, so check if the remaining triangle is not broken
                if (vh_down.is_valid() && vh_right_down.is_valid() && vh_right.is_valid())
                {
                    mesh.add_face(vh_down, vh_right_down, vh_right);
                }
                continue;
            }

            // check if vh_right is broken
            if (!vh_right.is_valid())
            {
                // yes it is, so check if the remaining triangle is not broken
                if (vh.is_valid() && vh_down.is_valid() && vh_right_down.is_valid())
                {
                    mesh.add_face(vh, vh_down, vh_right_down);
                }
                continue;
            }

            // check if vh_down is broken
            if (!vh_down.is_valid())
            {
                // yes it is, so check if the remaining triangle is not broken
                if (vh.is_valid() && vh_right_down.is_valid() && vh_right.is_valid())
                {
                    mesh.add_face(vh, vh_right_down, vh_right);
                }
                continue;
            }

            // check if vh is broken
            if (!vh_right_down.is_valid())
            {
                // yes it is, so check if the remaining triangle is not broken
                if (vh.is_valid() && vh_down.is_valid() && vh_right.is_valid())
                {
                    mesh.add_face(vh, vh_down, vh_right);
                }
                continue;
            }

            // no vertex is broken
            // depending on the length of the diagonals, we triangulate so that the shorter one is chosen
            float diagonal_length_1 = (mesh.point(vh) - mesh.point(vh_right_down)).length();
            float diagonal_length_2 = (mesh.point(vh_right) - mesh.point(vh_down)).length();

            if (diagonal_length_1 <= diagonal_length_2)
            {
                mesh.add_face(vh, vh_down, vh_right_down);
                mesh.add_face(vh, vh_right_down, vh_right);
            }
            else
            {
                mesh.add_face(vh, vh_down, vh_right);
                mesh.add_face(vh_down, vh_right_down, vh_right);
            }
        }
    }
}

// ------------------- RQT triangulation -------------------

// ----  PUBLIC  ----
RQT_Triangulator::RQT_Triangulator(const Settings& settings_in) : settings(settings_in)
{
    RQT_side_len = get_closest_RQT_side_length(settings_in.image_height, settings_in.image_width);
    create_dependency_graph();
}

void RQT_Triangulator::triangulate_image(ImageView<const float> depthImage, OpenTriangleMesh& mesh_out)
{
    DepthProcessor2::Settings ip_settings;
    ip_settings.cameraParameters = settings.cameraParameters;
    DepthProcessor2 ip(ip_settings);

    // unproject the image
    std::vector<vec3> unprojected_image_vector(settings.image_height * settings.image_width);
    ImageView<vec3> unprojected_image(settings.image_height, settings.image_width, unprojected_image_vector.data());
    ip.unproject_depth_image(depthImage, unprojected_image);

    // create an ImageView for the selected vertices
    std::vector<MyMesh::VertexHandle> selected_vertices_vector(settings.image_height * settings.image_width);
    ImageView<MyMesh::VertexHandle> selected_vertices(settings.image_height, settings.image_width,
                                                      selected_vertices_vector.data());
    select_vertices_for_RQT(mesh_out, unprojected_image, selected_vertices);

    // actually triangulate the mesh using all the selected vertices
    triangulate_RQT_selected_vertices(mesh_out, selected_vertices);

    // --- post computation ---

    // activate status so that items can be deleted
    mesh_out.request_face_status();
    mesh_out.request_edge_status();
    mesh_out.request_vertex_status();

    // cleaning up the result (not calling these can kill circulators later on for some reason)
    mesh_out.delete_isolated_vertices();
    mesh_out.garbage_collection();

    mesh_out.release_face_status();
    mesh_out.release_edge_status();
    mesh_out.release_vertex_status();
}

// ----  PRIVATE  ----
void RQT_Triangulator::add_face_to_mesh(MyMesh& mesh, MyMesh::VertexHandle vh1, MyMesh::VertexHandle vh2,
                                        MyMesh::VertexHandle vh3)
{
    if (mesh.point(vh1)[2] == settings.broken_values || mesh.point(vh2)[2] == settings.broken_values ||
        mesh.point(vh3)[2] == settings.broken_values)
        return;

    mesh.add_face(vh1, vh2, vh3);
}

int RQT_Triangulator::get_closest_RQT_side_length(int height, int width)
{
    int max_image_side_length = std::max(height, width);
    // the width has to be a power of 2 with 1 added on top
    if (((max_image_side_length - 1) & (max_image_side_length - 2)) == 0)
    {
        // The width is already as needed
    }
    else if ((max_image_side_length & (max_image_side_length - 1)) == 0)
    {
        // The width is a power of 2, so only 1 needs to be added
        ++max_image_side_length;
    }
    else
    {
        // The width is neither a power of 2 nor a power of two plus 1.
        // --> Find the next power of two and add 1
        max_image_side_length = log2(max_image_side_length);
        ++max_image_side_length;
        max_image_side_length = pow(2, max_image_side_length);
        ++max_image_side_length;
    }
    return max_image_side_length;
}

void RQT_Triangulator::create_dependency_graph()
{
    // level 0 isn't captured by log, so add it manually
    int total_levels_without_zero = log2(RQT_side_len - 1);
    int total_levels              = total_levels_without_zero + 1;

    dependency_graph_vector = std::vector<std::vector<Point2D>>(RQT_side_len * RQT_side_len);
    ImageView<std::vector<Point2D>> dependency_graph =
        ImageView<std::vector<Point2D>>(RQT_side_len, RQT_side_len, dependency_graph_vector.data());

    // add something to the level 0 vertices so they will be skipped in the upcoming loop over the other levels
    dependency_graph(0, 0)                               = std::vector<Point2D>(1, Point2D(0, 0));
    dependency_graph(RQT_side_len - 1, 0)                = std::vector<Point2D>(1, Point2D(0, 0));
    dependency_graph(0, RQT_side_len - 1)                = std::vector<Point2D>(1, Point2D(0, 0));
    dependency_graph(RQT_side_len - 1, RQT_side_len - 1) = std::vector<Point2D>(1, Point2D(0, 0));

    for (int level = 1; level < total_levels; ++level)
    {
        int h_l   = pow(2, level);
        int d_l   = (RQT_side_len - 1) / pow(2, level);
        int hl_dl = h_l * d_l;

        // there are 5 vertices per quad in a quadtree (except for level 0):
        // top, bottom, left, right, center
        //
        // the following loop traverses them in this order:
        // top/bottom, top/bottom, ...
        // right/left, center, right/left, center, ...
        // top/bottom, top/bottom, ...
        // ...

        // true: top/bottom-row; false: right/left/center-row
        bool row_flag = true;
        // true: right/left; false: center
        bool side_center_flag = true;
        // true: left down and right up; false: left up and right down;
        // this should be true for the first vertex. However since there is a
        // flip before that, it is set to false here
        bool diagonal_dependency_dir = false;

        for (int y = 0; y <= hl_dl; y += d_l)
        {
            for (int x = 0; x <= hl_dl; x += d_l)
            {
                if (x == 0)
                {  // all side-center-rows start and end with side vertices
                    side_center_flag = true;
                }
                if (x == 0 && row_flag == false)
                {
                    // side-center-rows finish with the same center-diagonal-dependency
                    // as the next one starts. this flip neutralizes the one that comes
                    // after every center vertex handling
                    diagonal_dependency_dir = !diagonal_dependency_dir;
                }
                if (dependency_graph(y, x).size() > 0)
                {
                    // the current vertex is already part of a previously handled level
                    continue;
                }

                // all dependency_graph of vertices are 2 other vertices (or 1 if the vertex is at the edge)
                int x1, x2, y1, y2;

                if (row_flag)
                {  // true: top/bottom-row
                    x1 = x;
                    x2 = x;
                    y1 = y + d_l;
                    y2 = y - d_l;

                    // add the dependency_graph if they don't point outside of the whole quadtree
                    if (y1 < RQT_side_len) dependency_graph(y, x).push_back(Point2D(x1, y1));
                    if (y2 >= 0) dependency_graph(y, x).push_back(Point2D(x2, y2));
                }
                else
                {  // false: right/left/center-row
                    if (side_center_flag)
                    {  // true: right/left
                        x1 = x + d_l;
                        x2 = x - d_l;
                        y1 = y;
                        y2 = y;

                        // add the dependency_graph if they don't point outside of the whole quadtree
                        if (x1 < RQT_side_len) dependency_graph(y, x).push_back(Point2D(x1, y1));
                        if (x2 >= 0) dependency_graph(y, x).push_back(Point2D(x2, y2));
                    }
                    else
                    {  // false: center
                        if (diagonal_dependency_dir)
                        {  // true: left down and right up
                            x1 = x + d_l;
                            x2 = x - d_l;
                            y1 = y - d_l;
                            y2 = y + d_l;

                            // add the dependency_graph if they don't point outside of the whole quadtree
                            if (x1 < RQT_side_len && y1 >= 0) dependency_graph(y, x).push_back(Point2D(x1, y1));
                            if (x2 >= 0 && y2 < RQT_side_len) dependency_graph(y, x).push_back(Point2D(x2, y2));
                        }
                        else
                        {  // false: left up and right down
                            x1 = x + d_l;
                            x2 = x - d_l;
                            y1 = y + d_l;
                            y2 = y - d_l;

                            // add the dependency_graph if they don't point outside of the whole quadtree
                            if (x1 < RQT_side_len && y1 < RQT_side_len)
                                dependency_graph(y, x).push_back(Point2D(x1, y1));
                            if (x2 >= 0 && y2 >= 0) dependency_graph(y, x).push_back(Point2D(x2, y2));
                        }
                        // diagonal dependency_graph are always alternating
                        diagonal_dependency_dir = !diagonal_dependency_dir;
                    }
                    // flip the flag since the next loop is the other kind of column
                    side_center_flag = !side_center_flag;
                }
            }
            // flip the flag since the next loop is the other kind of row
            row_flag = !row_flag;
        }
    }
    // clear tmp dependency_graph from level 0
    dependency_graph(0, 0).clear();
    dependency_graph(RQT_side_len - 1, 0).clear();
    dependency_graph(0, RQT_side_len - 1).clear();
    dependency_graph(RQT_side_len - 1, RQT_side_len - 1).clear();
}

void RQT_Triangulator::resolve_dependencies(MyMesh& mesh, ImageView<const vec3> unprojected_image, Point2D vertex,
                                            ImageView<MyMesh::VertexHandle> selected_vertices)
{
    // create a heap of vertices left to add to the selected_vertices
    std::vector<Point2D> vertex_heap;
    vertex_heap.push_back(vertex);
    while (!vertex_heap.empty())
    {
        // get next index to work with
        Point2D curr_vertex = vertex_heap.back();
        vertex_heap.pop_back();
        int x = curr_vertex.first;
        int y = curr_vertex.second;

        // check whether the vertex lies within the bounds of the image
        if (y >= settings.image_height || x >= settings.image_width)
        {
            continue;
        }

        // check if the vertex already is in the selected vertices (this means that its dependencies are already in
        // there too) don't add further dependencies of this vertex if it already existed
        if (selected_vertices(y, x) != MyMesh::VertexHandle()) continue;

        // add the current vertex to the selected vertices
        vec3 p                  = unprojected_image(y, x);
        selected_vertices(y, x) = mesh.add_vertex(OpenMesh::Vec3f(p[0], p[1], p[2]));
        if (mesh.has_vertex_colors())
        {
            mesh.set_color(selected_vertices(y, x), OpenMesh::Vec3f((float)y / (float)unprojected_image.height,
                                                                    (float)x / (float)unprojected_image.width,
                                                                    unprojected_image(y, x)[2] / 5.0f));
        }

        // add the vertices dependencies to the heap
        ImageView<std::vector<Point2D>> dependency_graph =
            ImageView<std::vector<Point2D>>(RQT_side_len, RQT_side_len, dependency_graph_vector.data());
        for (int i = 0; i < (int)dependency_graph(y, x).size(); ++i)
        {
            vertex_heap.push_back(dependency_graph(y, x)[i]);
        }
    }
}

void RQT_Triangulator::select_vertices_for_RQT(MyMesh& mesh, ImageView<const vec3> unprojected_image,
                                               ImageView<MyMesh::VertexHandle> selected_vertices)
{
    int image_width  = unprojected_image.width;
    int image_height = unprojected_image.height;

    // level 0 is special --> add it manually (vertices not within the image are ignored by resolve_dependencies)
    resolve_dependencies(mesh, unprojected_image, Point2D(0, 0), selected_vertices);
    resolve_dependencies(mesh, unprojected_image, Point2D(0, RQT_side_len - 1), selected_vertices);
    resolve_dependencies(mesh, unprojected_image, Point2D(RQT_side_len - 1, 0), selected_vertices);
    resolve_dependencies(mesh, unprojected_image, Point2D(RQT_side_len - 1, RQT_side_len - 1), selected_vertices);

    // create a heap of RQT quads whose 4 inner quads need to be checked against the error metric to determine if they
    // should be split or not (I do it this way so I'll know the triangulation). Each entry is a pair of the quads'
    // upper left corners' coordinates (x, y) in the image and the side-length of the quad
    std::vector<std::pair<Point2D, int>> quad_heap;
    quad_heap.push_back(std::pair<Point2D, int>(Point2D(0, 0), RQT_side_len));

    // define variables that will be used for every quad
    int y, x, curr_side_len;
    int x_mid, x_right, y_mid, y_down;
    int step_small, step_med;
    while (!quad_heap.empty())
    {
        // get the current quad-index
        x             = quad_heap.back().first.first;
        y             = quad_heap.back().first.second;
        curr_side_len = quad_heap.back().second;
        quad_heap.pop_back();

        // I check the metric for the next lower level (reason: so that i dont have to save the default triangulation)
        // --> side len 5 is the lowest possible
        if (curr_side_len < 5) continue;

        step_small = curr_side_len / 4;
        step_med   = curr_side_len / 2;
        x_mid      = x + step_med;
        y_mid      = y + step_med;
        x_right    = x + curr_side_len - 1;
        y_down     = y + curr_side_len - 1;

        // get error metric result

        // upper left
        // if the quad covers at most a single pixel-line of the image it can be completely discarded (it can be
        // represented by neighbour quads)
        if (y < image_height - 1 && x < image_width - 1)
        {
            if (  // any part of the quad doesn't cover the image (--> instantly split)
                y_mid >= image_height || x_mid >= image_width ||
                // check whether error threshold is exceeded
                check_quad_error_threshold_exceeded(unprojected_image, 0, settings.RQT_error_threshold, Point2D(x, y),
                                                    Point2D(x, y_mid), Point2D(x_mid, y_mid), Point2D(x_mid, y)))
            {
                // add the quad to the heap for further metric checks
                quad_heap.push_back(std::pair<Point2D, int>(Point2D(x, y), step_med + 1));

                // add the vertices of this split and their dependencies to the selected_vertices
                // up
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_small, y), selected_vertices);
                // left
                resolve_dependencies(mesh, unprojected_image, Point2D(x, y + step_small), selected_vertices);
                // center
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_small, y + step_small),
                                     selected_vertices);
                // right
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med, y + step_small), selected_vertices);
                // down
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_small, y + step_med), selected_vertices);
            }
        }

        // lower left
        if (y_mid < image_height - 1 && x < image_width - 1)
        {
            if (y_down >= image_height || x_mid >= image_width ||
                check_quad_error_threshold_exceeded(unprojected_image, 1, settings.RQT_error_threshold,
                                                    Point2D(x, y_mid), Point2D(x, y_down), Point2D(x_mid, y_down),
                                                    Point2D(x_mid, y_mid)))
            {
                quad_heap.push_back(std::pair<Point2D, int>(Point2D(x, y_mid), step_med + 1));

                // up
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_small, y_mid), selected_vertices);
                // left
                resolve_dependencies(mesh, unprojected_image, Point2D(x, y_mid + step_small), selected_vertices);
                // center
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_small, y_mid + step_small),
                                     selected_vertices);
                // right
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med, y_mid + step_small),
                                     selected_vertices);
                // down
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_small, y_mid + step_med),
                                     selected_vertices);
            }
        }

        // lower right
        if (y_mid < image_height - 1 && x_mid < image_width - 1)
        {
            if (y_down >= image_height || x_right >= image_width ||
                check_quad_error_threshold_exceeded(unprojected_image, 0, settings.RQT_error_threshold,
                                                    Point2D(x_mid, y_mid), Point2D(x_mid, y_down),
                                                    Point2D(x_right, y_down), Point2D(x_right, y_mid)))
            {
                quad_heap.push_back(std::pair<Point2D, int>(Point2D(x_mid, y_mid), step_med + 1));

                // up
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_small, y_mid),
                                     selected_vertices);
                // left
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med, y_mid + step_small),
                                     selected_vertices);
                // center
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_small, y_mid + step_small),
                                     selected_vertices);
                // right
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_med, y_mid + step_small),
                                     selected_vertices);
                // down
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_small, y_mid + step_med),
                                     selected_vertices);
            }
        }

        // upper right
        if (y < image_height - 1 && x_mid < image_width - 1)
        {
            if (y_mid >= image_height || x_right >= image_width ||
                check_quad_error_threshold_exceeded(unprojected_image, 1, settings.RQT_error_threshold,
                                                    Point2D(x_mid, y), Point2D(x_mid, y_mid), Point2D(x_right, y_mid),
                                                    Point2D(x_right, y)))
            {
                quad_heap.push_back(std::pair<Point2D, int>(Point2D(x_mid, y), step_med + 1));

                // up
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_small, y), selected_vertices);
                // left
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med, y + step_small), selected_vertices);
                // center
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_small, y + step_small),
                                     selected_vertices);
                // right
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_med, y + step_small),
                                     selected_vertices);
                // down
                resolve_dependencies(mesh, unprojected_image, Point2D(x + step_med + step_small, y + step_med),
                                     selected_vertices);
            }
        }
    }
}

// ----------- actual triangulation -----------

void RQT_Triangulator::triangulate_RQT_selected_vertices(MyMesh& current_mesh,
                                                         ImageView<OpenTriangleMesh::VertexHandle> selected_vertices)
{
    // heap for quads that need to be triangulated
    // consists of the coordinates of the left upper corner, the side_length of the quad and a bool indicating the
    // triangulation (false: top left to bottom right, true: top right to bottom left)
    std::vector<std::tuple<Point2D, int, bool>> quad_heap;
    quad_heap.push_back(std::tuple<Point2D, int, bool>(Point2D(0, 0), RQT_side_len, false));

    // flags to remember which parts of a quad are handled already. numeration starts with top left and goes in counter
    // clock wise order
    bool child_1, child_2, child_3, child_4;

    MyMesh::VertexHandle curr_center_vh;
    int curr_y, curr_x;
    int curr_max_y, curr_max_x;
    int curr_side_len, next_side_len, step_mid;
    // curr_triangulation == false: top left to bottom right, true: top right to bottom left
    bool curr_triangulation;
    // flags to help determine the correct triangulation
    bool up_exists, left_exists, down_exists, right_exists;
    while (!quad_heap.empty())
    {
        // get current data
        curr_x             = std::get<0>(quad_heap.back()).first;
        curr_y             = std::get<0>(quad_heap.back()).second;
        curr_side_len      = std::get<1>(quad_heap.back());
        curr_max_y         = curr_y + curr_side_len - 1;
        curr_max_x         = curr_x + curr_side_len - 1;
        step_mid           = curr_side_len / 2;
        next_side_len      = step_mid + 1;
        curr_triangulation = std::get<2>(quad_heap.back());
        quad_heap.pop_back();

        // if the current quad is not part of the image it can be discarded
        if (curr_y >= settings.image_height || curr_x >= settings.image_width)
        {
            continue;
        }

        // if the right-most lowest-most vertex is not within the image, the whole quad can be split without thought
        if (curr_max_y >= settings.image_height || curr_max_x >= settings.image_width)
        {
            if (curr_side_len > 3)
            {
                // the quads will be split since at least two of them are out of the images bounds and the others
                // have to fulfill the quadtree restriction
                quad_heap.push_back(std::tuple<Point2D, int, bool>(Point2D(curr_x, curr_y), next_side_len, false));
                quad_heap.push_back(
                    std::tuple<Point2D, int, bool>(Point2D(curr_x, curr_y + step_mid), next_side_len, true));
                quad_heap.push_back(std::tuple<Point2D, int, bool>(Point2D(curr_x + step_mid, curr_y + step_mid),
                                                                   next_side_len, false));
                quad_heap.push_back(
                    std::tuple<Point2D, int, bool>(Point2D(curr_x + step_mid, curr_y), next_side_len, true));
                continue;
            }
            else
            {
                // quads with side len <= 3 (actually only 3) still can't be split
                // --> triangulate them
                if (curr_y == settings.image_height - 1 || curr_x == settings.image_width - 1)
                {
                    // only one row or column of the quad is part of the image --> no triangulation possible
                    continue;
                }
                else
                {
                    // exactly two rows or columns (or both) of the quad are part of the image

                    // left upper sub-quad is safe
                    add_face_to_mesh(current_mesh, selected_vertices(curr_y, curr_x),
                                     selected_vertices(curr_y + step_mid, curr_x),
                                     selected_vertices(curr_y + step_mid, curr_x + step_mid));
                    add_face_to_mesh(current_mesh, selected_vertices(curr_y, curr_x),
                                     selected_vertices(curr_y + step_mid, curr_x + step_mid),
                                     selected_vertices(curr_y, curr_x + step_mid));

                    // at most one will lie within bounds
                    // left lower quad
                    if (curr_y < settings.image_height - 2)
                    {
                        add_face_to_mesh(current_mesh, selected_vertices(curr_max_y, curr_x),
                                         selected_vertices(curr_y + step_mid, curr_x + step_mid),
                                         selected_vertices(curr_y + step_mid, curr_x));
                        add_face_to_mesh(current_mesh, selected_vertices(curr_max_y, curr_x),
                                         selected_vertices(curr_max_y, curr_x + step_mid),
                                         selected_vertices(curr_y + step_mid, curr_x + step_mid));
                    }
                    // right upper quad
                    else if (curr_x < settings.image_width - 2)
                    {
                        add_face_to_mesh(current_mesh, selected_vertices(curr_y + step_mid, curr_x + step_mid),
                                         selected_vertices(curr_y, curr_max_x),
                                         selected_vertices(curr_y, curr_x + step_mid));
                        add_face_to_mesh(current_mesh, selected_vertices(curr_y + step_mid, curr_x + step_mid),
                                         selected_vertices(curr_y + step_mid, curr_max_x),
                                         selected_vertices(curr_y, curr_max_x));
                    }
                    // right lower quad is not in the image for sure
                    continue;
                }
            }
        }

        // if this quads' center is not selected, it can simply be triangulated with two triangles and nothing more
        // needs to be done
        curr_center_vh = selected_vertices(curr_y + step_mid, curr_x + step_mid);
        if (curr_center_vh == MyMesh::VertexHandle() || curr_side_len < 3)
        {
            if (curr_triangulation == false)
            {
                add_face_to_mesh(current_mesh, selected_vertices(curr_y, curr_x),
                                 selected_vertices(curr_max_y, curr_max_x), selected_vertices(curr_y, curr_max_x));
                add_face_to_mesh(current_mesh, selected_vertices(curr_y, curr_x), selected_vertices(curr_max_y, curr_x),
                                 selected_vertices(curr_max_y, curr_max_x));
            }
            else
            {
                add_face_to_mesh(current_mesh, selected_vertices(curr_max_y, curr_x),
                                 selected_vertices(curr_max_y, curr_max_x), selected_vertices(curr_y, curr_max_x));
                add_face_to_mesh(current_mesh, selected_vertices(curr_max_y, curr_x),
                                 selected_vertices(curr_y, curr_max_x), selected_vertices(curr_y, curr_x));
            }
            continue;
        }

        // all the vertex handles I'll probably need for the triangulation later on
        MyMesh::VertexHandle vhc = selected_vertices(curr_y + step_mid, curr_x + step_mid);
        MyMesh::VertexHandle vh1 = selected_vertices(curr_y, curr_x);          // 1----4
        MyMesh::VertexHandle vh2 = selected_vertices(curr_max_y, curr_x);      // |    |
        MyMesh::VertexHandle vh3 = selected_vertices(curr_max_y, curr_max_x);  // |    |
        MyMesh::VertexHandle vh4 = selected_vertices(curr_y, curr_max_x);      // 2----3

        // check if any of the quads needs to be checked deeper and if so push it to the heap
        up_exists    = (selected_vertices(curr_y, curr_x + step_mid) != MyMesh::VertexHandle());
        left_exists  = (selected_vertices(curr_y + step_mid, curr_x) != MyMesh::VertexHandle());
        right_exists = (selected_vertices(curr_y + step_mid, curr_max_x) != MyMesh::VertexHandle());
        down_exists  = (selected_vertices(curr_max_y, curr_x + step_mid) != MyMesh::VertexHandle());

        // if the side len is greater than 3 I can check the center vertices of the smaller quads (else they don't even
        // exist). If a center vertex exists I add the respective quad to the heap
        if (up_exists && left_exists && curr_side_len > 3)
        {  // top left
            quad_heap.push_back(std::tuple<Point2D, int, bool>(Point2D(curr_x, curr_y), next_side_len, false));
            child_1 = true;
        }
        else
        {
            child_1 = false;
        }
        if (left_exists && down_exists && curr_side_len > 3)
        {  // bottom left
            quad_heap.push_back(
                std::tuple<Point2D, int, bool>(Point2D(curr_x, curr_y + step_mid), next_side_len, true));
            child_2 = true;
        }
        else
        {
            child_2 = false;
        }
        if (down_exists && right_exists && curr_side_len > 3)
        {  // bottom right
            quad_heap.push_back(
                std::tuple<Point2D, int, bool>(Point2D(curr_x + step_mid, curr_y + step_mid), next_side_len, false));
            child_3 = true;
        }
        else
        {
            child_3 = false;
        }
        if (right_exists && up_exists && curr_side_len > 3)
        {  // top right
            quad_heap.push_back(
                std::tuple<Point2D, int, bool>(Point2D(curr_x + step_mid, curr_y), next_side_len, true));
            child_4 = true;
        }
        else
        {
            child_4 = false;
        }

        // do the checking for all the triangles not added to the heap.
        // there can be 0 - 2 such unhandled triangles for all 4 directions

        // check if the top needs 0, 1 or 2 triangles
        if (!up_exists)
        {
            add_face_to_mesh(current_mesh, vh1, vhc, vh4);
        }
        else
        {
            // triangulate current_mesh respective triangles if they are not handled by a recursive call
            if (!child_1)
            {  // top left
                add_face_to_mesh(current_mesh, vh1, vhc, selected_vertices(curr_y, curr_x + step_mid));
            }
            if (!child_4)
            {  // top right
                add_face_to_mesh(current_mesh, vhc, vh4, selected_vertices(curr_y, curr_x + step_mid));
            }
        }

        // also check the left
        if (!left_exists)
        {
            add_face_to_mesh(current_mesh, vh1, vh2, vhc);
        }
        else
        {
            if (!child_1)
            {  // left upper
                add_face_to_mesh(current_mesh, vhc, vh1, selected_vertices(curr_y + step_mid, curr_x));
            }
            if (!child_2)
            {  // left lower
                add_face_to_mesh(current_mesh, vh2, vhc, selected_vertices(curr_y + step_mid, curr_x));
            }
        }

        // also check below
        if (!down_exists)
        {
            add_face_to_mesh(current_mesh, vhc, vh2, vh3);
        }
        else
        {
            if (!child_2)
            {  // bottom left
                add_face_to_mesh(current_mesh, vhc, vh2, selected_vertices(curr_max_y, curr_x + step_mid));
            }
            if (!child_3)
            {  // bottom right
                add_face_to_mesh(current_mesh, vh3, vhc, selected_vertices(curr_max_y, curr_x + step_mid));
            }
        }

        // also check the right
        if (!right_exists)
        {
            add_face_to_mesh(current_mesh, vhc, vh3, vh4);
        }
        else
        {
            if (!child_3)
            {  // right lower
                add_face_to_mesh(current_mesh, vhc, vh3, selected_vertices(curr_y + step_mid, curr_max_x));
            }
            if (!child_4)
            {  // right upper
                add_face_to_mesh(current_mesh, vh4, vhc, selected_vertices(curr_y + step_mid, curr_max_x));
            }
        }
    }
}


// ----------- error calculation stuff -----------

bool RQT_Triangulator::check_metric(const vec3& point, const vec3& triangle_vertex, const vec3& normal, float threshold)
{
    // if the depth of the current pixel is broken, we can split immediately without checking the actual metric
    if (point[2] == settings.broken_values) return true;

    // get the point-to-plane distance
    float dist = dot(normal, point - triangle_vertex);

    // dist / depth^2
    return dist / (point.norm() * point.norm()) > threshold;
}

bool RQT_Triangulator::check_triangle_error_threshold_exceeded(ImageView<const vec3> unprojected_image,
                                                               int triangle_orientation, float threshold, Point2D a,
                                                               Point2D b, Point2D c)
{
    vec3 a_v = unprojected_image(a.second, a.first);
    vec3 b_v = unprojected_image(b.second, b.first);
    vec3 c_v = unprojected_image(c.second, c.first);


    // if the triangle to be checked is represented by broken values, we immediately split
    if (a_v[2] == settings.broken_values || b_v[2] == settings.broken_values || c_v[2] == settings.broken_values)
    {
        return true;
    }

    // compute triangle plane like in QuadricDecimater::calculate_fundamental_error_matrix
    vec3 normal = (b_v - a_v).cross(c_v - a_v);
    normal.normalize();

    int counter = 0;
    switch (triangle_orientation)
    {
        case 0:
            for (int y = a.second; y <= b.second; ++y)
            {
                for (int x = a.first + counter; x <= b.first; ++x)
                {
                    if (check_metric(unprojected_image(y, x), a_v, normal, threshold)) return true;
                }
                ++counter;
            }
            break;
        case 1:
            for (int y = c.second; y >= a.second; --y)
            {
                for (int x = c.first - counter; x >= a.first; --x)
                {
                    if (check_metric(unprojected_image(y, x), a_v, normal, threshold)) return true;
                }
                ++counter;
            }
            break;
        case 2:
            for (int y = a.second; y <= b.second; ++y)
            {
                for (int x = a.first; x <= c.first - counter; ++x)
                {
                    if (check_metric(unprojected_image(y, x), a_v, normal, threshold)) return true;
                }
                ++counter;
            }
            break;
        case 3:
            for (int y = a.second; y >= c.second; --y)
            {
                for (int x = a.first + counter; x <= c.first; ++x)
                {
                    if (check_metric(unprojected_image(y, x), a_v, normal, threshold)) return true;
                }
                ++counter;
            }
            break;
    }
    return false;
}

bool RQT_Triangulator::check_quad_error_threshold_exceeded(ImageView<const vec3> unprojected_image,
                                                           int triangle_orientation, float threshold, Point2D a,
                                                           Point2D b, Point2D c, Point2D d)
{
    // first check whether all pixels of the quad are broken
    // in this case we can stop with the splitting
    bool everything_broken = true;
    for (int y = a.second; y <= b.second; ++y)
    {
        for (int x = a.first; x <= d.first; ++x)
        {
            if (unprojected_image(y, x)[2] != settings.broken_values)
            {
                // actually not broken vertex
                everything_broken = false;
                break;
            }
        }
        if (everything_broken == false) break;
    }
    // don't split any further
    if (everything_broken) return false;

    // regularly check the triangle
    if (triangle_orientation == 0)
    {
        return check_triangle_error_threshold_exceeded(unprojected_image, 0, threshold, a, c, d) ||
               check_triangle_error_threshold_exceeded(unprojected_image, 1, threshold, a, b, c);
    }
    else
    {
        return check_triangle_error_threshold_exceeded(unprojected_image, 2, threshold, a, b, d) ||
               check_triangle_error_threshold_exceeded(unprojected_image, 3, threshold, b, c, d);
    }
}
}  // namespace Saiga
