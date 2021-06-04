/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"



namespace Saiga
{
class Triangulator
{
   public:
    virtual void triangulate_image(ImageView<const float> depthImage, OpenTriangleMesh& mesh_out) = 0;
};

// ------------------- naive triangulation -------------------

class SimpleTriangulator : public Triangulator
{
   public:
    using MyMesh = OpenTriangleMesh;

    struct Settings
    {
        // the value used for pixels that contain failed depth measurements or got discarded
        float broken_values = 0.0f;

        StereoCamera4Base<float> cameraParameters;
    };

    SimpleTriangulator(const Settings& settings_in);

    void triangulate_image(ImageView<const float> depthImage, OpenTriangleMesh& mesh_out) override;

   private:
    Settings settings;

    // takes a mesh and three VertexHandles that belong to the same mesh.
    // Checks whether any of those handles contain a broken value and creates a face if they don't
    void add_face_to_mesh(MyMesh& mesh, MyMesh::VertexHandle vh1, MyMesh::VertexHandle vh2, MyMesh::VertexHandle vh3);

    // unprojects the pixels of the input image and adds them to the mesh in case the respective pixel contains no
    // broken value
    void add_vertices_to_mesh(ImageView<const float> depthImageView, ImageView<OpenMesh::VertexHandle> pixel_vertices,
                              MyMesh& mesh);

    // takes an ImageView of VertexHandles that were added to the mesh (via add_vertices_to_mesh) and creates a simple
    // triangulation
    void completely_triangulate(MyMesh& mesh, ImageView<OpenMesh::VertexHandle> pixel_vertexHandles);
};

// ------------------- RQT triangulation -------------------

// This class is meant for getting one or more images and turning them into meshes
class RQT_Triangulator : public Triangulator
{
   public:
    using MyMesh  = OpenTriangleMesh;
    using Point2D = std::pair<int, int>;

    struct Settings
    {
        // the value used for pixels that contain failed depth measurements or got discarded
        float broken_values       = 0.0f;
        int image_height          = 240;
        int image_width           = 320;
        float RQT_error_threshold = 0.0015f;

        StereoCamera4Base<float> cameraParameters;
    };

    RQT_Triangulator(const Settings& settings_in);

    void triangulate_image(ImageView<const float> depthImage, OpenTriangleMesh& mesh_out) override;

   private:
    Settings settings;
    std::vector<std::vector<Point2D>> dependency_graph_vector;
    int RQT_side_len;

    // takes a mesh and three VertexHandles that belong to the same mesh.
    // Checks whether any of those handles contain a broken value and creates a face if they don't
    void add_face_to_mesh(MyMesh& mesh, MyMesh::VertexHandle vh1, MyMesh::VertexHandle vh2, MyMesh::VertexHandle vh3);

    // gets the next (2^n)+1 that is greater or equal than both settings.image_height and settings.image_width
    int get_closest_RQT_side_length(int height, int width);

    // creates the dependency graph for a quadtree and fills it with dependencies
    // this method refreshes both dependency_graph_vector and dependency_graph
    void create_dependency_graph();

    // -------------- vertex selection --------------

    // adds a vertex and all its dependencies to the selected vertices using the dependency_graph
    void resolve_dependencies(MyMesh& mesh, ImageView<const vec3> unprojected_image, Point2D vertex,
                              ImageView<MyMesh::VertexHandle> selected_vertices);

    // Uses a metric to determine when to split a quadtree, selects the needed
    // vertices, resolves dependencies. This function selects all neccessary vertices for the whole RQT
    void select_vertices_for_RQT(MyMesh& mesh, ImageView<const vec3> unprojected_image,
                                 ImageView<MyMesh::VertexHandle> selected_vertices);

    // -------------- actual triangulation --------------

    // creates a mesh from all VertexHandles given in selected_vertices
    //
    // the resulting mesh is added to current_mesh
    void triangulate_RQT_selected_vertices(MyMesh& current_mesh,
                                           ImageView<OpenTriangleMesh::VertexHandle> selected_vertices);

    // -------------- error calculation stuff --------------

    // computes the distance from point to a plane represented by the vertex triangle_vertex and a normal
    // close to http://www.iquilezles.org/www/articles/triangledistance/triangledistance.htm
    // allows a greater error in the distance, taken from the decimate enhanced error:
    // error /= pow(edge_to_camera_distance, 2)
    bool check_metric(const vec3& point, const vec3& triangle_vertex, const vec3& normal, float threshold);

    // evaluates whether a triangle exceeds a given error threashold or not using the point to plane distance for every
    // pixel
    // triangle_orientation: indicates which of the 4 possible triangulations of a triangle is given. This might
    // reduce the computation time since no general triangle rasterization is used
    //					-------		-------
    //					|\  0 |		| 2  /|
    //					|  \  |		|  /  |
    //					| 1  \|		|/  3 |
    //					-------		-------
    // a, b, c: Points of the triangle. Always given in counter-clock-wise order with "a" being the primarily left most
    // and secondarily upper most point
    bool check_triangle_error_threshold_exceeded(ImageView<const vec3> unprojected_image, int triangle_orientation,
                                                 float threshold, Point2D a, Point2D b, Point2D c);

    // This function checks whether a given quad of a RQT should be divided. The used metric can be seen in
    // check_triangle_error_threshold_exceeded.
    // triangle_orientation stands for the orientations as shown in check_triangle_error_threshold_exceeded
    // a, b, c and d are the corners of the quad starting in the left upper corner and continuing counter clock wise
    //					a--d
    //					|  |
    //					b--c
    bool check_quad_error_threshold_exceeded(ImageView<const vec3> unprojected_image, int triangle_orientation,
                                             float threshold, Point2D a, Point2D b, Point2D c, Point2D d);
};
}  // namespace Saiga


int main(){
    return 0;
}