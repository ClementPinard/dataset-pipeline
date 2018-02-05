// Copyright 2017 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <mutex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

#include "base/util.h"
#include "opengl/renderer.h"
#include "opengl/mesh.h"
#include "opengl/opengl_util.h"
#include "opt/image.h"
#include "opt/intrinsics.h"

typedef std::pair<std::size_t, std::size_t> IndexPair;
MAKE_HASHABLE(IndexPair, t.first, t.second)

namespace opt {

// Stores geometry data which is used for occlusion testing.
class OcclusionGeometry {
 public:
  OcclusionGeometry();
  ~OcclusionGeometry();
   
  // Initializes a splat-based occlusion geometry.
  void SetSplatPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr splat_points);
  
  // Initializes a mesh-based occlusion geometry. Can be called multiple times
  // to add several meshes. Returns true if successful.
  bool AddMesh(const std::string& mesh_file_path);
  bool AddMesh(const std::string& mesh_file_path, const Sophus::SE3f& transformation);
  bool AddMesh(const pcl::PolygonMesh& mesh);
  
  cv::Mat_<float> RenderDepthMap(
      const opt::Intrinsics& intrinsics,
      const opt::Image& image,
      int image_scale,
      bool mask_occlusion_boundaries = true,
      float splat_radius_factor = 15.f / 6000.f,
      float min_depth = 0.05f,
      float max_depth = 100.f) const;
  
 private:
  // Stores an edge and its two adjacent faces.
  struct EdgeWithFaces {
    std::size_t vertex_index1;
    std::size_t vertex_index2;
    std::size_t face_index1;
    std::size_t face_index2;
  };
  
  struct EdgesWithFaceNormalsMesh {
    std::vector<EdgeWithFaces> edges_with_faces;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices;
    Eigen::Matrix<float, 3, Eigen::Dynamic> face_normals;
  };
  
  template<class Camera>
  cv::Mat_<float> _RenderDepthMapWithSplatsCPU(
      float splat_radius_factor,
      const Intrinsics& intrinsics,
      const Image& image,
      int image_scale,
      const Camera& image_scale_camera) const;
  
  void AddHalfEdge(
      uint32_t vertex1,
      uint32_t vertex2,
      uint32_t face_index,
      std::unordered_map<IndexPair, std::size_t>* half_edge_map,
      std::vector<EdgeWithFaces>* edges_with_faces);
  
  void ComputeEdgeNormalsList(
      const pcl::PolygonMesh& mesh,
      EdgesWithFaceNormalsMesh* output);
  
  template<class Camera>
  void MaskOutOcclusionBoundaries(
      float splat_radius_factor,
      const Camera& camera,
      const Image& image,
      const cv::Mat_<float>& input,
      cv::Mat_<float>* output) const;
  
  template<class Camera>
  void DrawSplatsAtEdgeIfVisible(
      int splat_radius,
      const Camera& camera,
      const Eigen::Matrix3f& image_R_global,
      const Eigen::Vector3f& image_T_global,
      const Eigen::Vector3f& endpoint1,
      const Eigen::Vector3f& endpoint2,
      const cv::Mat_<float>& input,
      cv::Mat_<float>* output) const;
  
  template<class Camera>
  void DrawEdgeSplatIfVisible(
      int splat_radius,
      const Camera& camera,
      const Eigen::Matrix3f& image_R_global,
      const Eigen::Vector3f& image_T_global,
      const Eigen::Vector3f& point,
      const cv::Mat_<float>& input,
      cv::Mat_<float>* output) const;
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr splat_points_;
  
  mutable std::mutex mutable_mutex_;
  bool opengl_context_initialized_;
  opengl::OpenGLContext opengl_context_;
  mutable std::shared_ptr<opengl::Renderer> depthmap_renderer_;
  mutable opengl::RendererProgramStoragePtr program_storage_;
  std::vector<std::shared_ptr<opengl::Mesh>> meshes_;
  std::vector<std::shared_ptr<EdgesWithFaceNormalsMesh>> edge_meshes_;
};

}  // namespace opt
