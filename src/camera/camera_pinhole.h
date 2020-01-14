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

#include <Eigen/Core>

#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"

namespace camera {

// Models pre-rectified pinhole cameras.
class PinholeCamera : public CameraBaseImpl<PinholeCamera> {
 public:
  PinholeCamera(int width, int height, float fx, float fy, float cx, float cy);
  
  PinholeCamera(int width, int height, const float* parameters);
  
  inline PinholeCamera* CreateUpdatedCamera(const float* parameters) const {
    return new PinholeCamera(width_, height_, parameters);
  }
  
  static constexpr int ParameterCount() {
    return 4;
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    return normalized_point;
  }
  
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const int x, const int y) const {
    return Undistort(Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
  }
  
  template <typename Derived>
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const Eigen::MatrixBase<Derived>& pixel_position) const {
    return Undistort(Eigen::Vector2f(fx_inv() * pixel_position.x() + cx_inv(), fy_inv() * pixel_position.y() + cy_inv()));
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    return normalized_point;
  }

  void InitializeUnprojectionLookup() {}

  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 4 values each are returned for fx, fy, cx, cy.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f distorted_point = Distort(Eigen::Vector2f(point.x() / point.z(), point.y() / point.z()));
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
  }
  
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& /*normalized_point*/) const {
    return Eigen::Vector4f(1, 0, 0, 1);
  }
  
  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
  }
};

}  // namespace camera
