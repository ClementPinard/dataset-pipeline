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

// Models pre-rectified simple pinhole cameras.
class SimplePinholeCamera : public CameraBaseImpl<SimplePinholeCamera> {
 public:
  SimplePinholeCamera(int width, int height, float f, float cx, float cy);
  
  SimplePinholeCamera(int width, int height, const float* parameters);
  
  inline SimplePinholeCamera* CreateUpdatedCamera(const float* parameters) const {
    return new SimplePinholeCamera(width_, height_, parameters);
  }
  
  static constexpr int ParameterCount() {
    return 3;
  }

  static constexpr bool UniqueFocalLength() {
    return true;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    return normalized_point;
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    return normalized_point;
  }
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 4 values each are returned for fx, fy, cx, cy.
  template <typename Derived>
  inline void NormalizedDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& normalized_point, float* deriv_x, float* deriv_y) const {
    deriv_x[0] = normalized_point.x();
    deriv_x[1] = 1.f;
    deriv_x[2] = 0.f;
    deriv_y[0] = normalized_point.y();
    deriv_y[1] = 0.f;
    deriv_y[2] = 1.f;
  }
  
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& /*normalized_point*/) const {
    return Eigen::Vector4f(1, 0, 0, 1);
  }
  
  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = cx();
    parameters[2] = cy();
  }
};

}  // namespace camera
