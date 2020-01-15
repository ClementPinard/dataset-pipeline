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

#include <math.h>

#include <Eigen/Core>

#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"

namespace camera {

// Models pinhole cameras with a polynomial-tangential distortion model.
class PolynomialTangentialCamera : public CameraBaseImpl<PolynomialTangentialCamera> {
 public:
  PolynomialTangentialCamera(int width, int height, float fx, float fy, float cx,
                             float cy, float k1, float k2, float p1, float p2);
  
  PolynomialTangentialCamera(int width, int height, const float* parameters);
  
  inline PolynomialTangentialCamera* CreateUpdatedCamera(
      const float* parameters) const {
    return new PolynomialTangentialCamera(width_, height_, parameters);
  }
  
  static constexpr int ParameterCount() {
    return 4 + 4;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float x2 = normalized_point.x() * normalized_point.x();
    const float xy = normalized_point.x() * normalized_point.y();
    const float y2 = normalized_point.y() * normalized_point.y();
    const float r2 = x2 + y2;

    if (r2 > radius_cutoff_squared_) {
      return Eigen::Vector2f(99 * normalized_point.x(), 99 * normalized_point.y());
    }
    const float radial =
        r2 * (distortion_parameters_.x() + r2 * distortion_parameters_.y());
    const float dx = 2.f * distortion_parameters_.z() * xy + distortion_parameters_.w() * (r2 + 2.f * x2);
    const float dy = 2.f * distortion_parameters_.w() * xy + distortion_parameters_.z() * (r2 + 2.f * y2);
    return Eigen::Vector2f(
        normalized_point.x() + radial * normalized_point.x() + dx,
        normalized_point.y() + radial * normalized_point.y() + dy);
  }
  
  
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 8 values each are returned for fx, fy, cx, cy,
  // k1, k2, p1, p2.
  template <typename Derived>
  inline void NormalizedDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& normalized_point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    
    const float nx2 = normalized_point.x() * normalized_point.x();
    const float ny2 = normalized_point.y() * normalized_point.y();
    const float two_nx_ny = 2.f * normalized_point.x() * normalized_point.y();
    const float r2 = nx2 + ny2;
    const float fx_nx = fx() * normalized_point.x();
    const float fy_ny = fy() * normalized_point.y();
    
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_x[4] = fx_nx * r2;
    deriv_x[5] = deriv_x[4] * r2;
    deriv_x[6] = fx() * two_nx_ny;
    deriv_x[7] = fx() * (r2 + 2.f * nx2);
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    deriv_y[4] = fy_ny * r2;
    deriv_y[5] = deriv_y[4] * r2;
    deriv_y[6] = fy() * (r2 + 2.f * ny2);
    deriv_y[7] = fy() * two_nx_ny;
  }
  
  // Derivation with Matlab:
  // syms nx ny k1 k2 p1 p2
  // x2 = nx * nx;
  // xy = nx * ny;
  // y2 = ny * ny;
  // r2 = x2 + y2;
  // radial = r2 * (k1 + r2 * k2);
  // dx = 2 * p1 * xy + p2 * (r2 + 2 * x2);
  // dy = 2 * p2 * xy + p1 * (r2 + 2 * y2);
  // px = nx + radial * nx + dx;
  // py = ny + radial * ny + dy;
  // simplify(diff(px, nx))
  // simplify(diff(px, ny))
  // simplify(diff(py, nx))
  // simplify(diff(py, ny))
  // Returns (ddx/dnx, ddx/dny, ddy/dnx, ddy/dny) as in above order,
  // with dx,dy being the distorted coords and d the partial derivative
  // operator.
  // Note: in case of small distortions, you may want to use (1, 0, 0, 1)
  // as an approximation.
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float k1 = distortion_parameters_.x();
    const float k2 = distortion_parameters_.y();
    const float p1 = distortion_parameters_.z();
    const float p2 = distortion_parameters_.w();
    const float nx2 = nx * nx;
    const float nx3 = nx2 * nx;
    const float nx4 = nx3 * nx;
    const float ny2 = ny * ny;
    const float ny3 = ny2 * ny;
    const float ny4 = ny3 * ny;
    
    const float ddx_dnx = 5*k2*nx4 + 6*k2*nx2*ny2 + 3*k1*nx2 + 6*p2*nx + k2*ny4 + k1*ny2 + 2*p1*ny + 1;
    const float ddx_dny = 4*k2*nx3*ny + 4*k2*nx*ny3 + 2*k1*nx*ny + 2*p1*nx + 2*p2*ny;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = k2*nx4 + 6*k2*nx2*ny2 + k1*nx2 + 2*p2*nx + 5*k2*ny4 + 3*k1*ny2 + 6*p1*ny + 1;
    return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }

  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_.x();
    parameters[5] = distortion_parameters_.y();
    parameters[6] = distortion_parameters_.z();
    parameters[7] = distortion_parameters_.w();
  }
  
  // Returns the distortion parameters p1, p2, and p3.
  inline const Eigen::Vector4f& distortion_parameters() const {
    return distortion_parameters_;
  }

 private:
  
  // The distortion parameters k1, k2, p1, p2.
  Eigen::Vector4f distortion_parameters_;
};

}  // namespace camera
