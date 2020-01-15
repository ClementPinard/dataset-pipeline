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

// Models pinhole cameras with a polynomial distortion model.
class RadialCamera : public CameraBaseImpl<RadialCamera> {
 public:
  RadialCamera(int width, int height, float fx, float fy, float cx,
                   float cy, float k1, float k2);
  
  RadialCamera(int width, int height, const float* parameters);
  
  inline RadialCamera* CreateUpdatedCamera(const float* parameters) const {
    return new RadialCamera(width_, height_, parameters);
  }
  
  static constexpr int ParameterCount() {
    return 4 + 2;
  }

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r2 = normalized_point.x() * normalized_point.x() +
                     normalized_point.y() * normalized_point.y();

    const float factw =
        1.0f +
        r2 * (distortion_parameters_.x() +
              r2 * (distortion_parameters_.y()));
    return Eigen::Vector2f(factw * normalized_point.x(), factw * normalized_point.y());
  }

  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 7 values each are returned for fx, fy, cx, cy,
  // k1, k2.
  template <typename Derived>
  inline void NormalizedDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& normalized_point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    
    const float radius_square =
        normalized_point.x() * normalized_point.x() +
        normalized_point.y() * normalized_point.y();
    
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_x[4] = fx() * normalized_point.x() * radius_square;
    deriv_x[5] = deriv_x[4] * radius_square;
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    deriv_y[4] = fy() * normalized_point.y() * radius_square;
    deriv_y[5] = deriv_y[4] * radius_square;
  }
  
  // Derivation with Matlab:
  // syms nx ny px py pz
  // ru2 = nx*nx + ny*ny
  // factw = 1 + ru2 * (px + ru2 * py)
  // simplify(diff(nx * factw, nx))
  // simplify(diff(nx * factw, ny))
  // simplify(diff(ny * factw, nx))
  // simplify(diff(ny * factw, ny))
  // Returns (ddx/dnx, ddx/dny, ddy/dnx, ddy/dny) as in above order,
  // with dx,dy being the distorted coords and d the partial derivative
  // operator.
  // Note: in case of small distortions, you may want to use (1, 0, 0, 1)
  // as an approximation.
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nxs = nx * nx;
    const float nys = ny * ny;
    const float nxs_plus_nxy = nxs + nys;

    const float part1 = distortion_parameters_.y();
    const float part2 = distortion_parameters_.x() + part1 * nxs_plus_nxy;
    const float part3 =
        (2 * ny * part1) * nxs_plus_nxy +
        2 * ny * part2;

    const float ddx_dnx =
        nx * ((2 * nx * part1) * nxs_plus_nxy + 2 * nx * part2) +
        part2 * nxs_plus_nxy + 1;
    const float ddx_dny = nx * part3;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = ny * part3 + part2 * nxs_plus_nxy + 1;

    return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }
  
  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_.x();
    parameters[5] = distortion_parameters_.y();
  }

  // Returns the distortion parameters p1, p2, and p3.
  inline const Eigen::Vector2f& distortion_parameters() const {
    return distortion_parameters_;
  }
  
  // The distortion parameters p1, p2, and p3.
  Eigen::Vector2f distortion_parameters_;

};

}  // namespace camera
