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
#include <glog/logging.h>
#include "camera/camera_base.h"
#include "camera/camera_base_impl.h"

namespace camera {

// Models pinhole cameras with a polynomial distortion model.
class SimpleRadialCamera : public CameraBaseImpl<SimpleRadialCamera> {
 public:
  SimpleRadialCamera(int width, int height, float f,
                     float cx, float cy, float k);
  
  SimpleRadialCamera(int width, int height, const float* parameters);
  
  inline SimpleRadialCamera* CreateUpdatedCamera(const float* parameters) const {
    return new SimpleRadialCamera(width_, height_, parameters);
  }
  
  static constexpr int ParameterCount() {
    return 3 + 1;
  }

  static constexpr bool UniqueFocalLength() {
    return true;
  }

  void InitCutoff();

  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r2 = normalized_point.x() * normalized_point.x() +
                     normalized_point.y() * normalized_point.y();
    if(r2 > radius_cutoff_squared_){
      return Eigen::Vector2f(99 * normalized_point.x(), 99 * normalized_point.y());
    }
    const float factw =
        1.0f + r2 * k1_;
    return Eigen::Vector2f(factw * normalized_point.x(), factw * normalized_point.y());
  }
  
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 4 values each are returned for f, cx, cy, k.
  template <typename Derived>
  inline void NormalizedDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& normalized_point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    
    const float radius_square = normalized_point.squaredNorm();
    
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 1.f;
    deriv_x[2] = 0.f;
    deriv_x[3] = fx() * normalized_point.x() * radius_square;
    deriv_y[0] = distorted_point.y();
    deriv_y[1] = 0.f;
    deriv_y[2] = 1.f;
    deriv_y[3] = fy() * normalized_point.y() * radius_square;
  }
  
  // Derivation with Matlab:
  // syms nx ny px py pz
  // ru2 = nx*nx + ny*ny
  // factw = 1 + ru2 * k
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
    const float ru2 = nxs + nys;
    const float k = k1_;
    const float r2_cutoff = radius_cutoff_squared_;

    if(ru2 > r2_cutoff)
      return Eigen::Vector4f(0, 0, 0, 0);
    const float ddx_dnx = k * (ru2 + 2 * nxs) + 1;
    const float ddx_dny = 2 * nx * ny * k;
    const float ddy_dnx = ddx_dny;
    const float ddy_dny = k * (ru2 + 2 * nys) + 1;

    return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }
  
  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = cx();
    parameters[2] = cy();
    parameters[3] = k1_;
  }

  // Returns the distortion parameters k and r_cutoff.
  inline Eigen::Vector2f distortion_parameters() const {
    return Eigen::Vector2f(k1_, radius_cutoff_squared_);
  }

 private:
  
  // The distortion parameter k. and r_cutoff
  float k1_;
};

}  // namespace camera
