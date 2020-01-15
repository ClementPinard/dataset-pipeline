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

// The camera model used for the ETH3D benchmark.
class BenchmarkCamera : public CameraBaseImpl<BenchmarkCamera> {
 public:
  BenchmarkCamera(int width, int height, float fx, float fy,
                  float cx, float cy, float k1, float k2,
                  float p1, float p2, float k3, float k4,
                  float sx1, float sy1);
  
  BenchmarkCamera(int width, int height, const float* parameters);
  
  inline BenchmarkCamera* CreateUpdatedCamera(const float* parameters) const {
    return new BenchmarkCamera(width_, height_, parameters);
  }
  
  static constexpr int ParameterCount() {
    return 4 + 8;
  }
  
  template <typename Derived>
  inline Eigen::Vector2f Distort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r = sqrtf(normalized_point.x() * normalized_point.x() +
                          normalized_point.y() * normalized_point.y());
    float x, y;
    if (r > radius_cutoff_squared_) {
      return Eigen::Vector2f((normalized_point.x() < 0) ? -100 : 100,
                         (normalized_point.y() < 0) ? -100 : 100);
    }
    if (r > kEpsilon) {
      const float theta_by_r = atan2(r, 1.f) / r;
      x = theta_by_r * normalized_point.x();
      y = theta_by_r * normalized_point.y();
    } else {
      x = normalized_point.x();
      y = normalized_point.y();
    }
    
    return DistortWithoutFisheye(Eigen::Vector2f(x, y));
  }
  
  template <typename Derived>
  inline Eigen::Vector2f DistortWithoutFisheye(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];
    
    const float x2 = normalized_point.x() * normalized_point.x();
    const float xy = normalized_point.x() * normalized_point.y();
    const float y2 = normalized_point.y() * normalized_point.y();
    const float r2 = x2 + y2;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r6 * r2;
    
    const float radial =
        k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    const float dx = 2.f * p1 * xy + p2 * (r2 + 2.f * x2) + sx1 * r2;
    const float dy = 2.f * p2 * xy + p1 * (r2 + 2.f * y2) + sy1 * r2;
    return Eigen::Vector2f(
        normalized_point.x() + radial * normalized_point.x() + dx,
        normalized_point.y() + radial * normalized_point.y() + dy);
  }
  
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 12 values each are returned for fx, fy, cx, cy,
  // k1, k2, p1, p2, k3, k4, sx1, sy1.
  template <typename Derived>
  inline void NormalizedDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& normalized_point, float* deriv_x, float* deriv_y) const {
    
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    deriv_x[0] = distorted_point.x();
    deriv_x[1] = 0.f;
    deriv_x[2] = 1.f;
    deriv_x[3] = 0.f;
    deriv_y[0] = 0.f;
    deriv_y[1] = distorted_point.y();
    deriv_y[2] = 0.f;
    deriv_y[3] = 1.f;
    
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float nx2 = normalized_point.x() * normalized_point.x();
    const float ny2 = normalized_point.y() * normalized_point.y();
    const float two_nx_ny = 2.f * nx * ny;
    const float fx_nx = fx() * normalized_point.x();
    const float fy_ny = fy() * normalized_point.y();
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float atan_r_2 = atan_r * atan_r;
      const float atan_r_3_by_r = (atan_r_2 * atan_r) / r;
      const float two_nx_ny_atan_r_2_by_r2 = (two_nx_ny * atan_r_2) / r2;
      const float atan_r_2_by_r2 = atan_r_2 / r2;
      
      deriv_x[4] = fx_nx * atan_r_3_by_r;
      deriv_x[5] = deriv_x[4] * atan_r_2;
      deriv_x[6] = fx() * two_nx_ny_atan_r_2_by_r2;
      deriv_x[7] = fx() * atan_r_2_by_r2 * (3 * nx2 + ny2);
      deriv_x[8] = deriv_x[5] * atan_r_2;
      deriv_x[9] = deriv_x[8] * atan_r_2;
      deriv_x[10] = fx() * atan_r_2;
      deriv_x[11] = 0;
      
      deriv_y[4] = fy_ny * atan_r_3_by_r;
      deriv_y[5] = deriv_y[4] * atan_r_2;
      deriv_y[6] = fy() * atan_r_2_by_r2 * (nx2 + 3 * ny2);
      deriv_y[7] = fy() * two_nx_ny_atan_r_2_by_r2;
      deriv_y[8] = deriv_y[5] * atan_r_2;
      deriv_y[9] = deriv_y[8] * atan_r_2;
      deriv_y[10] = 0;
      deriv_y[11] = fy() * atan_r_2;
    } else {
      // The non-fisheye variant is used in this case.
      deriv_x[4] = fx_nx * r2;
      deriv_x[5] = deriv_x[4] * r2;
      deriv_x[6] = fx() * two_nx_ny;
      deriv_x[7] = fx() * (r2 + 2.f * nx2);
      deriv_x[8] = deriv_x[5] * r2;
      deriv_x[9] = deriv_x[8] * r2;
      deriv_x[10] = fx() * r2;
      deriv_x[11] = 0;
      
      deriv_y[4] = fy_ny * r2;
      deriv_y[5] = deriv_y[4] * r2;
      deriv_y[6] = fy() * (r2 + 2.f * ny2);
      deriv_y[7] = fy() * two_nx_ny;
      deriv_y[8] = deriv_y[5] * r2;
      deriv_y[9] = deriv_y[8] * r2;
      deriv_y[10] = 0;
      deriv_y[11] = fy() * r2;
    }
  }
  
  template <typename Derived>
  inline Eigen::Vector4f DistortionDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];
    
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;
    const float r = sqrtf(r2);
    if (r > radius_cutoff_squared_) {
      return Eigen::Vector4f(0, 0, 0, 0);
    }
    if (r > kEpsilon) {
      const float atan_r = atanf(r);
      const float r3 = r2 * r;
      
      const float term1 = r2 * (r2 + 1);
      const float term2 = atan_r / r3;
      
      // Derivatives of fisheye x / y coordinates by nx / ny:
      const float dnxf_dnx = ny2 * term2 + nx2 / term1;
      const float dnxf_dny = nx_ny / term1 - nx_ny * term2;
      const float dnyf_dnx = dnxf_dny;
      const float dnyf_dny = nx2 * term2 + ny2 / term1;
      
      // Compute fisheye x / y.
      const float theta_by_r = atan2(r, 1.f) / r;
      const float x = theta_by_r * nx;
      const float y = theta_by_r * ny;
      
      // Derivatives of distorted coordinates by fisheye x / y:
      // (same computation as in non-fisheye polynomial-tangential)

      const float x_y = x * y;
      const float x2 = x * x;
      const float y2 = y * y;
      
      const float rf2 = x2 + y2;
      const float rf4 = rf2 * rf2;
      const float rf6 = rf4 * rf2;
      const float rf8 = rf6 * rf2;
      
      // NOTE: Could factor out more terms here which might improve performance.
      const float term1f = 2*p1*x + 2*p2*y + 2*k1*x_y + 6*k3*x_y*rf4 + 8*k4*x_y*rf6 + 4*k2*x_y*rf2;
      const float ddx_dnxf = 2*k1*x2 + 4*k2*x2*rf2 + 6*k3*x2*rf4 + 8*k4*x2*rf6 + k2*rf4 + k3*rf6 + k4*rf8 + 6*p2*x + 2*p1*y + 2*sx1*x + k1*rf2 + 1;
      const float ddx_dnyf = 2*sx1*y + term1f;
      const float ddy_dnxf = 2*sy1*x + term1f;
      const float ddy_dnyf = 2*k1*y2 + 4*k2*y2*rf2 + 6*k3*y2*rf4 + 8*k4*y2*rf6 + k2*rf4 + k3*rf6 + k4*rf8 + 2*p2*x + 6*p1*y + 2*sy1*y + k1*rf2 + 1;
      return Eigen::Vector4f(ddx_dnxf * dnxf_dnx + ddx_dnyf * dnyf_dnx,
                         ddy_dnxf * dnxf_dnx + ddy_dnyf * dnyf_dnx,
                         ddx_dnxf * dnxf_dny + ddx_dnyf * dnyf_dny,
                         ddy_dnxf * dnxf_dny + ddy_dnyf * dnyf_dny);
    } else {
      // Non-fisheye variant is used in this case.
      const float r4 = r2 * r2;
      const float r6 = r4 * r2;
      const float r8 = r6 * r2;
      
      // NOTE: Could factor out more terms here which might improve performance.
      const float term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*r2;
      const float ddx_dnx = 2*k1*nx2 + 4*k2*nx2*r2 + 6*k3*nx2*r4 + 8*k4*nx2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*r2 + 1;
      const float ddx_dny = 2*sx1*ny + term1;
      const float ddy_dnx = 2*sy1*nx + term1;
      const float ddy_dny = 2*k1*ny2 + 4*k2*ny2*r2 + 6*k3*ny2*r4 + 8*k4*ny2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*r2 + 1;
      return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
    }
  }
  
  template <typename Derived>
  inline Eigen::Vector4f DistortionWithoutFisheyeDerivative(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float nx = normalized_point.x();
    const float ny = normalized_point.y();
    const float k1 = distortion_parameters_[0];
    const float k2 = distortion_parameters_[1];
    const float p1 = distortion_parameters_[2];
    const float p2 = distortion_parameters_[3];
    const float k3 = distortion_parameters_[4];
    const float k4 = distortion_parameters_[5];
    const float sx1 = distortion_parameters_[6];
    const float sy1 = distortion_parameters_[7];
    
    const float nx_ny = nx * ny;
    const float nx2 = nx * nx;
    const float ny2 = ny * ny;
    const float r2 = nx2 + ny2;

    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float r8 = r6 * r2;
    
    // NOTE: Could factor out more terms here which might improve performance.
    const float term1 = 2*p1*nx + 2*p2*ny + 2*k1*nx_ny + 6*k3*nx_ny*r4 + 8*k4*nx_ny*r6 + 4*k2*nx_ny*r2;
    const float ddx_dnx = 2*k1*nx2 + 4*k2*nx2*r2 + 6*k3*nx2*r4 + 8*k4*nx2*r6 + k2*r4 + k3*r6 + k4*r8 + 6*p2*nx + 2*p1*ny + 2*sx1*nx + k1*r2 + 1;
    const float ddx_dny = 2*sx1*ny + term1;
    const float ddy_dnx = 2*sy1*nx + term1;
    const float ddy_dny = 2*k1*ny2 + 4*k2*ny2*r2 + 6*k3*ny2*r4 + 8*k4*ny2*r6 + k2*r4 + k3*r6 + k4*r8 + 2*p2*nx + 6*p1*ny + 2*sy1*ny + k1*r2 + 1;
    return Eigen::Vector4f(ddx_dnx, ddx_dny, ddy_dnx, ddy_dny);
  }
  
  inline void GetParameters(float* parameters) const {
    parameters[0] = fx();
    parameters[1] = fy();
    parameters[2] = cx();
    parameters[3] = cy();
    parameters[4] = distortion_parameters_[0];
    parameters[5] = distortion_parameters_[1];
    parameters[6] = distortion_parameters_[2];
    parameters[7] = distortion_parameters_[3];
    parameters[8] = distortion_parameters_[4];
    parameters[9] = distortion_parameters_[5];
    parameters[10] = distortion_parameters_[6];
    parameters[11] = distortion_parameters_[7];
  }

  // Returns the distortion parameters.
  inline const float* distortion_parameters() const {
    return distortion_parameters_;
  }

 private:
  
  // The distortion parameters k1, k2, p1, p2, k3, k4, sx1, sy1.
  float distortion_parameters_[8];
  
  static constexpr float kEpsilon = 1e-6f;
};

}  // namespace camera
