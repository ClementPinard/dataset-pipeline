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

namespace camera {

// Models pinhole cameras with a polynomial distortion model.
class SimpleRadialCamera : public CameraBase {
 public:
  SimpleRadialCamera(int width, int height, float f,
                     float cx, float cy, float k);
  
  SimpleRadialCamera(int width, int height, const float* parameters);
  
  inline SimpleRadialCamera* CreateUpdatedCamera(const float* parameters) const {
    return new SimpleRadialCamera(width_, height_, parameters);
  }
  
  ~SimpleRadialCamera();
  
  static constexpr int ParameterCount() {
    return 3 + 1;
  }

  CameraBase* ScaledBy(float factor) const override;
  CameraBase* ShiftedBy(float cx_offset, float cy_offset) const override;
  void InitializeUnprojectionLookup() override;
  void InitCutoff();

  template <typename Derived>
  inline Eigen::Vector2f ProjectToNormalizedTextureCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(nfx() * distorted_point.x() + ncx(),
                       nfy() * distorted_point.y() + ncy());
  }
  
  template <typename Derived>
  inline Eigen::Vector2f ProjectToImageCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    return Eigen::Vector2f(fx() * distorted_point.x() + cx(),
                       fy() * distorted_point.y() + cy());
  }

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

  inline Eigen::Vector2f UnprojectFromImageCoordinates(const int x, const int y) const {
    return undistortion_lookup_[y * width_ + x];
  }

  template <typename Derived>
  inline Eigen::Vector2f UnprojectFromImageCoordinates(const Eigen::MatrixBase<Derived>& pixel_position) const {
    // Manual implementation of bilinearly filtering the lookup.
    Eigen::Vector2f clamped_pixel = Eigen::Vector2f(
        std::max(0.f, std::min(width() - 1.001f, pixel_position.x())),
        std::max(0.f, std::min(height() - 1.001f, pixel_position.y())));
    Eigen::Vector2i int_pos = Eigen::Vector2i(clamped_pixel.x(), clamped_pixel.y());
    Eigen::Vector2f factor =
        Eigen::Vector2f(clamped_pixel.x() - int_pos.x(), clamped_pixel.y() - int_pos.y());
    Eigen::Vector2f top_left = undistortion_lookup_[int_pos.y() * width_ + int_pos.x()];
    Eigen::Vector2f top_right =
        undistortion_lookup_[int_pos.y() * width_ + (int_pos.x() + 1)];
    Eigen::Vector2f bottom_left =
        undistortion_lookup_[(int_pos.y() + 1) * width_ + int_pos.x()];
    Eigen::Vector2f bottom_right =
        undistortion_lookup_[(int_pos.y() + 1) * width_ + (int_pos.x() + 1)];
    return Eigen::Vector2f(
        (1 - factor.y()) *
                ((1 - factor.x()) * top_left.x() + factor.x() * top_right.x()) +
            factor.y() *
                ((1 - factor.x()) * bottom_left.x() + factor.x() * bottom_right.x()),
        (1 - factor.y()) *
                ((1 - factor.x()) * top_left.y() + factor.x() * top_right.y()) +
            factor.y() *
                ((1 - factor.x()) * bottom_left.y() + factor.x() * bottom_right.y()));
  }

  // This iterative Undistort() function should not be used in
  // time critical code. An undistortion texture may be preferable,
  // as used by the UnprojectFromImageCoordinates() methods. Undistort() is only
  // used for calculating this undistortion texture once.
  // Notably, this function employs the Newton method in contrast to the
  // corresponding functions in calibration-provider and OpenCV, as those
  // diverge in large parts of an image with commonly used parameter settings.
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point, float uu, float vv, bool* converged) const {
    const std::size_t kNumUndistortionIterations = 100;
    
    // Gauss-Newton.
    const float kUndistortionEpsilon = 1e-10f;
    if (converged) {
      *converged = false;
    }
    for (std::size_t i = 0; i < kNumUndistortionIterations; ++i) {
      Eigen::Vector2f distorted = Distort(Eigen::Vector2f(uu, vv));
      // (Non-squared) residuals.
      float dx = distorted.x() - distorted_point.x();
      float dy = distorted.y() - distorted_point.y();
      
      // Accumulate H and b.
      Eigen::Vector4f ddxy_dxy = DistortionDerivative(Eigen::Vector2f(uu, vv));
      float H_0_0 = ddxy_dxy.x() * ddxy_dxy.x() + ddxy_dxy.z() * ddxy_dxy.z();
      float H_1_0_and_0_1 = ddxy_dxy.x() * ddxy_dxy.y() + ddxy_dxy.z() * ddxy_dxy.w();
      float H_1_1 = ddxy_dxy.y() * ddxy_dxy.y() + ddxy_dxy.w() * ddxy_dxy.w();
      float b_0 = dx * ddxy_dxy.x() + dy * ddxy_dxy.z();
      float b_1 = dx * ddxy_dxy.y() + dy * ddxy_dxy.w();
      
      // Solve the system and update the parameters.
      float x_1 = (b_1 - H_1_0_and_0_1 / H_0_0 * b_0) /
                  (H_1_1 - H_1_0_and_0_1 * H_1_0_and_0_1 / H_0_0);
      float x_0 = (b_0 - H_1_0_and_0_1 * x_1) / H_0_0;
      uu -= x_0;
      vv -= x_1;
      
      if (dx * dx + dy * dy < kUndistortionEpsilon) {
        if (converged) {
          *converged = true;
        }
        break;
      }
    }
    
    return Eigen::Vector2f(uu, vv);
  }

  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const float r_d = sqrtf(normalized_point.x() * normalized_point.x() +
                            normalized_point.y() * normalized_point.y());
    float r = r_d;
    float r2 = r * r;
    constexpr int kMaxIterations = 50;
    float residual_non_squared =
        r_d - (r * (1.0f + r2 * k1_));
    for (int j = 0; j < kMaxIterations; ++j) {
      float jac = 1.0f + r2 * (3.0f * k1_);
      float delta = residual_non_squared / jac;
      float r_next = r + delta;
      float r2_next = r_next * r_next;

      float residual_non_squared_next =
          r_d -
          (r_next *
           (1.0f + r2_next * k1_));
      if (residual_non_squared_next * residual_non_squared_next <
          residual_non_squared * residual_non_squared) {
        r = r_next;
        r2 = r2_next;
        residual_non_squared = residual_non_squared_next;
      } else {
        break;
      }
    }
    float undistortion_factor = r / r_d;
    return Eigen::Vector2f(undistortion_factor * normalized_point.x(),
                       undistortion_factor * normalized_point.y());
  }

  // Tries to return the innermost undistorted point which maps to the given
  // distorted point (as opposed to returning any undistorted point that maps
  // correctly).
  template <typename Derived1, typename Derived2>
  Eigen::Vector2f UndistortFromInside(
      const Eigen::MatrixBase<Derived1>& distorted_point,
      bool* converged,
      Eigen::MatrixBase<Derived2>* second_best_result,
      bool* second_best_available) const {
    // Try different initialization points and take the result with the smallest
    // radius.
    constexpr int kNumGridSteps = 10;
    constexpr float kGridHalfExtent = 1.5f;
    constexpr float kImproveThreshold = 0.99f;
    
    *converged = false;
    *second_best_available = false;
    
    float best_radius = 999999;  // std::numeric_limits<float>::infinity();
    Eigen::Vector2f best_result;
    float second_best_radius = 999999;
    
    for (int y = 0; y < kNumGridSteps; ++ y) {
      float y_init = distorted_point.y() + kGridHalfExtent * (y - 0.5f * kNumGridSteps) / (0.5f * kNumGridSteps);
      for (int x = 0; x < kNumGridSteps; ++ x) {
        float x_init = distorted_point.x() + kGridHalfExtent * (x - 0.5f * kNumGridSteps) / (0.5f * kNumGridSteps);
        
        bool test_converged;
        Eigen::Vector2f result = Undistort(distorted_point, x_init, y_init, &test_converged);
        if (test_converged) {
          float radius = sqrtf(result.x() * result.x() + result.y() * result.y());
          if (radius < kImproveThreshold * best_radius) {
            second_best_radius = best_radius;
            *second_best_result = best_result;
            *second_best_available = *converged;
            
            best_radius = radius;
            best_result = result;
            *converged = true;
          } else if (radius > 1 / kImproveThreshold * best_radius &&
                     radius < kImproveThreshold * second_best_radius) {
            second_best_radius = radius;
            *second_best_result = result;
            *second_best_available = true;
          }
        }
      }
    }
    
    return best_result;
  }
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived>
  inline void ProjectionToNormalizedTextureCoordinatesDerivative(
      const Eigen::MatrixBase<Derived>& point, Eigen::Vector3f* deriv_x, Eigen::Vector3f* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = DistortionDerivative(normalized_point);
    const Eigen::Vector4f projection_deriv =
        Eigen::Vector4f(nfx() * distortion_deriv.x(),
                    nfx() * distortion_deriv.y(),
                    nfy() * distortion_deriv.z(),
                    nfy() * distortion_deriv.w());
    *deriv_x = Eigen::Vector3f(
        projection_deriv.x() / point.z(), projection_deriv.y() / point.z(),
        -1.0f * (projection_deriv.x() * point.x() + projection_deriv.y() * point.y()) /
            (point.z() * point.z()));
    *deriv_y = Eigen::Vector3f(
        projection_deriv.z() / point.z(), projection_deriv.w() / point.z(),
        -1.0f * (projection_deriv.z() * point.x() + projection_deriv.w() * point.y()) /
            (point.z() * point.z()));
  }
  
  // Returns the derivatives of the normalized projected coordinates with
  // respect to the 3D change of the input point.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivative(
      const Eigen::MatrixBase<Derived>& point, Eigen::Vector3f* deriv_x, Eigen::Vector3f* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = DistortionDerivative(normalized_point);
    const Eigen::Vector4f projection_deriv =
        Eigen::Vector4f(fx() * distortion_deriv.x(),
                    fx() * distortion_deriv.y(),
                    fy() * distortion_deriv.z(),
                    fy() * distortion_deriv.w());
    *deriv_x = Eigen::Vector3f(
        projection_deriv.x() / point.z(), projection_deriv.y() / point.z(),
        -1.0f * (projection_deriv.x() * point.x() + projection_deriv.y() * point.y()) /
            (point.z() * point.z()));
    *deriv_y = Eigen::Vector3f(
        projection_deriv.z() / point.z(), projection_deriv.w() / point.z(),
        -1.0f * (projection_deriv.z() * point.x() + projection_deriv.w() * point.y()) /
            (point.z() * point.z()));
  }
  
  // Returns the derivatives of the image coordinates with respect to the
  // intrinsics. For x and y, 4 values each are returned for f, cx, cy, k.
  template <typename Derived>
  inline void ProjectionToImageCoordinatesDerivativeByIntrinsics(
      const Eigen::MatrixBase<Derived>& point, float* deriv_x, float* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector2f distorted_point = Distort(normalized_point);
    
    const float radius_square =
        normalized_point.x() * normalized_point.x() +
        normalized_point.y() * normalized_point.y();
    
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
  float k1_, radius_cutoff_squared_;

  Eigen::Vector2f* undistortion_lookup_;
};

}  // namespace camera
