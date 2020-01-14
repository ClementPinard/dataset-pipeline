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
template <class Child> class CameraBaseImpl : public CameraBase {
 public:
  
  CameraBaseImpl(int width, int height, float fx, float fy, float cx, float cy, Type type)
  : CameraBase(width, height, fx, fy, cx, cy, type),
    undistortion_lookup_(0) {}
  
  ~CameraBaseImpl() {
  delete[] undistortion_lookup_;
  }

  static constexpr bool UniqueFocalLength() {
    return false;
  }

  // Returns a camera object which is scaled by the given factor.
  CameraBase* ScaledBy(float factor) const override {
  CHECK_NE(factor, 0.0f);
  int scaled_width = static_cast<int>(factor * width_);
  int scaled_height = static_cast<int>(factor * height_);
  const Child* child = static_cast<const Child*>(this);
  float parameters[child->ParameterCount()];
  child->GetParameters(parameters);
  if(!child->UniqueFocalLength()){
    parameters[0] *= factor;
    parameters[1] *= factor;
    parameters[2] = factor * (cx() + 0.5f) - 0.5f;
    parameters[3] = factor * (cy() + 0.5f) - 0.5f;
  }else{
    parameters[0] *= factor;
    parameters[1] = factor * (cx() + 0.5f) - 0.5f;
    parameters[2] = factor * (cy() + 0.5f) - 0.5f;
  }
  return new Child(
      scaled_width, scaled_height, parameters);
  }

  // Returns a camera object which is shifted by the given offset (in image
  // coordinates).
  CameraBase* ShiftedBy(float cx_offset, float cy_offset) const override {
    const Child* child = static_cast<const Child*>(this);
    float parameters[child->ParameterCount()];
    child->GetParameters(parameters);
    if(!child->UniqueFocalLength()){
      parameters[2] += cx_offset;
      parameters[3] += cy_offset;
    }else{
      parameters[1] += cx_offset;
      parameters[2] += cy_offset;
    }
  return new Child(
      width_, height_, parameters);
  }

  template <typename Derived>
  inline Eigen::Vector2f ProjectToNormalizedTextureCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = static_cast<const Child*>(this)->template Distort<Derived>(normalized_point);
    return Eigen::Vector2f(nfx() * distorted_point.x() + ncx(),
                       nfy() * distorted_point.y() + ncy());
  }
  
  template <typename Derived>
  inline Eigen::Vector2f ProjectToImageCoordinates(const Eigen::MatrixBase<Derived>& normalized_point) const {
    const Eigen::Vector2f distorted_point = static_cast<const Child*>(this)->template Distort<Derived>(normalized_point);
    return Eigen::Vector2f(fx() * distorted_point.x() + cx(),
                       fy() * distorted_point.y() + cy());
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
  template <typename Derived>
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point, float uu, float vv, bool* converged) const {
    const std::size_t kNumUndistortionIterations = 100;
    
    // Gauss-Newton.
    const float kUndistortionEpsilon = 1e-10f;
    if (converged) {
      *converged = false;
    }
    for (std::size_t i = 0; i < kNumUndistortionIterations; ++i) {
      Eigen::Vector2f distorted = static_cast<const Child*>(this)->template Distort<Derived>(Eigen::Vector2f(uu, vv));
      // (Non-squared) residuals.
      float dx = distorted.x() - distorted_point.x();
      float dy = distorted.y() - distorted_point.y();
      
      // Accumulate H and b.
      Eigen::Vector4f ddxy_dxy = static_cast<const Child*>(this)->template DistortionDerivative<Derived>(Eigen::Vector2f(uu, vv));
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
  inline Eigen::Vector2f Undistort(const Eigen::MatrixBase<Derived>& distorted_point) const {
    return Undistort(distorted_point, distorted_point.x(), distorted_point.y(), nullptr);
  }

  void InitializeUnprojectionLookup() {
    undistortion_lookup_ = new Eigen::Vector2f[height() * width()];
    Eigen::Vector2f* ptr = undistortion_lookup_;
    for (int y = 0; y < height(); ++y) {
      for (int x = 0; x < width(); ++x) {
        *ptr = Undistort(
            Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
        ++ptr;
      }
    }
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
        Eigen::Vector2f result = static_cast<const Child*>(this)->template Undistort<Derived1>(distorted_point, x_init, y_init, &test_converged);
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
  template <typename Derived1, typename Derived2, typename Derived3>
  inline void ProjectionToImageCoordinatesDerivative(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>* deriv_x, Eigen::MatrixBase<Derived3>* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = static_cast<const Child*>(this)->DistortionDerivative(normalized_point);
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
template <typename Derived1, typename Derived2, typename Derived3>
  inline void ProjectionToNormalizedTextureCoordinatesDerivative(
      const Eigen::MatrixBase<Derived1>& point, Eigen::MatrixBase<Derived2>* deriv_x, Eigen::MatrixBase<Derived3>* deriv_y) const {
    const Eigen::Vector2f normalized_point =
        Eigen::Vector2f(point.x() / point.z(), point.y() / point.z());
    const Eigen::Vector4f distortion_deriv = static_cast<const Child*>(this)->DistortionDerivative(normalized_point);
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

  template <typename Derived>
  inline void update_radius(const Eigen::Vector2f& nxy,
                            const Eigen::MatrixBase<Derived>& second_best_result,
                            bool converged,
                            bool second_best_available,
                            float increaseFactor,
                            float* result, float* max_result) {
    if (converged) {
        float radius = nxy.norm();
        if (increaseFactor * radius > *result) {
          *result = increaseFactor * radius;
        }
        if (second_best_available) {
          float second_best_radius = sqrtf(second_best_result.x() * second_best_result.x() + second_best_result.y() * second_best_result.y());
          if (second_best_radius < *max_result) {
            *max_result = second_best_radius;
          }
        }
      }
  }

  inline void InitCutoff() {
    constexpr float kIncreaseFactor = 1.01f;
    
    // Unproject some sample points at the image borders to find out where to
    // stop projecting points that are too far out. Those might otherwise get
    // projected into the image again at some point with certain distortion
    // parameter settings.
    
    // Disable cutoff while running this function such that the unprojection works.
    radius_cutoff_squared_ = std::numeric_limits<float>::infinity();
    float result = 0;
    float maximum_result = std::numeric_limits<float>::infinity();

    bool converged;
    Eigen::Vector2f second_best_result = Eigen::Vector2f::Zero();
    bool second_best_available;
    
    for (int x = 0; x < width_; ++ x) {
      Eigen::Vector2f nxy = UndistortFromInside(
          Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * 0 + cy_inv()),
          &converged, &second_best_result, &second_best_available);
      update_radius(nxy, second_best_result,
                    converged, second_best_available, kIncreaseFactor,
                    &result, &maximum_result);
      
      nxy = UndistortFromInside(
          Eigen::Vector2f(fx_inv() * x + cx_inv(), fy_inv() * (height_ - 1) + cy_inv()),
          &converged, &second_best_result, &second_best_available);
      update_radius(nxy, second_best_result,
                    converged, second_best_available, kIncreaseFactor,
                    &result, &maximum_result);
    }
    
    for (int y = 1; y < height_ - 1; ++ y) {
      Eigen::Vector2f nxy = UndistortFromInside(
          Eigen::Vector2f(fx_inv() * 0 + cx_inv(), fy_inv() * y + cy_inv()),
          &converged, &second_best_result, &second_best_available);
      update_radius(nxy, second_best_result,
                    converged, second_best_available, kIncreaseFactor,
                    &result, &maximum_result);
      
      nxy = UndistortFromInside(
          Eigen::Vector2f(fx_inv() * (width_ - 1) + cx_inv(), fy_inv() * y + cy_inv()),
          &converged, &second_best_result, &second_best_available);
      update_radius(nxy, second_best_result,
                    converged, second_best_available, kIncreaseFactor,
                    &result, &maximum_result);
    }
    
    radius_cutoff_squared_= (result < maximum_result) ? result : maximum_result;
    radius_cutoff_squared_ += 1;
  }

  inline const float radius_cutoff_squared() const{
    return radius_cutoff_squared_;
  }
  
 protected:
  Eigen::Vector2f* undistortion_lookup_;
  float radius_cutoff_squared_;
};

}  // namespace camera
