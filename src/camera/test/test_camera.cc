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


#include <memory>

#include <Eigen/StdVector>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

#include "camera/camera_models.h"

namespace {
template <typename Camera>
void UndistortAndDistortImageCornersTest(const Camera& test_camera) {
  // Test the corners of the image.
  std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> test_points;
  test_points.push_back(Eigen::Vector2f(0, 0));
  test_points.push_back(Eigen::Vector2f(test_camera.width() - 1, 0));
  test_points.push_back(Eigen::Vector2f(0, test_camera.height() - 1));
  test_points.push_back(Eigen::Vector2f(test_camera.width() - 1,
                                        test_camera.height() - 1));
  
  for (unsigned int i = 0; i < test_points.size(); ++i) {
    // Undistort point and distort the result again.
    Eigen::Vector2f nxy = Eigen::Vector2f(
        test_camera.fx_inv() * test_points[i].x() + test_camera.cx_inv(),
        test_camera.fy_inv() * test_points[i].y() + test_camera.cy_inv());
    Eigen::Vector2f result = test_camera.Distort(test_camera.Undistort(nxy));
    
    // Verify that the result is identical to the input.
    EXPECT_NEAR(nxy.x(), result.x(), 1e-5f)
        << "Test failed for " << test_points[i].x() << ", " << test_points[i].y()
        << " (undistorted: " << test_camera.Undistort(nxy).x() << ", "
        << test_camera.Distort(nxy).y() << ")";
    EXPECT_NEAR(nxy.y(), result.y(), 1e-5f)
        << "Test failed for " << test_points[i].x() << ", " << test_points[i].y()
        << " (undistorted: " << test_camera.Undistort(nxy).x() << ", "
        << test_camera.Distort(nxy).y() << ")";
    
    // In case of failure, show a debug image.
    if (fabs(nxy.x() - result.x()) > 1e-5f ||
        fabs(nxy.y() - result.y()) > 1e-5f) {
      int failing_pixels = 0;
      cv::Mat_<cv::Vec3b> debug_image(test_camera.height(), test_camera.width(), cv::Vec3b(0,0,0));
      for (int test_y = 0; test_y < test_camera.height(); ++ test_y) {
        for (int test_x = 0; test_x < test_camera.width(); ++ test_x) {
          Eigen::Vector2f test_nxy = Eigen::Vector2f(test_camera.fx_inv() * test_x + test_camera.cx_inv(),
                                        test_camera.fy_inv() * test_y + test_camera.cy_inv());
          Eigen::Vector2f test_result = test_camera.Distort(test_camera.Undistort(test_nxy));
          bool ok = !(fabs(test_nxy.x() - test_result.x()) > 1e-5f ||
                      fabs(test_nxy.y() - test_result.y()) > 1e-5f);
          if (!ok) {
            ++ failing_pixels;
            debug_image(test_y, test_x) = cv::Vec3b(0, 0, 255);
          }
          if(test_x % 20 == 0 || test_y % 20 == 0){
            Eigen::Vector2f distorted = test_camera.Distort(test_nxy);
            int test_x2 = round(test_camera.fx() * distorted.x() + test_camera.cx());
            int test_y2 = round(test_camera.fy() * distorted.y() + test_camera.cy());
            if(test_x2 > 0 && test_y2 > 0 && test_x2 < test_camera.width()-1 && test_y2 < test_camera.height()-1)
              debug_image(test_y2, test_x2) = cv::Vec3b(255, 255, 255);
          }

        }
      }
      std::ostringstream window_title;
      window_title << "Red shows where undistort(distort(x)) fails (" << failing_pixels << " pixels)";
      cv::imshow(window_title.str(), debug_image);
      cv::waitKey(0);
    }
  }
}

template <typename Camera>
void DistortAndUndistortTest(const Camera& test_camera) {
  const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> kTestPoints = {
      Eigen::Vector2f(0.0f, 0.0f), // Eigen::Vector2f(1.0f, 1.0f), // This was outside the cutoff radius for some cameras, resulting in errors.
      Eigen::Vector2f(0.0f, 1.0f), Eigen::Vector2f(1.0f, 0.0f),
      Eigen::Vector2f(0.5f, 0.5f), Eigen::Vector2f(0.1f, 0.2f),
      Eigen::Vector2f(0.8f, 0.9f), Eigen::Vector2f(0.5f, 0.6f),
      Eigen::Vector2f(0.1f, 0.9f)};
  
  for (unsigned int i = 0; i < kTestPoints.size(); ++i) {
    float x = kTestPoints[i].x() * test_camera.width();
    float y = kTestPoints[i].y() * test_camera.height();
    Eigen::Vector2f nxy = Eigen::Vector2f(test_camera.fx_inv() * x + test_camera.cx_inv(),
                             test_camera.fy_inv() * y + test_camera.cy_inv());
    Eigen::Vector2f result = test_camera.Undistort(test_camera.Distort(nxy));
    
    EXPECT_NEAR(nxy.x(), result.x(), 1e-5f)
        << "Test failed for " << kTestPoints[i].x() << ", " << kTestPoints[i].y()
        << " (distorted: " << test_camera.Distort(nxy).x() << ", "
        << test_camera.Distort(nxy).y() << ")";
    EXPECT_NEAR(nxy.y(), result.y(), 1e-5f)
        << "Test failed for " << kTestPoints[i].x() << ", " << kTestPoints[i].y()
        << " (distorted: " << test_camera.Distort(nxy).x() << ", "
        << test_camera.Distort(nxy).y() << ")";
  }
}

template <typename Camera>
void NumericalProjectionToNormalizedTextureCoordinatesDerivative(
    const Camera& camera,
    const Eigen::Vector3f& at,
    Eigen::Vector3f* out_x,
    Eigen::Vector3f* out_y) {
  const float kStep = 0.001f;
  const float kTwoSteps = 2 * kStep;
  
  Eigen::Vector3f at_plus_x = Eigen::Vector3f(at.x() + kStep, at.y(), at.z());
  Eigen::Vector2f proj_plus_x = camera.ProjectToNormalizedTextureCoordinates(
      Eigen::Vector2f(at_plus_x.x() / at_plus_x.z(), at_plus_x.y() / at_plus_x.z()));
  Eigen::Vector3f at_minus_x = Eigen::Vector3f(at.x() - kStep, at.y(), at.z());
  Eigen::Vector2f proj_minus_x = camera.ProjectToNormalizedTextureCoordinates(
      Eigen::Vector2f(at_minus_x.x() / at_minus_x.z(), at_minus_x.y() / at_minus_x.z()));
  
  Eigen::Vector3f at_plus_y = Eigen::Vector3f(at.x(), at.y() + kStep, at.z());
  Eigen::Vector2f proj_plus_y = camera.ProjectToNormalizedTextureCoordinates(
      Eigen::Vector2f(at_plus_y.x() / at_plus_y.z(), at_plus_y.y() / at_plus_y.z()));
  Eigen::Vector3f at_minus_y = Eigen::Vector3f(at.x(), at.y() - kStep, at.z());
  Eigen::Vector2f proj_minus_y = camera.ProjectToNormalizedTextureCoordinates(
      Eigen::Vector2f(at_minus_y.x() / at_minus_y.z(), at_minus_y.y() / at_minus_y.z()));
  
  Eigen::Vector3f at_plus_z = Eigen::Vector3f(at.x(), at.y(), at.z() + kStep);
  Eigen::Vector2f proj_plus_z = camera.ProjectToNormalizedTextureCoordinates(
      Eigen::Vector2f(at_plus_z.x() / at_plus_z.z(), at_plus_z.y() / at_plus_z.z()));
  Eigen::Vector3f at_minus_z = Eigen::Vector3f(at.x(), at.y(), at.z() - kStep);
  Eigen::Vector2f proj_minus_z = camera.ProjectToNormalizedTextureCoordinates(
      Eigen::Vector2f(at_minus_z.x() / at_minus_z.z(), at_minus_z.y() / at_minus_z.z()));
  
  *out_x = Eigen::Vector3f((proj_plus_x.x() - proj_minus_x.x()) / kTwoSteps,
                           (proj_plus_y.x() - proj_minus_y.x()) / kTwoSteps,
                           (proj_plus_z.x() - proj_minus_z.x()) / kTwoSteps);
  *out_y = Eigen::Vector3f((proj_plus_x.y() - proj_minus_x.y()) / kTwoSteps,
                           (proj_plus_y.y() - proj_minus_y.y()) / kTwoSteps,
                           (proj_plus_z.y() - proj_minus_z.y()) / kTwoSteps);
}

template <typename Camera>
void CheckProjectionToNormalizedTextureCoordinatesDerivative(
    const Camera& camera,
    const Eigen::Vector3f& at) {
  Eigen::Vector3f result_x, result_y;
  camera.ProjectionToNormalizedTextureCoordinatesDerivative(at, &result_x, &result_y);
  Eigen::Vector3f numerical_x, numerical_y;
  NumericalProjectionToNormalizedTextureCoordinatesDerivative(camera, at, &numerical_x,
                                            &numerical_y);
  
  EXPECT_NEAR(result_x.x(), numerical_x.x(), 1e-3f);
  EXPECT_NEAR(result_x.y(), numerical_x.y(), 1e-3f);
  EXPECT_NEAR(result_x.z(), numerical_x.z(), 1e-3f);
  EXPECT_NEAR(result_y.x(), numerical_y.x(), 1e-3f);
  EXPECT_NEAR(result_y.y(), numerical_y.y(), 1e-3f);
  EXPECT_NEAR(result_y.z(), numerical_y.z(), 1e-3f);
}

template <typename Camera>
void NumericalProjectionToImageCoordinatesDerivative(
    const Camera& camera,
    const Eigen::Vector3f& at,
    Eigen::Vector3f* out_x,
    Eigen::Vector3f* out_y) {
  const float kStep = 0.001f;
  const float kTwoSteps = 2 * kStep;
  
  Eigen::Vector3f at_plus_x = Eigen::Vector3f(at.x() + kStep, at.y(), at.z());
  Eigen::Vector2f proj_plus_x = camera.ProjectToImageCoordinates(
      Eigen::Vector2f(at_plus_x.x() / at_plus_x.z(), at_plus_x.y() / at_plus_x.z()));
  Eigen::Vector3f at_minus_x = Eigen::Vector3f(at.x() - kStep, at.y(), at.z());
  Eigen::Vector2f proj_minus_x = camera.ProjectToImageCoordinates(
      Eigen::Vector2f(at_minus_x.x() / at_minus_x.z(), at_minus_x.y() / at_minus_x.z()));
  
  Eigen::Vector3f at_plus_y = Eigen::Vector3f(at.x(), at.y() + kStep, at.z());
  Eigen::Vector2f proj_plus_y = camera.ProjectToImageCoordinates(
      Eigen::Vector2f(at_plus_y.x() / at_plus_y.z(), at_plus_y.y() / at_plus_y.z()));
  Eigen::Vector3f at_minus_y = Eigen::Vector3f(at.x(), at.y() - kStep, at.z());
  Eigen::Vector2f proj_minus_y = camera.ProjectToImageCoordinates(
      Eigen::Vector2f(at_minus_y.x() / at_minus_y.z(), at_minus_y.y() / at_minus_y.z()));
  
  Eigen::Vector3f at_plus_z = Eigen::Vector3f(at.x(), at.y(), at.z() + kStep);
  Eigen::Vector2f proj_plus_z = camera.ProjectToImageCoordinates(
      Eigen::Vector2f(at_plus_z.x() / at_plus_z.z(), at_plus_z.y() / at_plus_z.z()));
  Eigen::Vector3f at_minus_z = Eigen::Vector3f(at.x(), at.y(), at.z() - kStep);
  Eigen::Vector2f proj_minus_z = camera.ProjectToImageCoordinates(
      Eigen::Vector2f(at_minus_z.x() / at_minus_z.z(), at_minus_z.y() / at_minus_z.z()));
  
  *out_x = Eigen::Vector3f((proj_plus_x.x() - proj_minus_x.x()) / kTwoSteps,
                           (proj_plus_y.x() - proj_minus_y.x()) / kTwoSteps,
                           (proj_plus_z.x() - proj_minus_z.x()) / kTwoSteps);
  *out_y = Eigen::Vector3f((proj_plus_x.y() - proj_minus_x.y()) / kTwoSteps,
                           (proj_plus_y.y() - proj_minus_y.y()) / kTwoSteps,
                           (proj_plus_z.y() - proj_minus_z.y()) / kTwoSteps);
}

template <typename Camera>
void CheckProjectionToImageCoordinatesDerivative(const Camera& camera,
                                                 const Eigen::Vector3f& at) {
  Eigen::Vector3f result_x, result_y;
  camera.ProjectionToImageCoordinatesDerivative(at, &result_x, &result_y);
  Eigen::Vector3f numerical_x, numerical_y;
  NumericalProjectionToImageCoordinatesDerivative(camera, at, &numerical_x,
                                                  &numerical_y);
  
  EXPECT_NEAR(result_x.x(), numerical_x.x(), 250 * 1e-3f)
      << "Test failed for point " << at.x() << ", " << at.y() << ", " << at.z();
  EXPECT_NEAR(result_x.y(), numerical_x.y(), 250 * 1e-3f);
  EXPECT_NEAR(result_x.z(), numerical_x.z(), 250 * 1e-3f);
  EXPECT_NEAR(result_y.x(), numerical_y.x(), 250 * 1e-3f);
  EXPECT_NEAR(result_y.y(), numerical_y.y(), 250 * 1e-3f);
  EXPECT_NEAR(result_y.z(), numerical_y.z(), 250 * 1e-3f);
}

template <typename Camera>
void TestProjectionToImageCoordinatesDerivative(const Camera& camera) {
  const Eigen::Vector3f kTest3DPoints[] = {
      Eigen::Vector3f(0.0f, 0.0f, 3.0f),
      // Eigen::Vector3f(1.0f, 2.5f, 2.0f), // This was outside the cutoff radius for some cameras, resulting in errors.
      Eigen::Vector3f(1.0f, 3.0f, 8.0f),
      Eigen::Vector3f(-0.1f, 0.7f, -0.8f)};
  
  for (unsigned int i = 0; i < sizeof(kTest3DPoints) / sizeof(kTest3DPoints[0]); ++i) {
    if (kTest3DPoints[i].x() == 0.0f && kTest3DPoints[i].y() == 0.0f &&
        camera.type() == camera::CameraBase::Type::kFOV) {
      continue;
    }
    CheckProjectionToImageCoordinatesDerivative(camera, kTest3DPoints[i]);
  }
}

template<class Camera>
void CreateDeltaCamera(
    const Camera& base_camera,
    const float* delta,
    std::shared_ptr<Camera>* result) {
  constexpr int kParameterCount = Camera::ParameterCount();
  float parameters[kParameterCount];
  base_camera.GetParameters(parameters);
  for (int i = 0; i < kParameterCount; ++ i) {
    parameters[i] += delta[i];
  }
  result->reset(new Camera(base_camera.width(), base_camera.height(), parameters));
}

template<class Camera>
void NumericalProjectionToImageCoordinatesDerivativeByIntrinsics(
    const Camera& base_camera,
    const float nx, const float ny,
    const float* delta,
    float* result_x, float* result_y) {
  constexpr int kNumParameters = Camera::ParameterCount();
  const float kStep = 0.025f;
  const float kTwoSteps = 2 * kStep;
  
  std::shared_ptr<Camera> plus, minus;
  float plus_delta[kNumParameters], minus_delta[kNumParameters];
  for (int i = 0; i < kNumParameters; ++ i) {
    plus_delta[i] = kStep * delta[i];
    minus_delta[i] = -1.f * plus_delta[i];
  }
  CreateDeltaCamera(base_camera, plus_delta, &plus);
  CreateDeltaCamera(base_camera, minus_delta, &minus);

  Eigen::Vector2f proj_plus_x = plus->ProjectToImageCoordinates(Eigen::Vector2f(nx, ny));
  Eigen::Vector2f proj_minus_x = minus->ProjectToImageCoordinates(Eigen::Vector2f(nx, ny));
  *result_x = (proj_plus_x.x() - proj_minus_x.x()) / kTwoSteps;
  *result_y = (proj_plus_x.y() - proj_minus_x.y()) / kTwoSteps;
}

template<class Camera>
void TestProjectionToImageCoordinatesDerivativeByIntrinsics(
    const Camera& camera) {
  constexpr int kNumParameters = Camera::ParameterCount();
  const Eigen::Vector3f kTest3DPoints[] = {
      Eigen::Vector3f(0.0f, 0.0f, 3.0f),
      Eigen::Vector3f(1.0f, 2.5f, 4.0f),
      Eigen::Vector3f(1.0f, 3.0f, 8.0f),
      Eigen::Vector3f(-0.1f, 0.4f, 0.8f)};
  float result_x[kNumParameters], result_y[kNumParameters];
  for (unsigned int i = 0; i < sizeof(kTest3DPoints) / sizeof(kTest3DPoints[0]); ++i) {
    const float nx = kTest3DPoints[i].x() / kTest3DPoints[i].z();
    const float ny = kTest3DPoints[i].y() / kTest3DPoints[i].z();
  
    Eigen::Vector2f pxy = camera.ProjectToImageCoordinates(Eigen::Vector2f(nx, ny));
    if (pxy.x() < 0 || pxy.y() < 0 ||
        pxy.x() >= camera.width() ||
        pxy.y() >= camera.height()) {
      LOG(WARNING) << "Test point " << i << " does not project into the camera image.";
      continue;
    }
    
    camera.ProjectionToImageCoordinatesDerivativeByIntrinsics(
        kTest3DPoints[i], result_x, result_y);
    
    float numerical_x, numerical_y;
    float delta[kNumParameters];
    memset(delta, 0, kNumParameters * sizeof(delta[0]));
    for (int c = 0; c < kNumParameters; ++ c) {
      delta[c] = 1;
      NumericalProjectionToImageCoordinatesDerivativeByIntrinsics(
          camera, nx, ny, delta, &numerical_x, &numerical_y);
      delta[c] = 0;
      
      EXPECT_NEAR(result_x[c], numerical_x, 1.5e-3f)
          << "Failure for point " << i << ", component " << c;
      EXPECT_NEAR(result_y[c], numerical_y, 1.5e-3f)
          << "Failure for point " << i << ", component " << c;
    }
  }
}

template<class Camera>
void TestProjectionToNormalizedTextureCoordinatesDerivative(const Camera& camera) {
  const Eigen::Vector3f kTest3DPoints[] = {
      Eigen::Vector3f(0.0f, 0.0f, 3.0f),
      // Eigen::Vector3f(1.0f, 2.5f, 2.0f), // This was outside the cutoff radius for some cameras, resulting in errors.
      Eigen::Vector3f(1.0f, 3.0f, 8.0f),
      Eigen::Vector3f(-0.1f, 0.7f, -0.8f)};
  for (unsigned int i = 0; i < sizeof(kTest3DPoints) / sizeof(kTest3DPoints[0]); ++i) {
    if (kTest3DPoints[i].x() == 0.0f && kTest3DPoints[i].y() == 0.0f &&
        camera.type() == camera::CameraBase::Type::kFOV) {
      continue;
    }
    CheckProjectionToNormalizedTextureCoordinatesDerivative(
        camera, kTest3DPoints[i]);
  }
}

template<class Camera>
void TestParameterStorage() {
  constexpr int kNumParameters = Camera::ParameterCount();
  
  float parameters[kNumParameters];
  for (int i = 0; i < kNumParameters; ++ i) {
    parameters[i] = 10 * i;
  }
  Camera camera(10, 10, parameters);
  
  float results[kNumParameters];
  camera.GetParameters(results);
  for (int i = 0; i < kNumParameters; ++ i) {
    EXPECT_FLOAT_EQ(parameters[i], results[i]);
  }
}

template<class Camera>
void RunCameraModelTests(const Camera& camera) {
  // Tests whether parameters are correctly set and retrieved.
  TestParameterStorage<Camera>();
  
  // Tests whether undistortion followed by distortion results in the identity
  // function. Do not test this for the FOV camera since the corners do not
  // show anything there with the used parameters.
  if (camera.type() != camera::CameraBase::Type::kFOV) {
    UndistortAndDistortImageCornersTest(camera);
  }
  
  // Tests whether distortion followed by undistortion results in the identity
  // function.
  DistortAndUndistortTest(camera);
  
  // Compares the analytical projection-to-image-coordinates derivative to
  // numerical derivatives.
  TestProjectionToImageCoordinatesDerivative(camera);
  
  // Compares the analytical projection-to-normalized-texture-coordinates
  // derivative to numerical derivatives.
  TestProjectionToNormalizedTextureCoordinatesDerivative(camera);
  
  // Compares the analytical projection-to-image-coordinates derivative by
  // the intrinics to numerical derivatives.
  TestProjectionToImageCoordinatesDerivativeByIntrinsics(camera);
}
}  // namespace


constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;
constexpr double kFX = 250.0;
constexpr double kCX = 319.5;
constexpr double kFY = 200.0;
constexpr double kCY = 239.5;

constexpr float kOmega = 1.0f;

constexpr float kK1 = 0.23f;
constexpr float kK2 = -0.66f;
constexpr float kK3 = 0.64f;

TEST(Camera, Pinhole) {
  camera::PinholeCamera pinhole_camera(kImageWidth, kImageHeight, kFX, kFY, kCX,
                                       kCY);
  RunCameraModelTests(pinhole_camera);
}

TEST(Camera, SimplePinhole) {
  camera::SimplePinholeCamera simplepinhole_camera(kImageWidth, kImageHeight, kFX, kCX,
                                       kCY);
  RunCameraModelTests(simplepinhole_camera);
}

TEST(Camera, Radial) {
  camera::RadialCamera radial_camera(kImageWidth, kImageHeight, kFX,
                                             kFY, kCX, kCY, kK1, -kK2);
  RunCameraModelTests(radial_camera);
}

TEST(Camera, SimpleRadial) {
  camera::SimpleRadialCamera simple_radial_camera(kImageWidth, kImageHeight, 450,
                                             kCX, kCY, kK1);
  RunCameraModelTests(simple_radial_camera);
}

TEST(Camera, Polynomial) {
  camera::PolynomialCamera polynomial_camera(kImageWidth, kImageHeight, kFX,
                                             kFY, kCX, kCY, kK1, kK2, kK3);
  RunCameraModelTests(polynomial_camera);
}

TEST(Camera, FisheyeFOV) {
  camera::FisheyeFOVCamera fisheye_fov_camera(kImageWidth, kImageHeight, kFX,
                                              kFY, kCX, kCY, kOmega);
  RunCameraModelTests(fisheye_fov_camera);
}

TEST(Camera, PolynomialTangential) {
  camera::PolynomialTangentialCamera polynomial_tangential_camera(
      kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
      0.0703954, 0.000438661, -0.000680887);
  RunCameraModelTests(polynomial_tangential_camera);
}

TEST(Camera, FisheyePolynomial4) {
  camera::FisheyePolynomial4Camera
      fisheye_polynomial_4_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, 0.221184,
          0.128597, 0.0623079, 0.20419);
  RunCameraModelTests(fisheye_polynomial_4_camera);
}

TEST(Camera, FisheyePolynomialTangential) {
  camera::FisheyePolynomialTangentialCamera
      fisheye_polynomial_tangential_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6, -0.101082,
          0.0703954, 0.000438661, -0.000680887);
  RunCameraModelTests(fisheye_polynomial_tangential_camera);
}

TEST(Camera, Benchmark) {
  camera::BenchmarkCamera
      benchmark_camera(
          kImageWidth, kImageHeight, 340.926, 341.124, 302.4, 201.6,
          0.221184, 0.128597, 0.000531602, -0.000388873, 0.0623079,
          0.20419, -0.000805024, 4.07704e-05);
  RunCameraModelTests(benchmark_camera);
}
