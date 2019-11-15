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


#include "camera/camera_simple_pinhole.h"

#include <glog/logging.h>

namespace camera {
SimplePinholeCamera::SimplePinholeCamera(int width, int height, float f,
                                         float cx, float cy)
    : CameraBase(width, height, f, f, cx, cy, Type::kSimplePinhole) {}

SimplePinholeCamera::SimplePinholeCamera(int width, int height, const float* parameters)
    : CameraBase(width, height, parameters[0], parameters[0], parameters[1], parameters[2], Type::kSimplePinhole) {}

CameraBase* SimplePinholeCamera::ScaledBy(float factor) const {
  CHECK_NE(factor, 0.0f);
  int scaled_width = static_cast<int>(factor * width_);
  int scaled_height = static_cast<int>(factor * height_);
  return new SimplePinholeCamera(scaled_width, scaled_height,
                                 factor * fx(),
                                 factor * (cx() + 0.5f) - 0.5f,
                                 factor * (cy() + 0.5f) - 0.5f);
}

CameraBase* SimplePinholeCamera::ShiftedBy(float cx_offset, float cy_offset) const {
  return new SimplePinholeCamera(width_, height_, fx(), cx() + cx_offset,
                           cy() + cy_offset);
}
}  // namespace camera
