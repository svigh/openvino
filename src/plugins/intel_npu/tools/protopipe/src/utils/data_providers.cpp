//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "data_providers.hpp"

#include <sstream>

#include "utils.hpp"
#include "utils/error.hpp"

UniformGenerator::UniformGenerator(double low, double high, int seed): m_low(low), m_high(high), m_seed(seed) {
    ASSERT(low <= high);
}

void UniformGenerator::generate(cv::Mat& mat) {
    cv::setRNGSeed(m_seed);
    cv::randu(mat, m_low, m_high);
}

std::string UniformGenerator::str() const {
    std::stringstream ss;
    ss << "{dist: uniform, range: [" << m_low << ", " << m_high << "]}";
    return ss.str();
}

RandomProvider::RandomProvider(IRandomGenerator::Ptr impl, const std::vector<int>& dims, const int depth)
        : m_impl(impl), m_dims(dims), m_depth(depth) {
}

// NB: G-API's ONNX and OV backends expect a 2D HxW Mat with channels() set to the number of
// image channels for U8 inputs, so they can determine the tensor layout (NCHW vs NHWC).
// An N-D cv::Mat always reports channels()==1 regardless of the actual tensor shape, which
// causes the layout deduction in gonnxbackend.cpp (preprocess()) to throw
// "Couldn't identify input tensor layout".
//
// For a 4D U8 tensor {N, D1, D2, D3}: if D3 < D1 and D3 < D2 the last dim is the
// channels dim (NHWC); otherwise the second dim is channels (NCHW).
// For a 3D U8 tensor {D0, D1, D2}: analogously HWC vs CHW.
// In both cases the result is a 2D Mat(H, W, CV_8UC(C)).
static bool needsImageConversion(int depth, size_t ndims) {
    return depth == CV_8U && (ndims == 4u || ndims == 3u);
}

struct ImageLayout {
    int C, H, W;
};

static ImageLayout deduceLayout(const std::vector<int>& dims) {
    if (dims.size() == 4u) {
        // dims = {N, D1, D2, D3}
        const int D1 = dims[1], D2 = dims[2], D3 = dims[3];
        if (D3 < D1 && D3 < D2) {
            return {D3, D1, D2};  // NHWC
        }
        return {D1, D2, D3};  // NCHW
    }
    // dims.size() == 3: {D0, D1, D2}
    const int D0 = dims[0], D1 = dims[1], D2 = dims[2];
    if (D2 < D0 && D2 < D1) {
        return {D2, D0, D1};  // HWC
    }
    return {D0, D1, D2};  // CHW
}

void RandomProvider::pull(cv::Mat& mat) {
    if (needsImageConversion(m_depth, m_dims.size())) {
        const auto layout = deduceLayout(m_dims);
        mat.create(layout.H, layout.W, CV_MAKETYPE(m_depth, layout.C));
    } else {
        utils::createNDMat(mat, m_dims, m_depth);
    }
    m_impl->generate(mat);
}

cv::GMatDesc RandomProvider::desc() {
    if (m_dims.size() == 2u) {
        return cv::GMatDesc{m_depth, 1, cv::Size(m_dims[1], m_dims[0])};
    }
    if (needsImageConversion(m_depth, m_dims.size())) {
        const auto layout = deduceLayout(m_dims);
        return cv::GMatDesc{m_depth, layout.C, cv::Size(layout.W, layout.H)};
    }
    return cv::GMatDesc{m_depth, m_dims};
}

CircleBuffer::CircleBuffer(const std::vector<cv::Mat>& buffer): m_buffer(buffer), m_pos(0u) {
    ASSERT(!m_buffer.empty());
}

CircleBuffer::CircleBuffer(std::vector<cv::Mat>&& buffer): m_buffer(std::move(buffer)), m_pos(0u) {
    ASSERT(!m_buffer.empty());
}

CircleBuffer::CircleBuffer(cv::Mat mat): CircleBuffer(std::vector<cv::Mat>{mat}) {
}

void CircleBuffer::pull(cv::Mat& mat) {
    m_buffer[m_pos++].copyTo(mat);
    if (m_pos == m_buffer.size()) {
        m_pos = 0;
    }
}

cv::GMatDesc CircleBuffer::desc() {
    return cv::descr_of(m_buffer[0]);
}
