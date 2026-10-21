#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// PyTorch bindings validate CUDA tensors and call visibility kernels in place.

// CUDA z-buffer declaration.
at::Tensor depth_image_cuda(at::Tensor input_uv, at::Tensor input_depth, at::Tensor output, unsigned int size, unsigned int width,unsigned int height);
// Fixed-neighborhood visibility filter declaration.
at::Tensor visibility_filter_cuda(at::Tensor input, at::Tensor output, unsigned int width, unsigned int height, unsigned int threshold);
// Geometry filter with intrinsic=[fx,fy,cx,cy] and pixel radius.
at::Tensor visibility_filter_cuda2(at::Tensor input_depth, at::Tensor intrinsic, at::Tensor output, unsigned int width,unsigned int height, float threshold, unsigned int radius);
// In-place 2D flow downsampling declaration.
at::Tensor downsample_flow_cuda(at::Tensor input_uv, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel);
// In-place int32 mask downsampling declaration.
at::Tensor downsample_mask_cuda(at::Tensor input_mask, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel);
// In-place nonzero-depth downsampling declaration.
at::Tensor downsample_depth_cuda(at::Tensor input_depth, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel);

// Use the current x.is_cuda() API.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor depth_image(at::Tensor input_uv, at::Tensor input_depth, at::Tensor output, unsigned int size, unsigned int width,unsigned int height) {
    // Keep the nearest metric depth per integer pixel.
    CHECK_INPUT(input_uv);
    CHECK_INPUT(input_depth);
    CHECK_INPUT(output);
    return depth_image_cuda(input_uv, input_depth, output, size, width, height);
}

at::Tensor visibility_filter(at::Tensor input, at::Tensor output, unsigned int width, unsigned int height, unsigned int threshold){
    // Remove occluded pixels with a fixed depth neighborhood.
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    return visibility_filter_cuda(input, output, width, height, threshold);
}

at::Tensor visibility_filter2(at::Tensor input_depth, at::Tensor intrinsic, at::Tensor output, unsigned int width, unsigned int height, float threshold, unsigned int radius){
    // Filter occlusion by back-projecting neighbors with camera intrinsics.
    CHECK_INPUT(input_depth);
    CHECK_INPUT(intrinsic);
    CHECK_INPUT(output);
    return visibility_filter_cuda2(input_depth, intrinsic, output, width, height, threshold, radius);
}

at::Tensor downsample_flow(at::Tensor input_uv, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    // Aggregate pixel flow over kernel-sized windows.
    CHECK_INPUT(input_uv);
    CHECK_INPUT(output);
    return downsample_flow_cuda(input_uv, output, width_out, height_out, kernel);
}

at::Tensor downsample_mask(at::Tensor input_mask, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    // Downsample an integer validity mask over kernel-sized windows.
    CHECK_INPUT(input_mask);
    CHECK_INPUT(output);
    return downsample_mask_cuda(input_mask, output, width_out, height_out, kernel);
}

at::Tensor downsample_depth(at::Tensor input_depth, at::Tensor output, unsigned int width_out ,unsigned int height_out, unsigned int kernel) {
    // Spatially average nonzero depth over kernel-sized windows.
    CHECK_INPUT(input_depth);
    CHECK_INPUT(output);
    return downsample_depth_cuda(input_depth, output, width_out, height_out, kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_image", &depth_image, "Generate Depth Image(CUDA)");
  m.def("visibility", &visibility_filter, "visibility_filter (CUDA)");
  m.def("visibility2", &visibility_filter2, "visibility_filter2 (CUDA)");
  m.def("downsample_flow", &downsample_flow, "downsample_flow (CUDA)");
  m.def("downsample_mask", &downsample_mask, "downsample_mask (CUDA)");
  m.def("downsample_depth", &downsample_depth, "downsample_depth (CUDA)");
}
