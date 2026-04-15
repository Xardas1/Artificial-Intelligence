import numpy as np
from numpy.lib.stride_tricks import as_strided

def create_kernels(number_to_create, depth):
    every_kernel = []
    for f_u in range(number_to_create):
        every_kernel.append(np.random.rand(3, 3, depth))
    return np.array(every_kernel)

def reshape_dimensions_convolution(input_height, input_width, filter_height, filter_width, stride):
    output_height = int(((input_height - filter_height) / stride) + 1)
    output_width  = int(((input_width - filter_width)  / stride) + 1)
    return (output_height, output_width)

def reshape_dimensions_pooling(input_height, input_width, f, s):
    output_height = int(((input_height - f) / s) + 1)
    output_width = int(((input_width - f) / s) + 1)
    return (output_height, output_width)

def forward_pass(image, bias1, bias2):
    shape = image.shape
    kernels = create_kernels(32, 4)
    kernels2 =  create_kernels(64, 32)
    to_stack_values = []
    dimensions = reshape_dimensions_convolution(shape[0], shape[1], 3, 3, 1) 
    for i in range(len(kernels)):
        to_stack_values.append(convolve_one_step_vectorized(image, kernels[i], 36))
    stacked_filters = np.stack(to_stack_values, axis = 2)
    first_ReLU = ReLUv2(stacked_filters)
    pool_to_reshape = reshape_dimensions_pooling(first_ReLU.shape[0], first_ReLU.shape[1], 2, 2)
    first_max_pool = np.array(max_pooling(first_ReLU)).reshape(pool_to_reshape[0], pool_to_reshape[1], 32)
    to_stack_values.clear()
    shape2 = first_max_pool.shape
    for f in range(len(kernels2)):
        to_stack_values.append(convolve_one_step_vectorized(first_max_pool, kernels2[f], 288))
    stacked_filters2 = np.stack(to_stack_values, axis = 2)
    second_ReLU = ReLUv2(stacked_filters2)
    pool_to_reshape2 = reshape_dimensions_pooling(second_ReLU.shape[0], second_ReLU.shape[1], 2, 2)
    second_max_pool = np.array(max_pooling(second_ReLU)).reshape(pool_to_reshape2[0], pool_to_reshape2[1], 64)
    fully_connected = fully_connected_vectorized(second_max_pool, 120, bias1)
    classification = fully_connected_vectorized(fully_connected, 10, bias2)
    normalized_classification = classification / 1e13
    softmax = More_Robust_Softmax(normalized_classification, 1000)
    return softmax

def backward_pass():
    pass

def convolve_one_step(input_image, kernel_used):
    intermediate_values = []
    dimensions = input_image.shape
    dimensions_kernel = kernel_used.shape
    final_values = []
    dimensions_to_reshape = reshape_dimensions_convolution(dimensions[0], dimensions[1], dimensions_kernel[0], dimensions_kernel[1], 1)
    for i in range(dimensions[0] - 2):
        for j in range(dimensions[1] - 2):
            for k in range(dimensions[2] ):
                slice_of_image = input_image[i : i + 3, j : j + 3, k]
                real_kernel = kernel_used[:, :, k]
                value = np.vdot(slice_of_image, real_kernel)
                intermediate_values.append(value)
                if len(intermediate_values) == dimensions[2]:
                    final_values.append(sum_dimension(intermediate_values))
                    intermediate_values.clear()
    return np.array(final_values).reshape(dimensions_to_reshape)

def sum_dimension(list_of_three):
    return sum(list_of_three)

def max_pooling(image_array):
    pooling_values = []
    for m in range(0, image_array.shape[0], 2):
        for n in range(0, image_array.shape[1], 2):
            for o in range(image_array.shape[2]):
                if n + 2 > image_array.shape[1]:
                    slice_of_pooling = image_array[m : m + 2, n - 1: n + 1 , o]
                else:
                    slice_of_pooling = image_array[m : m + 2, n : n + 2 , o]
                pooling_values.append(slice_of_pooling.max())
    return pooling_values

def ReLUv1(x):
    after_relu = []
    for r in x:
        if r > 0:
            after_relu.append(r)
        else:
            after_relu.append(0)
    return after_relu

def ReLUv2(x):
    return np.maximum(0, x)

def fully_connected(after_last_max_pool, number_of_nodes, bias):
    almost_values = []
    nodes_values = []
    shape_of_flatten = np.prod(after_last_max_pool.shape)
    flatten_array = after_last_max_pool.reshape(shape_of_flatten)
    weights = np.random.rand(len(flatten_array), number_of_nodes)
    for x in range(weights.shape[1]):
        for y in range(weights.shape[0]):
            if y == weights.shape[0] - 1:
                almost_values.append(weights[y : y + 1, x] * flatten_array[y])
                nodes_values.append(sum(almost_values) + bias[x])
                almost_values.clear()
            else:
                almost_values.append(weights[y : y + 1, x] * flatten_array[y] + bias[y])
    return np.squeeze(np.array(nodes_values))

def fully_connected_vectorized(last_max_pool, nodes, bias):
    shape_to_flatten = np.prod(last_max_pool.shape)
    flatten_input = last_max_pool.reshape(shape_to_flatten, 1)
    weights = np.random.rand(nodes, len(flatten_input))
    output = np.matmul(weights, flatten_input)
    return output + bias

def reshape_to_matrix_image(height, width, kernel_height, kernel_width, depth):
    number_of_patches = np.prod(reshape_dimensions_convolution(height, width, kernel_height, kernel_width, 1))
    each_patch_size = kernel_height * kernel_width * depth
    return (number_of_patches, each_patch_size)

def convolve_one_step_vectorized(imput_image, kernel_used, matmuldim):
    shape_of_image = imput_image.shape
    kernel_used_shape = kernel_used.shape
    dim_to_reshape = reshape_dimensions_convolution(shape_of_image[0], shape_of_image[1], kernel_used_shape[0], kernel_used_shape[1], 1)
    reshaped_image  = reshape_to_matrix_valid(imput_image, (3, 3), 1, -1, matmuldim)
    reshaped_kernel = np.squeeze(reshape_to_matrix_valid(kernel_used, (3, 3), 1, -1, 1))
    vectorized_output = np.matmul(reshaped_image, reshaped_kernel)
    return vectorized_output.reshape(dim_to_reshape)

def reshape_to_matrix_valid(image, patch_size, stride, rA, rB):
    shape = (image.shape[0] - patch_size[0] + 1,
             image.shape[1] - patch_size[1] + 1,
             patch_size[0],
             patch_size[1],
             image.shape[2])
    strides = (image.strides[0],
               image.strides[1],
               image.strides[0],
               image.strides[1],
               image.strides[2])
    patches = as_strided(image, shape, strides)
    return patches.reshape(rA, rB)

def Softmax(values):
    numerator = np.exp(values)
    denominator = np.sum(numerator)
    return numerator / denominator

def More_Robust_Softmax(values, T):
    numerator = np.exp((values - np.max(values)) / T)
    denominator = np.sum(numerator)
    return numerator / denominator

def categorical_cross_entropy(y1, pi, epsilon=1e-12):
    pi = np.clip(pi, epsilon, 1. - epsilon)
    return -np.sum(np.log(pi) * y1)

