import random

import numpy as np
import torch
from torch import dtype

import gen_utils


def impl_gen_leakyrelu(input_shape, alpha):
    input_tensor_decl = gen_utils.generate_decl_tensor_range("A", input_shape, "float",-10)

    layer = torch.nn.LeakyReLU(alpha)
    size = 1
    for i in input_shape:
        size *= i
    input_tensor = torch.arange(-10, -10 +  size , dtype=torch.float32).reshape(input_shape)
    torch_output = layer(input_tensor).detach()
    output_data = torch_output.numpy()
    output_tensor_decl = gen_utils.generate_tensor_data("R", output_data.shape, output_data.flatten().tolist(), "float")

    # Generate the C++ function call
    cpp_code = f"""
    {input_tensor_decl}
    LeakyReLU layer{{{alpha}F}};
    auto out = layer.forward(A);
    {output_tensor_decl}
    {gen_utils.test_macro}(R, out);
"""
    return cpp_code


# Define a list of input shapes and paddings for test cases
input_shapes = [
    (2, 3, 5, 4),
    (1, 1, 4, 5),
    (2, 3, 6, 6),
    (1, 1, 9, 8),
]

alphas = [
    0.001, 0.1, -2.4, 1.1
]


def gen_function(index):
    if index >= len(input_shapes):
        return None

    input_shape = input_shapes[index % len(input_shapes)]
    alpha = alphas[index % len(alphas)]
    return impl_gen_leakyrelu(input_shape, alpha)
