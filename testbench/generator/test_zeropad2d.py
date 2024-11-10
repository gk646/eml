import random

import torch
import gen_utils

def impl_gen_zeropad2d(input_shape, padding):
    input_size = input_shape[0] * input_shape[1]
    input_tensor_decl = gen_utils.generate_decl_tensor_range("A", input_shape, input_size)

    pad_tuple = (padding[1],padding[1],padding[0],padding[0])
    pad = torch.nn.ZeroPad2d(pad_tuple)
    input_tensor = torch.arange(1,input_size + 1).reshape(1,1,input_shape[0],input_shape[1])
    torch_output = pad(input_tensor).detach()

    output_data = torch_output.numpy()

    output_tensor_decl = gen_utils.generate_tensor_data("R", output_data.shape, output_data.flatten().tolist())

    # Generate the C++ function call
    cpp_code = f"""
    {input_tensor_decl}
    ZeroPad2D layer{{ {{ {padding[0]},{padding[1]} }} }};
    auto out = layer.forward(A);
    {output_tensor_decl}
    return Equals(R, out);
"""
    return cpp_code


# Define a list of input shapes and paddings for test cases
input_shapes = [
    (5, 4),
    (4, 5),
    (6, 6),
    (7, 8),
    (8, 8)
]

paddings = [
    (1, 1),
    (2, 3),
    (2, 2),
    (1, 0),
]


def gen_function(index):
    if index >= len(input_shapes) * len(paddings):
        return None

    random.shuffle(input_shapes)
    random.shuffle(paddings)
    # Get input shape and padding based on the index
    input_shape = input_shapes[index % len(input_shapes)]
    padding = paddings[index % len(paddings)]

    return impl_gen_zeropad2d(input_shape, padding)
