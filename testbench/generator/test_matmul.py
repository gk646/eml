import torch
import gen_utils


def impl_gen_matmul(A_shape, B_shape, transpose_A=False, transpose_B=False):
    A_tensor = torch.arange(1, 1 + A_shape[0] * A_shape[1], dtype=torch.float32).reshape(A_shape)
    B_tensor = torch.arange(1, 1 + B_shape[0] * B_shape[1], dtype=torch.float32).reshape(B_shape)

    output_tensor = torch.tensor([])
    if transpose_A:
        output_tensor = torch.matmul(A_tensor.transpose(0, 1), B_tensor).detach().numpy()
    elif transpose_B:
        output_tensor = torch.matmul(A_tensor, B_tensor.transpose(0, 1)).detach().numpy()
    else:
        output_tensor = torch.matmul(A_tensor, B_tensor).detach().numpy()

    A_data = A_tensor.flatten().tolist()
    B_data = B_tensor.flatten().tolist()
    output_data = output_tensor.flatten().tolist()

    output_array = gen_utils.generate_stack_array_data("expected", output_data)

    A_tensor_decl = gen_utils.generate_decl_tensor_range("A", A_shape, len(A_data))
    B_tensor_decl = gen_utils.generate_decl_tensor_range("B", B_shape, len(B_data))
    C_tensor_decl = gen_utils.generate_tensor_empty("C", (A_shape[0], B_shape[1]), len(output_data))

    matmul_decl = f"Matmul{'ATrans' if transpose_A else ''}{'BTrans' if transpose_B else ''}"

    cpp_code = f"""
    {A_tensor_decl}
    {B_tensor_decl}
    {C_tensor_decl}
    {matmul_decl}(A, B,C);
    {output_array}
    Tensor<qint32_t> R{{C.shape()}};
    R.allocateCustom(expected, {len(output_data)});
    return Equals(C, R);
"""
    return cpp_code


A_shapes = [(4, 4), (3, 5), (5, 3), (5, 5)]  # Square and non-square matrices
B_shapes = [(4, 4), (5, 3), (3, 5), (5, 5)]  # Compatible shapes for each A
transpose_options = [False, True]  # Whether to transpose A or B


def gen_function(index):
    if index >= 15:
        return None

    A_shape = A_shapes[index % len(A_shapes)]
    B_shape = B_shapes[index % len(B_shapes)]

    transpose_A = (index == 0)
    transpose_B = (index == 3)

    return impl_gen_matmul(A_shape, B_shape, transpose_A, transpose_B)
