def generate_stack_array_data(name, data, dtype="qint32_t"):
    """Generate a C++ stack array from a Python list of data."""
    data_str = ', '.join([f"{int(x)}" for x in data])
    return f"{dtype} {name}[] = {{ {data_str} }};\n"


def generate_stack_array(name, len, dtype="qint32_t"):
    """Generate a C++ stack array from a Python list of data."""
    return f"{dtype} {name}[{len}] ;\n"


def generate_tensor_data(name: str, shape: tuple, data : list, dtype="qint32_t"):
    """Generate C++ tensor initialization and allocation."""
    shape_str = ', '.join(map(str, shape))
    string = ""
    string += f"{generate_stack_array_data(name + "_data", data)}"
    string += f"\tTensor<{dtype}> {name}{{ {shape_str} }};\n\t{name}.allocateCustom({name + "_data"}, {len(data)});\n"
    return string


def generate_tensor_empty(name: str, shape: tuple, size: int, dtype="qint32_t"):
    """Generate C++ tensor initialization and allocation."""
    shape_str = ', '.join(map(str, shape))
    string = ""
    string += f"{generate_stack_array(name + "_data", size)}"
    string += f"\tTensor<{dtype}> {name}{{ {shape_str} }};\n\t{name}.allocateCustom({name + "_data"}, {size});\n"
    return string


def generate_decl_tensor_range(name: str, shape: tuple, size: int, dtype="qint32_t"):
    """Generate C++ tensor initialization and allocation."""
    shape_str = ', '.join(map(str, shape))
    string = ""
    string += f"{generate_stack_array(name + "_data", size)}"
    string += f"\tTensor<{dtype}> {name}{{ {shape_str} }};\n\t{name}.allocateCustom({name + "_data"}, {size});\n\tArange({name},1);\n"
    return string


test_macro = "EML_ASSERT_TENSOR_EQUALS"