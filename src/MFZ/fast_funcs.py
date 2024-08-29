import ctypes


def np_binary_array_to_string_array(binary_array):

    string_array = []

    for i in range(len(binary_array)):

        current_bits = binary_array[i]

        trunc_nb = ''.join([str(i) for i in current_bits])

        string_array.append(trunc_nb)

    return string_array


def negabinary_to_int(s: str) -> int:
    """negabinary to decimal."""
    out = 0
    for i, digit in enumerate(s[::-1]):
        if digit == "1":
            out += (-2) ** i
    return out


def float_parts(arr, exponent_width=8):

    signs = []
    exponents = []
    mantissas = []

    for num in arr:

        whole_float = bin(ctypes.c_uint32.from_buffer(ctypes.c_float(num)).value).replace('0b', '').rjust(32, '0')

        sign_bit = whole_float[0]
        exponent_bits = whole_float[1:1+exponent_width]
        mantissa_bits = whole_float[1+exponent_width::]

        signs.append(sign_bit)
        exponents.append(exponent_bits)
        mantissas.append(mantissa_bits)
    
    return signs, exponents, mantissas



def strip_and_pad(bin_array, width):

    # Join the binaries and remove the '0b' prefix
    stripped_binaries = [s.replace('0b', '') for s in bin_array]

    # Pad the binaries to width bits
    padded = [s.rjust(width, '0') for s in stripped_binaries]

    return padded


def int_to_negabinary(i: int) -> str:
    """decimal to negabinary."""
    Schroeppel = 0xAAAAAAAA # this is 10101010101010101010101010101010 in binary

    return bin((i + Schroeppel) ^ Schroeppel).replace("0b", "").rjust(24, "0")