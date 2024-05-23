import struct

import numpy as np


class BFP(object):

    def __init__(self, array=None, error_tol=None, float_format='single', exponent=None, negabin_array=None, truncation_bit=None) -> None:

        self._float_specs(float_format)

        assert array is None or isinstance(array, np.ndarray), "array must be a numpy array or None"

        if array is None:
            assert exponent is not None and negabin_array is not None, "If array is None, exponent and mantissas must be provided"
            self.block_size = len(negabin_array)

        self.exponent = exponent
        self.negabin_array = negabin_array
        self.truncation_bit = truncation_bit
        self.block_size = None
        self.error_tol = error_tol

        if array is not None:
            self.block_size = len(array)
            self._encode(array)

    def bitstream(self) -> np.ndarray:

        bitstream = np.array([], dtype=np.uint8)

        table = bytearray.maketrans(b'01', b'\x00\x01')

        if self.negabin_array == []:
            negabin_array = ['0'*(self.mantissa_width+1)]*self.block_size
        else:
            negabin_array = self.negabin_array

        for i in range(self.block_size):

            bit_bunch = bytearray(negabin_array[i], "ascii").translate(table)

            bitstream = np.append(bitstream, bit_bunch)

        return bitstream
    
    def bytestream(self) -> np.ndarray:

        bytestream = np.array([], dtype=np.uint8)

        if self.negabin_array == []:
            negabin_array = ['0'*(self.mantissa_width+1)]*self.block_size
        else:
            negabin_array = self.negabin_array

        for i in range(self.block_size):

            bit_bunch = negabin_array[i]

            bytestream = np.append(bytestream, [int(bit_bunch[0:8],2), int(bit_bunch[8:16],2), int(bit_bunch[16::],2)])

        return bytestream.astype(np.uint8)

    def float(self) -> None:

        values = []

        for bin_string in self.negabin_array:

            if self.truncation_bit is not None:
                full_bin_string = bin_string.ljust(self.mantissa_width+1, '0')
            else:
                full_bin_string = bin_string

            values.append(self._decode(full_bin_string))

        return np.array(values)

    def _encode(self, array) -> None:

        signs, exponents, mantissas = self._float_parts(array)

        common_exp = max([int(exp, 2) for exp in exponents]) + 1

        self.exponent = bin(common_exp)

        # Make new mantissas via bit shifts
        block_mantissas = []

        for exp,num in zip(exponents,mantissas):

            # Convert to integer
            int_num = int('1'+num[0:-1], 2)
            int_exp = int(exp, 2)

            exp_difference = int_exp - common_exp

            # Shift mantissa to the left
            shifted_mantissa = int_num >> (-exp_difference)
            
            # Convert back to binary
            block_mantissas.append(bin(shifted_mantissa))

        block_mantissas = self._strip_and_pad(block_mantissas, 23)
        block_mantissas = self.threshold(block_mantissas)

        if block_mantissas != []:
            negabin_array = self.nega_encode(signs, block_mantissas)
            negabin_array, self.truncation_bit = self.bit_plane_truncate(negabin_array)
        else:
            negabin_array = []
            self.truncation_bit = bin(23).replace('0b', '').rjust(8, '0')

        self.negabin_array = negabin_array
        
        if self.negabin_array == []:
            # This exponent indicates that the block is filled with ONLY zeros
            self.exponent = '11111111'

        return 

    def _decode(self, negabin_str) -> float:

        # Convert the exponent to a value
        exponent_value = int(self.exponent, 2)

        # Convert the mantissa to a value
        mantissa_value = self.negabinary_to_int(negabin_str) / 2**22

        # Calculate the value
        value = mantissa_value * 2**(exponent_value - self.exp_offset)

        return value

    def bit_plane_truncate(self, negabin_array):

        first_one_bit = np.array([negabin[::-1].find('1') for negabin in negabin_array])

        if np.all(first_one_bit == -1):
            return [], bin(0).replace('0b', '').rjust(8, '0')

        truncation_bit = np.min(first_one_bit[first_one_bit != -1])

        new_negabin_array = []

        for k in range(self.block_size):
            new_negabin_array.append(negabin_array[k][0:-truncation_bit])

        return new_negabin_array, bin(self.mantissa_width - truncation_bit).replace('0b', '').rjust(8, '0')

    def nega_encode(self, signs, mantissa_array):

        num_array = []

        for i in range(self.block_size):
            full_int = pow(-1, int(signs[i], 2)) * int(mantissa_array[i],2)

            num_array.append(self.int_to_negabinary(full_int))

        return num_array

    def int_to_negabinary(self, i: int) -> str:
        """decimal to negabinary."""
        Schroeppel = 0xAAAAAAAA # this is 10101010101010101010101010101010 in binary

        return bin((i + Schroeppel) ^ Schroeppel).replace("0b", "").rjust(24, "0")
    
    def negabinary_to_int(self, s: str) -> int:
        """negabinary to decimal."""
        out = 0
        for i, digit in enumerate(s[::-1]):
            if digit == "1":
                out += (-2) ** i
        return out

    def threshold(self, mantissa_array):

        if self.error_tol is None:
            return mantissa_array

        # Convert the error tolerance to a mantissa threshold
        
        _, exponent, mantissa = self._float_parts([self.error_tol])

        threshold_num = int('1'+mantissa[0][0:-1], 2)

        threshold_exp = int(exponent[0], 2)

        exp_difference = threshold_exp - int(self.exponent,2)

        if -exp_difference < 0:
            new_mantissas = []
            return new_mantissas

        # Shift mantissa to the left
        mantissa_threshold = threshold_num >> (-exp_difference)

        mantissa_threshold = bin(mantissa_threshold).replace('0b', '')

        new_mantissas = []

        for mantissa in mantissa_array:

            trimmed_mantissa = mantissa[-len(mantissa_threshold)::]

            remaining_mantissa = mantissa[0:-len(mantissa_threshold)]

            # Convert the mantissa to a value
            mantissa_value = int(trimmed_mantissa, 2)

            if mantissa_value < int(mantissa_threshold,2):
                new_mantissas.append(remaining_mantissa + '0'*len(mantissa_threshold))
            elif trimmed_mantissa[1::] =='':
                new_mantissas.append(remaining_mantissa + trimmed_mantissa)
            elif int(trimmed_mantissa[1::], 2) < int(mantissa_threshold,2):
                new_mantissas.append(remaining_mantissa + trimmed_mantissa[0] + '0'*(len(mantissa_threshold)-1))

        return new_mantissas

    def _float_specs(self, float_format='single'):
        if float_format == 'single':
            self.exponent_width = 8
            self.mantissa_width = 23
            self.exp_offset = 127
        elif float_format == 'double':
            self.exponent_width = 11
            self.mantissa_width = 52
            self.exp__offset = 1023
        else:
            raise ValueError("Invalid float format. Choose 'single' or 'double'")

    def _strip_and_pad(self, bin_array, width):

        # Join the binaries and remove the '0b' prefix
        stripped_binaries = [s.replace('0b', '') for s in bin_array]

        # Pad the binaries to width bits
        padded = [s.rjust(width, '0') for s in stripped_binaries]

        return padded

    def _float_parts(self, arr):

        signs = []
        exponents = []
        mantissas = []
        
        for num in arr:

            # Pack the float into hex data (IEEE 754 format)
            packed_hex = struct.pack('>f', num)

            # Convert the packed hex data to an integer
            integers = [c for c in packed_hex]

            # Convert the integers to binary
            binaries = [bin(i) for i in integers]

            # Strip and pad the binaries
            padded = self._strip_and_pad(binaries, 8)

            # Join the binaries
            bit_string = ''.join(padded)
            
            # Convert to bitstrings
            sign_bit = bit_string[0]
            exponent_bits = bit_string[1:1+self.exponent_width]
            mantissa_bits = bit_string[1+self.exponent_width::]
            
            signs.append(sign_bit)
            exponents.append(exponent_bits)
            mantissas.append(mantissa_bits)
            
        return signs, exponents, mantissas
    
    def __repr__(self) -> str:

        if self.exponent == '11111111':
            return str([0.0]*self.block_size)

        return str(self.float())