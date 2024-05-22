import struct

import numpy as np


class BFP(object):

    def __init__(self, array, error_tol=None, float_format='single', signs=None, exponent=None, mantissas=None, truncation_bit=None) -> None:

        self._float_specs(float_format)

        self.exponent = exponent
        self.signs = signs
        self.mantissas = mantissas
        self.truncation_bit = truncation_bit

        self.error_tol = error_tol

        if signs is None or exponent is None or mantissas is None:
            self._encode(array)

    def bitstream(self) -> np.ndarray:

        bitstream = np.array([], dtype=np.uint8)

        table = bytearray.maketrans(b'01', b'\x00\x01')

        if self.mantissas == []:
            mantissas = ['0'*23]*len(self.signs)
        else:
            mantissas = self.mantissas

        for i in range(len(self.signs)):

            bit_bunch = self.signs[i] + mantissas[i]

            bit_bunch = bytearray(bit_bunch, "ascii").translate(table)

            bitstream = np.append(bitstream, bit_bunch)

        return bitstream
    
    def bytestream(self) -> np.ndarray:

        bytestream = np.array([], dtype=np.uint8)

        if self.mantissas == []:
            mantissas = ['0'*23]*len(self.signs)
        else:
            mantissas = self.mantissas

        for i in range(len(self.signs)):

            bit_bunch = self.signs[i].replace('0b', '') + mantissas[i].replace('0b', '').rjust(23, '0')

            bytestream = np.append(bytestream, [int(bit_bunch[0:8],2), int(bit_bunch[8:16],2), int(bit_bunch[16::],2)])

        return bytestream.astype(np.uint8)

    def float(self) -> None:

        values = []

        for sign, mantissa in zip(self.signs, self.mantissas):

            if self.truncation_bit is not None:
                full_mantissa = mantissa.ljust(self.mantissa_width, '0')
            else:
                full_mantissa = mantissa    

            values.append(self._decode(sign, full_mantissa))

        return values

    def _encode(self, array) -> None:

        signs, exponents, mantissas = self._float_parts(array)

        common_exp = max([int(exp, 2) for exp in exponents])

        self.exponent = bin(common_exp)

        self.signs = ['0' if sign == '1' else '1' for sign in signs]

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
            block_mantissas, self.truncation_bit = self.bit_plane_truncate(block_mantissas)
        else:
            self.truncation_bit = bin(23).replace('0b', '').rjust(8, '0')

        self.mantissas = block_mantissas
        
        if self.mantissas == []:
            # This exponent indicates that the block is filled with ONLY zeros
            self.exponent = '11111111'

        return 

    def _decode(self, sign, mantissa) -> float:

        # Convert the sign bit to a value
        sign_value = pow(-1, 1 - int(sign,2))

        # Convert the exponent to a value
        exponent_value = int(self.exponent, 2)

        # Convert the mantissa to a value
        mantissa_value = int(mantissa, 2) / 2**22

        # Calculate the value
        value = sign_value * mantissa_value * 2**(exponent_value - self.exp_offset)

        return value

    def bit_plane_truncate(self, mantissa_array):

        first_one_bit = np.array([mantissa[::-1].find('1') for mantissa in mantissa_array])

        if np.all(first_one_bit == -1):
            return [], bin(0).replace('0b', '').rjust(8, '0')

        truncation_bit = np.min(first_one_bit[first_one_bit != -1])

        new_mantissas = []

        for k in range(len(self.signs)):
            new_mantissas.append(mantissa_array[k][0:-truncation_bit])

        return new_mantissas, bin(self.mantissa_width - truncation_bit).replace('0b', '').rjust(8, '0')

    def nega_encode(self, signs, mantissa_array):

        for i in range(len(self.signs)):
            pass

        return 


    def int_to_negabinary(self, i: int) -> str:
        """Decimal to negabinary."""
        if i == 0:
            digits = ["0"]
        else:
            digits = []
            while i != 0:
                i, remainder = divmod(i, -2)
                if remainder < 0:
                    i, remainder = i + 1, remainder + 2
                digits.append(str(remainder))
        return "".join(digits[::-1]).rjust(24,"0")
    
    def negabinary_to_int(self, s: str) -> int:
        """Negabinary to decimal."""
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
            return str([0.0]*len(self.signs))

        return str(self.float())