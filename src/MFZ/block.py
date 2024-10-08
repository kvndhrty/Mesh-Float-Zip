import matplotlib.pyplot as plt

import constriction

import numpy as np

from MFZ.mhb import make_mhb, make_sphara_mhb, make_random_basis

from MFZ.bfp import BFP

from MFZ.fast_funcs import np_binary_array_to_string_array

class Block(object):

    def __init__(self, points : np.ndarray , cache : bool = True, epsilon : float = 1.0, basis_function = make_mhb):

        self.points = points

        self.decoder = None

        if cache:
            self.basis = basis_function(points)
            self.basis_pinv = np.linalg.pinv(self.basis)


    def __repr__(self) -> str:
        return f"Block({len(self.points)} points, {self.points.shape[-1]} dimensions)"
    

    def display(self, ax=None, **kwargs):
        #displays an image of the mesh as a scatter plot

        #set default plot values 
        for key, value in {'s': 1, 'edgecolor': 'k', 'facecolor' : 'w', 'alpha' : 0.5}.items():
            if key not in kwargs:
                kwargs[key] = value

        if ax is None:
            ax = plt.gca()

        ax.scatter(self.points[:,0], self.points[:,1], **kwargs)
        plt.show()
        
        return ax
    

    def _encode_bitstream(self, bfp_data):

        bits = bfp_data.bitstream()

        compressed_stream = []

        current_encoding = '1'

        for i in range(bits.shape[1]):

            bit_plane = bits[:,i]

            if bit_plane.sum() == 0:
                compressed_stream.append('0')
                continue

            for j in range(bits.shape[0]):

                if bit_plane[j] == 1:
                    current_encoding += '1'

                    if (j+1 > bits.shape[0]) or (bit_plane[j+1::].sum() == 0):
                        current_encoding += '0'
                        break
                    else:
                        current_encoding += '1'
                elif bit_plane[j] == 0:
                    current_encoding += '0'

            compressed_stream.append(current_encoding)
            current_encoding = '1'

        return compressed_stream, None


    def _decode_bitstream(self, compressed_stream, encoded_field_length):

        decoded_bits = np.zeros((len(self.points),encoded_field_length), dtype=np.uint8)

        for i in range(encoded_field_length):

            current_encoding = compressed_stream[i]

            if current_encoding == '0':
                continue

            most_recent_bit = 0
            bit_index = 0

            for bit in current_encoding[1::]:

                if most_recent_bit == 1:
                    if bit == '1':
                        most_recent_bit = 0
                    elif bit == '0':
                        break

                else:

                    if bit == '1':
                        decoded_bits[bit_index,i] = 1
                        most_recent_bit = 1
                    elif bit == '0':
                        most_recent_bit = 0
                    
                    bit_index += 1

        return decoded_bits


    def _constrict(self, bfp_data):

        bits = bfp_data.bitstream()

        encoder = constriction.stream.queue.RangeEncoder()

        model_params, model_tuple = self._fit_model(bits)
        
        for i in range(bits.shape[1]):

            bit_plane = bits[:,i]

            nu_1 = model_params[i]

            weighted_model = constriction.stream.model.Bernoulli(nu_1)

            encoder.encode(np.array(bit_plane, dtype=np.int32).reshape(-1), weighted_model)

        compressed = encoder.get_compressed()

        encoder.clear()

        return compressed, model_tuple
    

    def _deconstrict(self, compressed, model_tuple, encoded_field_length):

        decoder = constriction.stream.queue.RangeDecoder(compressed)

        decoded_bits = np.empty((len(self.points)+1,encoded_field_length), dtype=np.uint8)

        model_params = self._eval_model(model_tuple, encoded_field_length)

        for i in range(encoded_field_length):

                nu_1 = model_params[i]

                weighted_model = constriction.stream.model.Bernoulli(nu_1)

                bits = decoder.decode(weighted_model, len(self.points)+1)

                decoded_bits[:,i] = bits

        return decoded_bits
    


    def raw_bitstream(self, data, error_tol=1e-3):

        #Project data onto basis

        data_spectrum = self.basis.T @ data

        #Convert data to block floating point representation 

        bfp_data = BFP(data_spectrum, error_tol=error_tol)

        return bfp_data.bitstream()


    def compress(self, data, error_tol=1e-3):

        cmprssd = None

        #Project data onto basis

        data_mean = np.mean(data, axis=0)

        mean_centered_data = data - data_mean

        data_spectrum = self.basis_pinv @ mean_centered_data

        #Convert data to block floating point representation 

        bfp_data = BFP(np.hstack((data_mean, data_spectrum)), error_tol=error_tol)

        #Compress the data
        if bfp_data.exponent == '11111111':
            cmprssd = []
            model_tuple = ''
        else:
            cmprssd, model_tuple = self._constrict(bfp_data)

        return {'mantissas' : cmprssd, 'model' : model_tuple, 'exponent' : bfp_data.exponent, 'truncate_bit' : bfp_data.truncation_bit}


    def decompress(self, block_data : dict):

        # Unpack dict

        compressed = block_data['mantissas']
        model_tuple = block_data['model']
        exponent = block_data['exponent']
        truncate_bit = block_data['truncate_bit']

        if truncate_bit is None:
            truncate_bit = '0' 

        if exponent == '11111111':
            return np.zeros((self.points.shape[0],))

        #Decompress the data

        encoded_field_length = int(truncate_bit,2) + 1

        bits = self._deconstrict(compressed, model_tuple, encoded_field_length)
        
        negabin_array = np_binary_array_to_string_array(bits)

        bfp_data = BFP(array=None, error_tol=None, exponent=exponent, negabin_array=negabin_array, truncation_bit=truncate_bit)

        float_data = np.array(bfp_data.float())

        return (self.basis @ float_data[1::]) + float_data[0]


    def count_bits(self, block_data : dict):

        compressed = block_data['mantissas']
        exponent = block_data['exponent']
        truncate_bit = block_data['truncate_bit']

        if exponent == '11111111':
            return 8

        if truncate_bit is None:
            truncate_bit_len = 0
        else:
            truncate_bit_len = 5

        total_bits = 0

        for word in compressed:
            total_bits += len(bin(word).replace('0b', '').rjust(32, '0'))

        # total mantissa bits + total model bits + total exponent bits

        return total_bits + len(exponent) + truncate_bit_len + 9


    def _fit_model(self, bits):

        dist = np.sum(bits, axis=0) / bits.shape[0]

        if any(dist > 0.40):
            line_endpoint = np.min(np.argwhere(dist > 0.40))
        else:
            line_endpoint = bits.shape[1] - 1

        if line_endpoint == 0:
            avg_zeros = 0.5
        else:
            avg_zeros = np.mean(dist[0:line_endpoint], axis=0)

        discrete_avg_zeros = np.floor(avg_zeros / (1/30)) * (1/30)

        if discrete_avg_zeros == 0:
            discrete_avg_zeros = 0.01

        nu_1 = bin(int(np.floor(avg_zeros / (1/30)))).replace('0b', '').rjust(4, '0')
        nu_2 = bin(line_endpoint).replace('0b', '').rjust(5, '0')

        model_params = np.ones(bits.shape[1]) * discrete_avg_zeros

        if line_endpoint < bits.shape[1] - 1:
            model_params[line_endpoint::] = 0.5

        assert all(model_params == self._eval_model((nu_1, nu_2), bits.shape[1])), "Model was not recoverable, decoding will fail"

        assert all(model_params < 1.0) and all(model_params > 0.0), "Model parameters are not within the correct range"

        return model_params, (nu_1, nu_2)
    

    def _eval_model(self, model_tuple, encoded_field_length):

        nu_1, nu_2 = model_tuple

        avg_zeros = int(nu_1,2) * (1/30)

        if avg_zeros == 0:
            avg_zeros = 0.01

        line_endpoint = int(nu_2,2)

        model_params = np.ones(encoded_field_length) * avg_zeros

        if line_endpoint < encoded_field_length - 1:
            model_params[line_endpoint::] = 0.5

        return model_params