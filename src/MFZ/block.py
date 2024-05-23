import matplotlib.pyplot as plt

import constriction

import numpy as np

from MFZ.mhb import make_mhb

from MFZ.bfp import BFP


class Block(object):

    def __init__(self, points, cache=True, epsilon=1.0):

        self.points = points

        self.decoder = None

        if cache:
            self.basis = make_mhb(points, epsilon=epsilon)

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
    
    def _constrict(self, bfp_data):

        bits = bfp_data.bitstream()

        nu_bits = bin(int(np.ceil(np.sum(bits) / len(bits) / (1/15)))).replace('0b', '')

        nu_1 = np.ceil(np.sum(bits) / len(bits) / (1/15)) * (1/15)

        if nu_1 == 0:
            nu_1 += 3e-2

        if nu_1 == 1:
            nu_1 -= 3e-2

        weighted_model = constriction.stream.model.Bernoulli(nu_1)

        encoder = constriction.stream.queue.RangeEncoder()

        encoder.encode(np.array(bits, dtype=np.int32).reshape(-1), weighted_model)

        compressed = encoder.get_compressed()

        encoder.clear()

        return compressed, nu_bits
    
    def byte_constrict(self, bfp_data):

        bytes = bfp_data.bytestream()

        weights = np.zeros(256, dtype=np.float32)

        for num in bytes:
            weights[num] += 1


        #weights = np.array([np.sum([(1-int(i)) for i in bin(i).replace('0b', '')]) for i in range(256)], dtype=np.float32)

        weights /= np.sum(weights)

        weighted_model = constriction.stream.model.Categorical(weights)

        encoder = constriction.stream.queue.RangeEncoder()

        encoder.encode(np.array(bytes, dtype=np.int32).reshape(-1), weighted_model)

        compressed = encoder.get_compressed()

        encoder.clear()

        return compressed, []
    
    def byte_compress(self, data, error_tol=1e-3):

        cmprssd = None

        #Project data onto basis

        data_spectrum = self.basis.T @ data

        #Convert data to block floating point representation 

        bfp_data = BFP(data_spectrum, error_tol=error_tol)

        #Compress the data
        if bfp_data.exponent == '11111111':
            cmprssd = []
            nu_bits = ''
        else:
            cmprssd, nu_bits = self.byte_constrict(bfp_data)

        return {'mantissas' : cmprssd, 'model' : nu_bits, 'exponent' : bfp_data.exponent}
    
    def compress(self, data, error_tol=1e-3):

        cmprssd = None

        #Project data onto basis

        data_spectrum = self.basis.T @ data

        #Convert data to block floating point representation 

        bfp_data = BFP(data_spectrum, error_tol=error_tol)

        #Compress the data
        if bfp_data.exponent == '11111111':
            cmprssd = []
            nu_bits = ''
        else:
            cmprssd, nu_bits = self._constrict(bfp_data)

        return {'mantissas' : cmprssd, 'model' : nu_bits, 'exponent' : bfp_data.exponent, 'truncate_bit' : bfp_data.truncation_bit}

    def decompress(self, block_data : dict):

        # Unpack dict

        compressed = block_data['mantissas']
        nu_bits = block_data['model']
        exponent = block_data['exponent']
        truncate_bit = block_data['truncate_bit']

        if truncate_bit is None:
            truncate_bit = '0' 

        if exponent == '11111111':
            return np.zeros((self.points.shape[0],))

        #Decompress the data

        decoder = constriction.stream.queue.RangeDecoder(compressed)

        nu_1 = int(nu_bits,2) * (1/15)
        
        if nu_1 == 0:
            nu_1 += 3e-2

        if nu_1 == 1:
            nu_1 -= 3e-2

        weighted_model = constriction.stream.model.Bernoulli(nu_1)

        encoded_field_length = int(truncate_bit,2) + 1

        try:
            bits = decoder.decode(weighted_model,len(self.points)*(encoded_field_length))
        except:
            raise ValueError("Decompression failed")
        
        negabin_array = []

        for i in range(len(self.points)):

            current_bits = bits[i*(encoded_field_length) : (i+1)*(encoded_field_length) + 1]

            trunc_nb = ''.join([str(i) for i in current_bits[0::]])

            negabin_array.append(trunc_nb)

        bfp_data = BFP(array=None, error_tol=None, exponent=exponent, negabin_array=negabin_array, truncation_bit=truncate_bit)

        return np.linalg.inv(self.basis.T) @ np.array(bfp_data.float())

    def count_bits(self, block_data : dict):

        compressed = block_data['mantissas']
        nu_bits = block_data['model']
        exponent = block_data['exponent']
        truncate_bit = block_data['truncate_bit']

        if truncate_bit is None:
            truncate_bit_len = 0
        else:
            truncate_bit_len = len(truncate_bit)/2

        total_bits = 0

        for word in compressed:
            total_bits += len(bin(word).replace('0b', ''))

        # total mantissa bits + total model bits + total exponent bits

        return total_bits + len(exponent)+ len(nu_bits) + truncate_bit_len
