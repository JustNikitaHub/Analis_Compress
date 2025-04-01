import heapq
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from typing import Tuple, List, Dict


class HuffmanNode:
    def __init__(self, byte=None, freq=0, left=None, right=None):
        self.byte = byte
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_dict(data: bytes) -> Dict[int, int]:
    freq_dict = defaultdict(int)
    for byte in data:
        freq_dict[byte] += 1
    return freq_dict

def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanNode:
    heap = []
    for byte, freq in freq_dict.items():
        heapq.heappush(heap, HuffmanNode(byte=byte, freq=freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heapq.heappop(heap)

def build_codes(root: HuffmanNode, code='', code_dict=None) -> Dict[int, str]:
    if code_dict is None:
        code_dict = {}
    
    if root.byte is not None:
        code_dict[root.byte] = code
    else:
        build_codes(root.left, code + '0', code_dict)
        build_codes(root.right, code + '1', code_dict)
    
    return code_dict

def huffman_encode(data: bytes) -> Tuple[str, HuffmanNode]:
    if not data:
        return "", None
    
    freq_dict = build_frequency_dict(data)
    root = build_huffman_tree(freq_dict)
    code_dict = build_codes(root)
    
    encoded_bits = ''.join([code_dict[byte] for byte in data])
    return encoded_bits, root

def huffman_decode(encoded_bits: str, root: HuffmanNode) -> bytes:
    if not encoded_bits or not root:
        return b''
    
    decoded = bytearray()
    current_node = root
    
    for bit in encoded_bits:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.byte is not None:
            decoded.append(current_node.byte)
            current_node = root
    
    return bytes(decoded)

def bits_to_bytes(bit_string: str) -> bytes:
    padding = 8 - len(bit_string) % 8
    if padding != 8:
        bit_string += '0' * padding
    
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        byte_array.append(int(byte, 2))
    
    return bytes(byte_array), padding

def bytes_to_bits(byte_data: bytes, padding: int) -> str:
    bit_string = ''.join([bin(byte)[2:].zfill(8) for byte in byte_data])
    if padding != 8:
        bit_string = bit_string[:-padding]
    return bit_string

def serialize_tree(root: HuffmanNode) -> bytes:
    def preorder(node):
        if node.byte is not None:
            return b'1' + bytes([node.byte])
        return b'0' + preorder(node.left) + preorder(node.right)
    return preorder(root)

def deserialize_tree(data: bytes) -> HuffmanNode:
    def helper(it):
        flag = next(it)
        if flag == ord('1'):
            return HuffmanNode(byte=next(it))
        left = helper(it)
        right = helper(it)
        return HuffmanNode(left=left, right=right)
    
    return helper(iter(data))

def compress_data(data: bytes) -> bytes:
    encoded_bits, tree = huffman_encode(data)
    encoded_bytes, padding = bits_to_bytes(encoded_bits)
    tree_bytes = serialize_tree(tree)
    return bytes([padding]) + len(tree_bytes).to_bytes(4, 'big') + tree_bytes + encoded_bytes

def decompress_data(compressed: bytes) -> bytes:
    padding = compressed[0]
    tree_size = int.from_bytes(compressed[1:5], 'big')
    tree_bytes = compressed[5:5+tree_size]
    encoded_bytes = compressed[5+tree_size:]
    
    tree = deserialize_tree(tree_bytes)
    encoded_bits = bytes_to_bits(encoded_bytes, padding)
    return huffman_decode(encoded_bits, tree)

def mtf_encode(data: bytes) -> Tuple[bytes, List[int]]:
    initial_alphabet = list(range(256))
    alphabet = initial_alphabet.copy()
    encoded = bytearray()
    for byte in data:
        index = alphabet.index(byte)
        encoded.append(index)
        alphabet.pop(index)
        alphabet.insert(0, byte)
    return bytes(encoded), initial_alphabet

def mtf_decode(encoded_data: bytes, initial_alphabet: List[int]) -> bytes:
    alphabet = initial_alphabet.copy()
    decoded = bytearray()
    for index in encoded_data:
        byte = alphabet[index]
        decoded.append(byte)
        alphabet.pop(index)
        alphabet.insert(0, byte)
    return bytes(decoded)

def bwt_transform(block: bytes) -> tuple[bytes, int]:
    n = len(block)
    if n == 0:
        return b'', 0

    indices = list(range(n))
    indices.sort(key=lambda i: (block[i:] + block[:i]))
    original_index = indices.index(0)
    transformed = bytes([block[(i - 1) % n] for i in indices])
    return transformed, original_index

def inverse_bwt(transformed: bytes, index: int) -> bytes:
    n = len(transformed)
    if n == 0:
        return b''
    
    L = list(transformed)
    F = sorted(L)
    
    count = defaultdict(int)
    rank_L = []
    for c in L:
        rank_L.append(count[c])
        count[c] += 1
    
    f_positions = defaultdict(list)
    for pos, c in enumerate(F):
        f_positions[c].append(pos)
    
    result = bytearray()
    current = index
    for _ in range(n):
        c = L[current]
        result.append(c)
        current = f_positions[c][rank_L[current]]
    
    result.reverse()
    return bytes(result)

def bwt_encode(data: bytes, block_size: int) -> bytes:
    encoded = bytearray()
    for i in range(0, len(data), block_size):
        block = data[i:i + block_size]
        transformed, idx = bwt_transform(block)
        encoded += len(transformed).to_bytes(4, 'big')
        encoded += transformed
        encoded += idx.to_bytes(4, 'big')
    return bytes(encoded)

def bwt_decode(encoded_data: bytes) -> bytes:
    decoded = bytearray()
    ptr = 0
    while ptr < len(encoded_data):
        if ptr + 4 > len(encoded_data):
            break
        block_size = int.from_bytes(encoded_data[ptr:ptr+4], 'big')
        ptr += 4
        if ptr + block_size + 4 > len(encoded_data):
            break
        transformed = encoded_data[ptr:ptr+block_size]
        ptr += block_size
        idx = int.from_bytes(encoded_data[ptr:ptr+4], 'big')
        ptr += 4
        decoded_block = inverse_bwt(transformed, idx)
        decoded += decoded_block
    return bytes(decoded)

def rle_encode(data: bytes) -> bytes:
    if not data:
        return b''

    compressed = bytearray()
    i = 0
    n = len(data)
    
    while i < n:
        current = data[i]
        count = 1
        while i + count < n and count < 255 and data[i + count] == current:
            count += 1
        
        if count > 1:
            compressed.append(count)
            compressed.append(current)
            i += count
        else:
            non_repeating = bytearray()
            while i < n:
                if i + 1 < n and data[i] != data[i + 1]:
                    non_repeating.append(data[i])
                    i += 1
                    if len(non_repeating) == 255:
                        break
                else:
                    if i + 1 < n and data[i] == data[i + 1]:
                        break
                    non_repeating.append(data[i])
                    i += 1
                    break
            if len(non_repeating) > 0:
                compressed.append(0)
                compressed.append(len(non_repeating))
                compressed.extend(non_repeating)
    return bytes(compressed)

def rle_decode(compressed: bytes) -> bytes:
    if not compressed:
        return b''

    decompressed = bytearray()
    i = 0
    n = len(compressed)
    
    while i < n:
        if i + 1 > n:
            break
            
        count = compressed[i]
        
        if count == 0:
            if i + 2 > n:
                break
            length = compressed[i + 1]
            if i + 2 + length > n:
                break
            decompressed.extend(compressed[i+2:i+2+length])
            i += 2 + length
        else:
            if i + 1 >= n:
                break
            byte = compressed[i + 1]
            decompressed.extend([byte] * count)
            i += 2
    
    return bytes(decompressed)

def lz77_encode(data, window_size=4096, lookahead_size=256):
    encoded_data = bytearray()
    i = 0
    while i < len(data):
        best_offset = 0
        best_length = 0
        search_start = max(0, i - window_size)
        for j in range(search_start, i):
            current_length = 0
            while (i + current_length < len(data) and 
                   j + current_length < i and 
                   data[j + current_length] == data[i + current_length] and 
                   current_length < lookahead_size):
                current_length += 1
                
            if current_length > best_length:
                best_length = current_length
                best_offset = i - j
        
        best_length = min(best_length, 255)
        if best_length > 0:
            if i + best_length < len(data):
                next_char = data[i + best_length]
                i += best_length + 1
            else:
                next_char = 0
                i += best_length
            
            encoded_data.extend(best_offset.to_bytes(2, 'big'))
            encoded_data.append(best_length)
            encoded_data.append(next_char)
        else:
            encoded_data.extend((0).to_bytes(2, 'big'))
            encoded_data.append(0)
            encoded_data.append(data[i])
            i += 1
    return bytes(encoded_data)

def lz77_decode(encoded_data, or_data, mode="1"):
    decoded_data = bytearray()
    i = 0
    while i < len(encoded_data):
        if i + 4 > len(encoded_data):
            break
        offset = int.from_bytes(encoded_data[i:i+2], 'big')
        length = encoded_data[i+2]
        next_char = encoded_data[i+3]
        i += 4
        if length > 0:
            start = len(decoded_data) - offset
            decoded_data.extend(decoded_data[start:start+length])
        
        decoded_data.append(next_char)
    
    if mode=="f":
        decoded_data.pop()
        return bytes(decoded_data)
    if len(decoded_data) > 0 and decoded_data[-1] == 0:
        try:
            original_data = bytes(or_data)
            if original_data[-1] != 0:
                decoded_data.pop()
        except:
            pass
    
    return bytes(decoded_data)

def png_to_raw(image_path, output_path):
    image = Image.open(image_path)
    if image.mode == '1':
        arr = np.array(image, dtype=np.uint8)
        raw_data = np.packbits(arr).tobytes()
    else:
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        elif image.mode == 'P':
            image = image.convert('RGB')
        raw_data = np.array(image).tobytes()
    with open(output_path, 'wb') as f:
        f.write(raw_data)

def raw_to_png(raw_path, output_path, width, height, mode):
    with open(raw_path, 'rb') as f:
        raw_data = f.read()
    if mode == '1':
        arr = np.unpackbits(np.frombuffer(raw_data, dtype=np.uint8))
        arr = arr[:width*height].reshape((height, width)) * 255
        image = Image.fromarray(arr.astype(np.uint8), 'L').convert('1')
    elif mode == 'L':
        expected_size = width * height * (1 if mode in ['1', 'L'] else 3)
        arr = np.frombuffer(raw_data[:expected_size], dtype=np.uint8)
        arr = arr.reshape((height, width))
        image = Image.fromarray(arr, 'L')
    else:
        arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        image = Image.fromarray(arr, 'RGB')
    image.save(output_path, format='PNG')

def lz78_compress(data: bytes, max_dict_size: int = 65535) -> bytes:
    dictionary = {b'': 0}
    compressed = bytearray()
    buffer = b''
    index = 1

    for byte in data:
        current_byte = bytes([byte])
        new_buffer = buffer + current_byte
        if new_buffer in dictionary:
            buffer = new_buffer
        else:
            compressed.extend(dictionary[buffer].to_bytes(2, 'big'))
            compressed.append(byte)
            
            if index < max_dict_size:
                dictionary[new_buffer] = index
                index += 1
            
            buffer = b''
    
    if buffer:
        last_byte = buffer[-1:]
        compressed.extend(dictionary[buffer[:-1]].to_bytes(2, 'big'))
        compressed.append(last_byte[0])
    
    return bytes(compressed)

def lz78_decompress(compressed: bytes) -> bytes:
    dictionary = {0: b''}
    index = 1
    decompressed = bytearray()
    i = 0

    while i + 2 < len(compressed):
        dict_index = int.from_bytes(compressed[i:i+2], 'big')
        byte = compressed[i+2]
        i += 3

        if dict_index not in dictionary:
            raise ValueError(f"Invalid dictionary index: {dict_index}")
        
        entry = dictionary[dict_index]
        new_entry = entry + bytes([byte])
        decompressed.extend(new_entry)
        
        if index < 65536:
            dictionary[index] = new_entry
            index += 1

    return bytes(decompressed)


def ha_compressor(data, mode):
    if mode=="txt" or mode=="txt2":
        step1 = compress_data(data)
        with open(f"ha_compressed_{mode}", 'wb') as f:
            f.write(step1)
        step2 = decompress_data(step1)
        with open(f"ha_decompressed_{mode}.txt", 'w',encoding='utf-8') as f:
            f.write(step2.decode('utf-8'))
    elif mode=="exe":
        step1 = compress_data(data)
        with open("ha_compressed_exe", 'wb') as f:
            f.write(step1)
        step2 = decompress_data(step1)
        with open("ha_decompressed_exe.exe", 'wb') as f:
            f.write(step2)
    elif mode=="grey":
        step1 = compress_data(data)
        with open("ha_compressed_grey", 'wb') as f:
            f.write(step1)
        step2 = decompress_data(step1)
        with open("_.raw", 'wb') as f:
            f.write(step2)
        raw_to_png("_.raw","ha_decompressed_grey.png",800,600,'L')
    elif mode=="wb":
        step1 = compress_data(data)
        print(len(step1))
        with open("ha_compressed_wb", 'wb') as f:
            f.write(step1)
        step2 = decompress_data(step1)
        with open("_.raw", 'wb') as f:
            f.write(step2)
        raw_to_png("_.raw","ha_decompressed_wb.png",800,600,'1')
    elif mode=="rgb":
        step1 = compress_data(data)
        with open("ha_compressed_rgb", 'wb') as f:
            f.write(step1)
        step2 = decompress_data(step1)
        with open("_.raw", 'wb') as f:
            f.write(step2)
        raw_to_png("_.raw","ha_decompressed_rgb.png",800,600,'RGB')
    print(
        f"- HA - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(step1)}\n"
        f"Коэффициент сжатия: {(len(step1)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(step2)}"
    )

def rle_compressor(data: bytes, mode: str) -> None:
    compressed = rle_encode(data)
    
    if mode == "txt" or mode == "txt2":
        with open(f"rle_compressed_{mode}", 'wb') as f:
            f.write(compressed)
        decompressed = rle_decode(compressed)
        with open(f"rle_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
            
    elif mode == "exe":
        with open("rle_compressed_exe", 'wb') as f:
            f.write(compressed)
        decompressed = rle_decode(compressed)
        with open("rle_decompressed_exe.exe", 'wb') as f:
            f.write(decompressed)
            
    elif mode == "grey":
        with open("rle_compressed_grey", 'wb') as f:
            f.write(compressed)
        decompressed = rle_decode(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "rle_decompressed_grey.png", 800, 600, 'L')
        
    elif mode == "wb":
        with open("rle_compressed_wb", 'wb') as f:
            f.write(compressed)
        decompressed = rle_decode(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "rle_decompressed_wb.png", 800, 600, '1')
        
    elif mode == "rgb":
        with open("rle_compressed_rgb", 'wb') as f:
            f.write(compressed)
        decompressed = rle_decode(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "rle_decompressed_rgb.png", 800, 600, 'RGB')
    
    original_size = len(data)
    compressed_size = len(compressed)
    print(
        f"- RLE - {mode}\n"
        f"Исходный размер: {original_size}\n"
        f"Сжатый размер: {compressed_size}\n"
        f"Коэффициент сжатия: {compressed_size/original_size:.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )

def bwt_rle_compressor(data: bytes, mode: str):
    block = 4096 if mode in ("txt","txt2") else 512
    step1 = bwt_encode(data, block)
    compressed = rle_encode(step1)

    step3 = rle_decode(compressed)
    decompressed = bwt_decode(step3)
    if mode == "txt" or mode == "txt2":
        with open(f"bwt_rle_compressed_{mode}", 'wb') as f:
            f.write(compressed)
        with open(f"bwt_rle_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
            
    elif mode == "exe":
        with open("bwt_rle_compressed_exe", 'wb') as f:
            f.write(compressed)
        with open("bwt_rle_decompressed_exe.exe", 'wb') as f:
            f.write(decompressed)
            
    elif mode == "grey":
        with open("bwt_rle_compressed_grey", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_rle_decompressed_grey.png", 800, 600, 'L')
        
    elif mode == "wb":
        with open("bwt_rle_compressed_wb", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_rle_decompressed_wb.png", 800, 600, '1')
        
    elif mode == "rgb":
        with open("bwt_rle_compressed_rgb", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_rle_decompressed_rgb.png", 800, 600, 'RGB')
    print(
        f"- BWT+RLE - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(compressed)}\n"
        f"Коэффициент сжатия: {(len(compressed)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )

def bwt_mtf_ha_compressor(data, mode):
    block = 4096 if mode in ("txt","txt2") else 512
    step1 = bwt_encode(data,block)

    step2, book = mtf_encode(step1)

    compressed = compress_data(step2)



    step3 = decompress_data(compressed)

    step4 = mtf_decode(step3, book)

    decompressed = bwt_decode(step4)


    if mode == "txt" or mode=="txt2":
        with open(f"bwt_mtf_ha_compressed_{mode}", 'wb') as f:
            f.write(compressed)
        with open(f"bwt_mtf_ha_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
            
    elif mode == "exe":
        with open("bwt_mtf_ha_compressed_exe", 'wb') as f:
            f.write(compressed)
        with open("bwt_mtf_ha_decompressed_exe.exe", 'wb') as f:
            f.write(decompressed)
            
    elif mode == "grey":
        with open("bwt_mtf_ha_compressed_grey", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_mtf_ha_decompressed_grey.png", 800, 600, 'L')
        
    elif mode == "wb":
        with open("bwt_mtf_ha_compressed_wb", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_mtf_ha_decompressed_wb.png", 800, 600, '1')
        
    elif mode == "rgb":
        with open("bwt_mtf_ha_compressed_rgb", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_mtf_ha_decompressed_rgb.png", 800, 600, 'RGB')
    print(
        f"- BWT+MTF+HA - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(compressed)}\n"
        f"Коэффициент сжатия: {(len(compressed)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )

def bwt_mtf_rle_ha_compressor(data, mode):
    block = 4096 if mode in ("txt","txt2") else 512
    step1 = bwt_encode(data,block)
    step2, book = mtf_encode(step1)
    step3 = rle_encode(step2)
    compressed = compress_data(step3)

    step4 = decompress_data(compressed)
    step5 = rle_decode(step4)
    step6 = mtf_decode(step5, book)
    decompressed = bwt_decode(step6)
    if mode == "txt" or mode=="txt2":
        with open(f"bwt_mtf_rle_ha_compressed_{mode}", 'wb') as f:
            f.write(compressed)
        with open(f"bwt_mtf_rle_ha_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
            
    elif mode == "exe":
        with open("bwt_mtf_rle_ha_compressed_exe", 'wb') as f:
            f.write(compressed)
        with open("bwt_mtf_rle_ha_decompressed_exe.exe", 'wb') as f:
            f.write(decompressed)
            
    elif mode == "grey":
        with open("bwt_mtf_rle_ha_compressed_grey", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_mtf_rle_ha_decompressed_grey.png", 800, 600, 'L')
        
    elif mode == "wb":
        with open("bwt_mtf_rle_ha_compressed_wb", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_mtf_rle_ha_decompressed_wb.png", 800, 600, '1')
        
    elif mode == "rgb":
        with open("bwt_mtf_rle_ha_compressed_rgb", 'wb') as f:
            f.write(compressed)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "bwt_mtf_rle_ha_decompressed_rgb.png", 800, 600, 'RGB')
    print(
        f"- BWT+MTF+RLE+HA - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(compressed)}\n"
        f"Коэффициент сжатия: {(len(compressed)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )

def lz77_compressor(data: bytes, mode: str):
    if mode == "txt" or mode=="txt2":
        compressed = lz77_encode(data)
        with open(f"lz77_compressed_{mode}", 'wb') as f:
            f.write(compressed)
        decompressed = lz77_decode(compressed, data)
        with open(f"lz77_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
            
    elif mode == "exe":
        compressed = lz77_encode(data)
        with open("lz77_compressed_exe", 'wb') as f:
            f.write(compressed)
            
        decompressed = lz77_decode(compressed, data, "f")
        with open("lz77_decompressed_exe.exe", 'wb') as f:
            f.write(decompressed)
            
    elif mode == "grey":
        compressed = lz77_encode(data)
        with open("lz77_compressed_grey", 'wb') as f:
            f.write(compressed)
            
        decompressed = lz77_decode(compressed, data)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "lz77_decompressed_grey.png", 800, 600, 'L')
        
    elif mode == "wb":
        compressed = lz77_encode(data)
        with open("lz77_compressed_wb", 'wb') as f:
            f.write(compressed)
            
        decompressed = lz77_decode(compressed, data)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "lz77_decompressed_wb.png", 800, 600, '1')
        
    elif mode == "rgb":
        compressed = lz77_encode(data)
        with open("lz77_compressed_rgb", 'wb') as f:
            f.write(compressed)
            
        decompressed = lz77_decode(compressed, data)
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", "lz77_decompressed_rgb.png", 800, 600, 'RGB')
    
    else:
        raise ValueError(f"Неподдерживаемый режим: {mode}")

    print(
        f"- LZ77 - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(compressed)}\n"
        f"Коэффициент сжатия: {(len(compressed)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )


def lz77_ha_compressor(data: bytes, mode: str):
        lz77_compressed = lz77_encode(data)
        ha_compressed = compress_data(lz77_compressed)
        

        with open(f"lz77_ha_compressed_{mode}", 'wb') as f:
            f.write(ha_compressed)
        

        ha_decompressed = decompress_data(ha_compressed)
        if(mode=="exe"):
            final_data = lz77_decode(ha_decompressed, data,"f")
        else:
            final_data = lz77_decode(ha_decompressed, data)
        

        if mode == "txt" or mode=="txt2":
            with open(f"lz77_ha_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
                f.write(final_data.decode('utf-8'))
        elif mode == "exe":
            with open(f"lz77_ha_decompressed_{mode}.exe", 'wb') as f:
                f.write(final_data)
        elif mode == "grey":
            with open("_.raw", 'wb') as f:
                f.write(final_data)
                raw_to_png("_.raw", f"lz77_ha_decompressed_{mode}.png", 800, 600, 'L')
        elif mode == "wb":
            with open("_.raw", 'wb') as f:
                f.write(final_data)
                raw_to_png("_.raw", f"lz77_ha_decompressed_{mode}.png", 800, 600, '1')
        elif mode == "rgb":
            with open("_.raw", 'wb') as f:
                f.write(final_data)
                raw_to_png("_.raw", f"lz77_ha_decompressed_{mode}.png", 800, 600, 'RGB')
        print(
        f"- LZ77+HA - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(ha_compressed)}\n"
        f"Коэффициент сжатия: {(len(ha_compressed)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(final_data)}")

def lz78_compressor(data: bytes, mode: str):
    compressed = lz78_compress(data)
    compressed_filename = f"lz78_compressed_{mode}"
    with open(compressed_filename, 'wb') as f:
        f.write(compressed)
    
    decompressed = lz78_decompress(compressed)
    
    if mode == "txt" or mode=="txt2":
        with open(f"lz78_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
    elif mode == "exe":
        with open(f"lz78_decompressed_{mode}.exe", 'wb') as f:
            f.write(decompressed)
    elif mode == "grey":
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", f"lz78_decompressed_{mode}.png", 800, 600, 'L')
    elif mode == "wb":
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", f"lz78_decompressed_{mode}.png", 800, 600, '1')
    elif mode == "rgb":
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", f"lz78_decompressed_{mode}.png", 800, 600, 'RGB')
    
    print(
        f"- LZ78 - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(compressed)}\n"
        f"Коэффициент сжатия: {(len(compressed)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )

def lz78_ha_compressor(data, mode):
    compressed = lz78_compress(data)
    ha = compress_data(compressed)

    compressed_filename = f"lz78_ha_compressed_{mode}"
    with open(compressed_filename, 'wb') as f:
        f.write(ha)
    
    ha_out = decompress_data(ha)
    decompressed = lz78_decompress(ha_out)
    
    if mode == "txt" or mode=="txt2":
        with open(f"lz78_ha_decompressed_{mode}.txt", 'w', encoding='utf-8') as f:
            f.write(decompressed.decode('utf-8'))
    elif mode == "exe":
        with open(f"lz78_ha_decompressed_{mode}.exe", 'wb') as f:
            f.write(decompressed)
    elif mode == "grey":
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", f"lz78_ha_decompressed_{mode}.png", 800, 600, 'L')
    elif mode == "wb":
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", f"lz78_ha_decompressed_{mode}.png", 800, 600, '1')
    elif mode == "rgb":
        with open("_.raw", 'wb') as f:
            f.write(decompressed)
        raw_to_png("_.raw", f"lz78_ha_decompressed_{mode}.png", 800, 600, 'RGB')
    
    print(
        f"- LZ78+HA - {mode}\n"
        f"Исходный размер: {len(data)}\n"
        f"Сжатый размер: {len(ha)}\n"
        f"Коэффициент сжатия: {(len(ha)/len(data)):.3f}\n"
        f"Размер после распаковки: {len(decompressed)}"
    )

def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    
    if isinstance(data, list):
        data = bytes(data)
    elif not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Неподдерживаемый тип данных: {type(data)}")
    
    counts = Counter(data)
    total = len(data)
    probs = [c/total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def process_text_with_blocks(text: bytes, block_sizes: List[int]) -> List[float]:
    entropy_results = []
    
    for block_size in block_sizes:
        entropy_list = []
        print(f"Вычисление энтропии для {block_size}", end=" = ")
        for i in range(0, len(text), block_size):
            block = text[i:i + block_size]
            if not block:
                continue
        
            bwt_block = bwt_encode(block,4096)
            mtf_block, _ = mtf_encode(bwt_block)
            entropy = calculate_entropy(mtf_block)
            entropy_list.append(entropy)
        
        avg_entropy = np.mean(entropy_list) if entropy_list else 0
        entropy_results.append(avg_entropy)
        print(avg_entropy)
    
    return entropy_results

def plot_results(block_sizes, entropy_results):
    plt.figure(figsize=(10, 6))
    plt.plot(block_sizes, entropy_results, marker='o')
    plt.xlabel('Размер блока в байтах')
    plt.ylabel('Средняя энтропия')
    plt.title('Зависимость энтропии от размера блоков (BWT + MTF)')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig('entropy_vs_block_size.png')

def calculate_compression_ratio(original_data, compressed_data):
    original_size = len(original_data)
    compressed_size = len(compressed_data)
    if compressed_size == 0:
        return 0
    return compressed_size/original_size

with open('enwik7.txt', 'rb') as f:
    data = f.read()
block_sizes = [100, 500, 1000, 5000, 10000, 25000, 50000]
entropy_results = process_text_with_blocks(data, block_sizes)
plot_results(block_sizes, entropy_results)

data=data[:10000]
search_buffer_sizes = [1024, 2048, 4096, 8192]
lookahead_buffer_sizes = [64, 128, 256, 512, 1024]
results = []
for search_size in search_buffer_sizes:
    for lookahead_size in lookahead_buffer_sizes:
        step1 = lz77_encode(data, search_size, lookahead_size)
        ratio = calculate_compression_ratio(data, step1)
        results.append((search_size, lookahead_size, ratio))
        print(f"Буфер поиска: {search_size}, Буфер окна: {lookahead_size}, Коэффициент сжатия: {ratio:.4f}")
plt.figure(figsize=(12, 8))
for lookahead_size in lookahead_buffer_sizes:
    ratios = [r[2] for r in results if r[1] == lookahead_size]
    plt.plot(search_buffer_sizes, ratios, marker='o', label=f'Размер окна: {lookahead_size}')
plt.xlabel('Размер буфера')
plt.ylabel('Коэффициент сжатия')
plt.title('Зависимость коэффициента сжатия от размера буфера и скользащего окна (LZ77)')
plt.grid(True)
plt.legend()
plt.savefig("lz77_compression_ratio_vs_buffers")


with open("lyrics.txt",'rb') as f:
    lyrics_data = f.read()
with open("enwik7.txt", "rb") as f:
    text_data = f.read()
text_data=text_data[:666624]
with open("ULTRAKILL.exe", "rb") as f:
    bin_data = f.read()
with open("color.raw", "rb") as f:
    color_data = f.read()
with open("output.raw", "rb") as f:
    grey_data = f.read()
with open("bw.raw", "rb") as f:
    wb_data = f.read()

datas=[text_data, lyrics_data, bin_data, color_data, grey_data, wb_data]
modes=["txt","txt2","exe","rgb","grey","wb"]
for i in range(len(datas)):
    ha_compressor(datas[i],modes[i])
    rle_compressor(datas[i],modes[i])
    bwt_rle_compressor(datas[i],modes[i])
    bwt_mtf_ha_compressor(datas[i],modes[i])
    bwt_mtf_rle_ha_compressor(datas[i],modes[i])
    lz77_compressor(datas[i],modes[i])
    lz77_ha_compressor(datas[i],modes[i])
    lz78_compressor(datas[i],modes[i])
    lz78_ha_compressor(datas[i],modes[i])