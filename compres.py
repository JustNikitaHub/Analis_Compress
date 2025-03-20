import heapq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
matplotlib.use('Qt5Agg')
class Node:
    def __init__(self, byte=None, freq=None, left=None, right=None):
        self.byte = byte
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [Node(byte, freq) for byte, freq in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    return heap[0]

def build_codes(node, prefix="", codebook={}):
    if node.byte is not None:
        codebook[node.byte] = prefix
    else:
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data):
    frequency = Counter(data)
    tree = build_huffman_tree(frequency)
    codebook = build_codes(tree)
    encoded_bits = [int(bit) for byte in data for bit in codebook[byte]]
    return encoded_bits, tree

def huffman_decode(encoded_bits, tree):
    decoded_data = []
    current_node = tree
    for bit in encoded_bits:
        if bit == 0:
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.byte is not None:
            decoded_data.append(current_node.byte)
            current_node = tree
    return bytes(decoded_data)

def bitlist_to_bytes(bit_list):
    remainder = len(bit_list) % 8
    if remainder > 0:
        bit_list += [0] * (8 - remainder)

    byte_array = bytearray()
    for i in range(0, len(bit_list), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bit_list[i + j]
        byte_array.append(byte)
    return bytes(byte_array)

def bytes_to_bitlist(byte_data):
    bit_list = []
    for byte in byte_data:
        for i in range(7, -1, -1):
            bit = (byte >> i) & 1
            bit_list.append(bit)
    return bit_list

def rle_encode(data):
    encoded_data = bytearray()
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
            if count == 256:
                encoded_data.append(data[i - 1])
                encoded_data.append(255)
                count = 1
        else:
            encoded_data.append(data[i - 1])
            encoded_data.append(count)
            count = 1
    encoded_data.append(data[-1])
    encoded_data.append(count)
    return bytes(encoded_data)

def rle_decode(encoded_data):
    decoded_data = bytearray()
    i = 0
    while i < len(encoded_data):
        if i + 1 >= len(encoded_data):
            break
        byte = encoded_data[i]
        count = encoded_data[i + 1]
        decoded_data.extend([byte] * count)
        i += 2
    return bytes(decoded_data)

def bwt_transform(data):
    rotations = [data[i:] + data[:i] for i in range(len(data))]
    rotations.sort()
    last_column = bytes(row[-1] for row in rotations)
    original_index = rotations.index(data)
    return last_column, original_index

def bwt_inverse_transform(last_column, original_index):
    table = [b''] * len(last_column)
    for _ in range(len(last_column)):
        table = sorted([last_column[i:i+1] + table[i] for i in range(len(last_column))])
    original_data = table[original_index]
    return original_data

def mtf_encode(data):
    alphabet = list(range(256))
    encoded_data = []
    for byte in data:
        index = alphabet.index(byte)
        encoded_data.append(index)
        alphabet.pop(index)
        alphabet.insert(0, byte)
    return encoded_data

def mtf_decode(encoded_data):
    alphabet = list(range(256))
    decoded_data = []
    for index in encoded_data:
        byte = alphabet[index]
        decoded_data.append(byte)
        alphabet.pop(index)
        alphabet.insert(0, byte)
    return bytes(decoded_data)

def lz77_encode(data, window_size, lookahead_size):
    encoded_data = bytearray()
    i = 0
    while i < len(data):
        best_offset = 0
        best_length = 0
        search_start = max(0, i - window_size)
        for j in range(search_start, i):
            current_length = 0
            while(i + current_length < len(data) and (j + current_length < i) and (data[j + current_length] == data[i + current_length]) and (current_length < lookahead_size)):
                current_length += 1

            if current_length > best_length:
                best_length = current_length
                best_offset = i - j
        best_length = min(best_length, 255)
        if best_length > 0:
            if i + best_length < len(data):
                next_char = data[i + best_length]
            else:
                next_char = 0
            encoded_data.extend(best_offset.to_bytes(2, 'big'))
            encoded_data.append(best_length)
            encoded_data.append(next_char)
            i += best_length + 1
        else:
            encoded_data.extend((0).to_bytes(2, 'big'))
            encoded_data.append(0)
            encoded_data.append(data[i])
            i += 1
    return bytes(encoded_data)

def lz77_decode(encoded_data):
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
        if next_char != 0:
            decoded_data.append(next_char)
    return bytes(decoded_data)

def lz78_encode(data):
    dictionary = {b'': 0}
    next_index = 1
    encoded_data = bytearray()
    current_string = b''

    for char in data:
        new_string = current_string + bytes([char])
        if new_string in dictionary:
            current_string = new_string
        else:
            encoded_data.extend(dictionary[current_string].to_bytes(2, 'big'))
            encoded_data.append(char)
            dictionary[new_string] = next_index
            next_index += 1
            current_string = b''
    if current_string:
        encoded_data.extend(dictionary[current_string].to_bytes(2, 'big'))
        encoded_data.append(0)
    return bytes(encoded_data)

def lz78_decode(encoded_data):
    dictionary = {0: b''}
    decoded_data = bytearray()
    next_index = 1
    i = 0
    while i < len(encoded_data):
        if i + 3 > len(encoded_data):
            break
        index = int.from_bytes(encoded_data[i:i+2], 'big')
        char = encoded_data[i+2]
        i += 3
        prefix = dictionary.get(index, b'')
        entry = prefix + bytes([char]) if char != 0 else prefix
        decoded_data.extend(entry)
        if next_index not in dictionary and char != 0:
            dictionary[next_index] = prefix + bytes([char])
            next_index += 1
    return bytes(decoded_data)

def calculate_entropy(data):
    if not data:
        return 0
    counter = Counter(data)
    probabilities = [count / len(data) for count in counter.values()]
    return -sum(p * np.log2(p) for p in probabilities)

def process_text_with_blocks(text, block_sizes):
    entropy_results = []
    for block_size in block_sizes:
        entropy_list = []
        print(f"Вычисление энтропии для {block_size}", end=" = ")
        for i in range(0, len(text), block_size):
            block = text[i:i + block_size]
            if not block:
                continue
            bwt_block, _ = bwt_transform(block)
            mtf_block = mtf_encode(bwt_block)
            entropy = calculate_entropy(mtf_block)
            entropy_list.append(entropy)
        avg_entropy = np.mean(entropy_list)
        entropy_results.append(avg_entropy)
        print(avg_entropy)
    return entropy_results

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
plt.figure(figsize=(10, 6))
plt.plot(block_sizes, entropy_results, marker='o')
plt.xlabel('Размер блока в байтах')
plt.ylabel('Средняя энтропия')
plt.title('Зависимость энтропии от размера блоков (BWT + MTF)')
plt.xscale('log')
plt.grid(True)
plt.savefig('entropy_vs_block_size.png')

search_buffer_sizes = [1024, 2048, 4096, 8192]
lookahead_buffer_sizes = [32, 64, 128, 256, 512]
data = data[0:len(data)//1000]
results = []
for search_size in search_buffer_sizes:
    for lookahead_size in lookahead_buffer_sizes:
        compressed_data = lz77_encode(data, search_size, lookahead_size)
        ratio = calculate_compression_ratio(data, compressed_data)
        results.append((search_size, lookahead_size, ratio))
        print(f"Буфер поиска: {search_size}, Буфер окна: {lookahead_size}, Коэффициент сжатия: {ratio:.4f}")
plt.figure(figsize=(12, 8))
for lookahead_size in lookahead_buffer_sizes:
    ratios = [r[2] for r in results if r[1] == lookahead_size]
    plt.plot(search_buffer_sizes, ratios, marker='o', label=f'Размер окна: {lookahead_size}')
plt.xlabel('Размер буфера')
plt.ylabel('Коэффициент сжатия')
plt.title('Зависимость коэффициента сжатия от размера буфера и скользащего окна (LZ77)')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.savefig("lz77_compression_ratio_vs_buffers")

window_size = 4096
lookahead_size = 256
#КОМПРЕСОРЫ
#1 - HA
encoded_bits, tree = huffman_encode(data)
original_bit_length = len(encoded_bits)
encoded_bytes = bitlist_to_bytes(encoded_bits)
with open('ha_encoded', 'wb') as ha:
    ha.write(encoded_bytes)
encoded_bits_again = bytes_to_bitlist(encoded_bytes)
encoded_bits_again = encoded_bits_again[:original_bit_length]
decoded_data = huffman_decode(encoded_bits_again, tree)
with open('ha_decoded.txt', 'w', encoding='utf-8') as ha:
    ha.write(decoded_data.decode())
assert data==decoded_data, "Ошибка алгоритма HA"
print(f"Работа HA: Исходные данные - {len(data)} байт, сжатые данные - {len(encoded_bytes)} байт, декомпрессия - {len(decoded_data)} байт. Сжатие - {len(encoded_bytes)/len(data)}. Точность декомпресии - {len(decoded_data)/len(data)}")

#2 - RLE
step1=rle_encode(data)
with open('rle_encoded', 'wb') as rle:
    rle.write(step1)
step2=rle_decode(step1)
with open('rle_decoded.txt', 'w', encoding='utf-8') as rle:
    rle.write(step2.decode())
assert data==step2, "Ошибка алгоритма RLE"
print(f"Работа RLE: Исходные данные - {len(data)} байт, сжатые данные - {len(step1)} байт, декомпрессия - {len(step2)} байт. Сжатие - {len(step1)/len(data)}. Точность декомпресии - {len(step2)/len(data)}")

#3 - BWT+RLE
step1, index=bwt_transform(data)
step2 = rle_encode(step1)
with open('bwt+rle_encoded', 'wb') as bwtrle:
    bwtrle.write(step2)
step3 = rle_decode(step2)
step4 = bwt_inverse_transform(step3, index)
with open('bwt+rle_decoded.txt', 'w', encoding='utf-8') as bwtrle:
    bwtrle.write(step4.decode())
assert data==step4, "Ошибка алгоритма BWT+RLE"
print(f"Работа BWT+RLE: Исходные данные - {len(data)} байт, сжатые данные - {len(step2)} байт, декомпрессия - {len(step4)} байт. Сжатие - {len(step2)/len(data)}. Точность декомпресии - {len(step4)/len(data)}")

#4 - BWT+MTF+HA
step1, index=bwt_transform(data)
step2 = mtf_encode(step1)
step3, tree = huffman_encode(step2)
lenbits = len(step3)
tobytes = bitlist_to_bytes(step3)
with open('bwt+mtf+ha_encoded', 'wb') as bwtmtfha:
    bwtmtfha.write(tobytes)
tobits = bytes_to_bitlist(tobytes)
tobits=tobits[:lenbits]
step4 = huffman_decode(tobits, tree)
step5 = mtf_decode(step4)
step6 = bwt_inverse_transform(step5,index)
with open('bwt+mtf+ha_decoded.txt', 'w', encoding='utf-8') as bwtmtfha:
    bwtmtfha.write(step6.decode())
assert data==step6, "Ошибка алгоритма BWT+MTF+HA"
print(f"Работа BWT+MTF+HA: Исходные данные - {len(data)} байт, сжатые данные - {len(tobytes)} байт, декомпрессия - {len(step6)} байт. Сжатие - {len(tobytes)/len(data)}. Точность декомпресии - {len(step6)/len(data)}")

#5 - BWT+MTF+RLE+HA
step1, index=bwt_transform(data)
step2 = mtf_encode(step1)
steprle = rle_encode(step2)
step3, tree = huffman_encode(steprle)
lenbits = len(step3)
tobytes = bitlist_to_bytes(step3)
with open('bwt+mtf+rle+ha_encoded', 'wb') as bwtmtfrleha:
    bwtmtfrleha.write(tobytes)
tobits = bytes_to_bitlist(tobytes)
tobits=tobits[:lenbits]
step4 = huffman_decode(tobits, tree)
steprle2 = rle_decode(step4)
step5 = mtf_decode(steprle2)
step6 = bwt_inverse_transform(step5,index)
with open('bwt+mtf+rle+ha_decoded.txt', 'w', encoding='utf-8') as bwtmtfrleha:
    bwtmtfrleha.write(step6.decode())
assert data==step6, "Ошибка алгоритма BWT+MTF+RLE+HA"
print(f"Работа BWT+MTF+RLE+HA: Исходные данные - {len(data)} байт, сжатые данные - {len(tobytes)} байт, декомпрессия - {len(step6)} байт. Сжатие - {len(tobytes)/len(data)}. Точность декомпресии - {len(step6)/len(data)}")

#6 - LZ77
step1 = lz77_encode(data, window_size, lookahead_size)
with open('lz77_encoded', 'wb') as lz77:
    lz77.write(step1)
step2 = lz77_decode(step1)
with open('lz77_decoded.txt', 'w', encoding='utf-8') as lz77:
    lz77.write(step2.decode())
assert data==step2, "Ошибка алгоритма LZ77"
print(f"Работа LZ77: Исходные данные - {len(data)} байт, сжатые данные - {len(step1)} байт, декомпрессия - {len(step2)} байт. Сжатие - {len(step1)/len(data)}. Точность декомпресии - {len(step2)/len(data)}")

#7 - LZ77+HA
step1 = lz77_encode(data,window_size,lookahead_size)
step2, tree = huffman_encode(step1)
lenbits = len(step2)
tobytes = bitlist_to_bytes(step2)
with open('lz77+ha_encoded', 'wb') as lz77ha:
    lz77ha.write(tobytes)
tobits = bytes_to_bitlist(tobytes)
tobits=tobits[:lenbits]
step3 = huffman_decode(tobits, tree)
step4 = lz77_decode(step3)
with open('lz77+ha_decoded.txt', 'w', encoding='utf-8') as lz77ha:
    lz77ha.write(step4.decode())
#assert data==step4, "Ошибка алгоритма LZ77+HA"
print(f"Работа LZ77+HA: Исходные данные - {len(data)} байт, сжатые данные - {len(tobytes)} байт, декомпрессия - {len(step4)} байт. Сжатие - {len(tobytes)/len(data)}. Точность декомпресии - {len(step4)/len(data)}")

#8 - LZ78
step1 = lz78_encode(data)
with open('lz78_encoded', 'wb') as lz78:
    lz78.write(step1)
step2 = lz78_decode(step1)
with open('lz78_decoded.txt', 'w', encoding='utf-8') as lz78:
    lz78.write(step2.decode())
assert data==step2, "Ошибка алгоритма LZ78"
print(f"Работа LZ78: Исходные данные - {len(data)} байт, сжатые данные - {len(step1)} байт, декомпрессия - {len(step2)} байт. Сжатие - {len(step1)/len(data)}. Точность декомпресии - {len(step2)/len(data)}")

#9 - LZ79+HA
step1 = lz78_encode(data)
step2, tree = huffman_encode(step1)
lenbits = len(step2)
tobytes = bitlist_to_bytes(step2)
with open('lz78+ha_encoded', 'wb') as lz78ha:
    lz78ha.write(tobytes)
tobits = bytes_to_bitlist(tobytes)
tobits=tobits[:lenbits]
step3 = huffman_decode(tobits, tree)
step4 = lz78_decode(step3)
with open('lz78+ha_decoded.txt', 'w', encoding='utf-8') as lz78ha:
    lz78ha.write(step4.decode())
assert data==step4, "Ошибка алгоритма LZ78+HA"
print(f"Работа LZ78+HA: Исходные данные - {len(data)} байт, сжатые данные - {len(tobytes)} байт, декомпрессия - {len(step4)} байт. Сжатие - {len(tobytes)/len(data)}. Точность декомпресии - {len(step4)/len(data)}")