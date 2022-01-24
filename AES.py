import numpy as np
import copy

s_box = {'00': '63', '01': '7C', '02': '77', '03': '7B', '04': 'F2', '05': '6B', '06': '6F', '07': 'C5', '08': '30',
         '09': '01', '0A': '67', '0B': '2B', '0C': 'FE', '0D': 'D7', '0E': 'AB', '0F': '76', '10': 'CA', '11': '82',
         '12': 'C9', '13': '7D', '14': 'FA', '15': '59', '16': '47', '17': 'F0', '18': 'AD', '19': 'D4', '1A': 'A2',
         '1B': 'AF', '1C': '9C', '1D': 'A4', '1E': '72', '1F': 'C0', '20': 'B7', '21': 'FD', '22': '93', '23': '26',
         '24': '36', '25': '3F', '26': 'F7', '27': 'CC', '28': '34', '29': 'A5', '2A': 'E5', '2B': 'F1', '2C': '71',
         '2D': 'D8', '2E': '31', '2F': '15', '30': '04', '31': 'C7', '32': '23', '33': 'C3', '34': '18', '35': '96',
         '36': '05', '37': '9A', '38': '07', '39': '12', '3A': '80', '3B': 'E2', '3C': 'EB', '3D': '27', '3E': 'B2',
         '3F': '75', '40': '09', '41': '83', '42': '2C', '43': '1A', '44': '1B', '45': '6E', '46': '5A', '47': 'A0',
         '48': '52', '49': '3B', '4A': 'D6', '4B': 'B3', '4C': '29', '4D': 'E3', '4E': '2F', '4F': '84', '50': '53',
         '51': 'D1', '52': '00', '53': 'ED', '54': '20', '55': 'FC', '56': 'B1', '57': '5B', '58': '6A', '59': 'CB',
         '5A': 'BE', '5B': '39', '5C': '4A', '5D': '4C', '5E': '58', '5F': 'CF', '60': 'D0', '61': 'EF', '62': 'AA',
         '63': 'FB', '64': '43', '65': '4D', '66': '33', '67': '85', '68': '45', '69': 'F9', '6A': '02', '6B': '7F',
         '6C': '50', '6D': '3C', '6E': '9F', '6F': 'A8', '70': '51', '71': 'A3', '72': '40', '73': '8F', '74': '92',
         '75': '9D', '76': '38', '77': 'F5', '78': 'BC', '79': 'B6', '7A': 'DA', '7B': '21', '7C': '10', '7D': 'FF',
         '7E': 'F3', '7F': 'D2', '80': 'CD', '81': '0C', '82': '13', '83': 'EC', '84': '5F', '85': '97', '86': '44',
         '87': '17', '88': 'C4', '89': 'A7', '8A': '7E', '8B': '3D', '8C': '64', '8D': '5D', '8E': '19', '8F': '73',
         '90': '60', '91': '81', '92': '4F', '93': 'DC', '94': '22', '95': '2A', '96': '90', '97': '88', '98': '46',
         '99': 'EE', '9A': 'B8', '9B': '14', '9C': 'DE', '9D': '5E', '9E': '0B', '9F': 'DB', 'A0': 'E0', 'A1': '32',
         'A2': '3A', 'A3': '0A', 'A4': '49', 'A5': '06', 'A6': '24', 'A7': '5C', 'A8': 'C2', 'A9': 'D3', 'AA': 'AC',
         'AB': '62', 'AC': '91', 'AD': '95', 'AE': 'E4', 'AF': '79', 'B0': 'E7', 'B1': 'C8', 'B2': '37', 'B3': '6D',
         'B4': '8D', 'B5': 'D5', 'B6': '4E', 'B7': 'A9', 'B8': '6C', 'B9': '56', 'BA': 'F4', 'BB': 'EA', 'BC': '65',
         'BD': '7A', 'BE': 'AE', 'BF': '08', 'C0': 'BA', 'C1': '78', 'C2': '25', 'C3': '2E', 'C4': '1C', 'C5': 'A6',
         'C6': 'B4', 'C7': 'C6', 'C8': 'E8', 'C9': 'DD', 'CA': '74', 'CB': '1F', 'CC': '4B', 'CD': 'BD', 'CE': '8B',
         'CF': '8A', 'D0': '70', 'D1': '3E', 'D2': 'B5', 'D3': '66', 'D4': '48', 'D5': '03', 'D6': 'F6', 'D7': '0E',
         'D8': '61', 'D9': '35', 'DA': '57', 'DB': 'B9', 'DC': '86', 'DD': 'C1', 'DE': '1D', 'DF': '9E', 'E0': 'E1',
         'E1': 'F8', 'E2': '98', 'E3': '11', 'E4': '69', 'E5': 'D9', 'E6': '8E', 'E7': '94', 'E8': '9B', 'E9': '1E',
         'EA': '87', 'EB': 'E9', 'EC': 'CE', 'ED': '55', 'EE': '28', 'EF': 'DF', 'F0': '8C', 'F1': 'A1', 'F2': '89',
         'F3': '0D', 'F4': 'BF', 'F5': 'E6', 'F6': '42', 'F7': '68', 'F8': '41', 'F9': '99', 'FA': '2D', 'FB': '0F',
         'FC': 'B0', 'FD': '54', 'FE': 'BB', 'FF': '16'}

inverse_s_box = {'00': '52', '01': '09', '02': '6A', '03': 'D5', '04': '30', '05': '36', '06': 'A5', '07': '38',
                 '08': 'BF',
                 '09': '40', '0A': 'A3', '0B': '9E', '0C': '81', '0D': 'F3', '0E': 'D7', '0F': 'FB', '10': '7C',
                 '11': 'E3',
                 '12': '39', '13': '82', '14': '9B', '15': '2F', '16': 'FF', '17': '87', '18': '34', '19': '8E',
                 '1A': '43',
                 '1B': '44', '1C': 'C4', '1D': 'DE', '1E': 'E9', '1F': 'CB', '20': '54', '21': '7B', '22': '94',
                 '23': '32',
                 '24': 'A6', '25': 'C2', '26': '23', '27': '3D', '28': 'EE', '29': '4C', '2A': '95', '2B': '0B',
                 '2C': '42',
                 '2D': 'FA', '2E': 'C3', '2F': '4E', '30': '08', '31': '2E', '32': 'A1', '33': '66', '34': '28',
                 '35': 'D9',
                 '36': '24', '37': 'B2', '38': '76', '39': '5B', '3A': 'A2', '3B': '49', '3C': '6D', '3D': '8B',
                 '3E': 'D1',
                 '3F': '25', '40': '72', '41': 'F8', '42': 'F6', '43': '64', '44': '86', '45': '68', '46': '98',
                 '47': '16',
                 '48': 'D4', '49': 'A4', '4A': '5C', '4B': 'CC', '4C': '5D', '4D': '65', '4E': 'B6', '4F': '92',
                 '50': '6C',
                 '51': '70', '52': '48', '53': '50', '54': 'FD', '55': 'ED', '56': 'B9', '57': 'DA', '58': '5E',
                 '59': '15',
                 '5A': '46', '5B': '57', '5C': 'A7', '5D': '8D', '5E': '9D', '5F': '84', '60': '90', '61': 'D8',
                 '62': 'AB',
                 '63': '00', '64': '8C', '65': 'BC', '66': 'D3', '67': '0A', '68': 'F7', '69': 'E4', '6A': '58',
                 '6B': '05',
                 '6C': 'B8', '6D': 'B3', '6E': '45', '6F': '06', '70': 'D0', '71': '2C', '72': '1E', '73': '8F',
                 '74': 'CA',
                 '75': '3F', '76': '0F', '77': '02', '78': 'C1', '79': 'AF', '7A': 'BD', '7B': '03', '7C': '01',
                 '7D': '13',
                 '7E': '8A', '7F': '6B', '80': '3A', '81': '91', '82': '11', '83': '41', '84': '4F', '85': '67',
                 '86': 'DC',
                 '87': 'EA', '88': '97', '89': 'F2', '8A': 'CF', '8B': 'CE', '8C': 'F0', '8D': 'B4', '8E': 'E6',
                 '8F': '73',
                 '90': '96', '91': 'AC', '92': '74', '93': '22', '94': 'E7', '95': 'AD', '96': '35', '97': '85',
                 '98': 'E2',
                 '99': 'F9', '9A': '37', '9B': 'E8', '9C': '1C', '9D': '75', '9E': 'DF', '9F': '6E', 'A0': '47',
                 'A1': 'F1',
                 'A2': '1A', 'A3': '71', 'A4': '1D', 'A5': '29', 'A6': 'C5', 'A7': '89', 'A8': '6F', 'A9': 'B7',
                 'AA': '62',
                 'AB': '0E', 'AC': 'AA', 'AD': '18', 'AE': 'BE', 'AF': '1B', 'B0': 'FC', 'B1': '56', 'B2': '3E',
                 'B3': '4B',
                 'B4': 'C6', 'B5': 'D2', 'B6': '79', 'B7': '20', 'B8': '9A', 'B9': 'DB', 'BA': 'C0', 'BB': 'FE',
                 'BC': '78',
                 'BD': 'CD', 'BE': '5A', 'BF': 'F4', 'C0': '1F', 'C1': 'DD', 'C2': 'A8', 'C3': '33', 'C4': '88',
                 'C5': '07',
                 'C6': 'C7', 'C7': '31', 'C8': 'B1', 'C9': '12', 'CA': '10', 'CB': '59', 'CC': '27', 'CD': '80',
                 'CE': 'EC',
                 'CF': '5F', 'D0': '60', 'D1': '51', 'D2': '7F', 'D3': 'A9', 'D4': '19', 'D5': 'B5', 'D6': '4A',
                 'D7': '0D',
                 'D8': '2D', 'D9': 'E5', 'DA': '7A', 'DB': '9F', 'DC': '93', 'DD': 'C9', 'DE': '9C', 'DF': 'EF',
                 'E0': 'A0',
                 'E1': 'E0', 'E2': '3B', 'E3': '4D', 'E4': 'AE', 'E5': '2A', 'E6': 'F5', 'E7': 'B0', 'E8': 'C8',
                 'E9': 'EB',
                 'EA': 'BB', 'EB': '3C', 'EC': '83', 'ED': '53', 'EE': '99', 'EF': '61', 'F0': '17', 'F1': '2B',
                 'F2': '04',
                 'F3': '7E', 'F4': 'BA', 'F5': '77', 'F6': 'D6', 'F7': '26', 'F8': 'E1', 'F9': '69', 'FA': '14',
                 'FB': '63',
                 'FC': '55', 'FD': '21', 'FE': '0C', 'FF': '7D'}

RConstant = (
    '01000000', '02000000', '04000000', '08000000', '10000000', '20000000', '40000000', '80000000', '1B000000',
    '36000000',)


def keyExpansion(key):
    k = [key]
    w = [key[:8], key[8:16], key[16:24], key[24:]]
    for i in range(1, 11):
        t = xor(s_box[w[4 * (i - 1) + 3][2:4]] + s_box[w[4 * (i - 1) + 3][4:6]] + s_box[w[4 * (i - 1) + 3][6:]] + s_box[
            w[4 * (i - 1) + 3][:2]], RConstant[i - 1])
        w.append(xor(t, w[4 * (i - 1)]))
        w.append(xor(w[4 * i], w[4 * (i - 1) + 1]))
        w.append(xor(w[4 * i + 1], w[4 * (i - 1) + 2]))
        w.append(xor(w[4 * i + 2], w[4 * (i - 1) + 3]))
        k.append(w[4 * i] + w[4 * i + 1] + w[4 * i + 2] + w[4 * i + 3])
    return k


constant_matrix = '02010103030201010103020101010302'

inverse_constant_matrix = '0E090D0B0B0E090D0D0B0E09090D0B0E'


def sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]
    return s


def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inverse_s_box[s[i][j]]
    return s


def shift_rows(s):
    s[1][0], s[1][1], s[1][2], s[1][3] = s[1][1], s[1][2], s[1][3], s[1][0]
    s[2][0], s[2][1], s[2][2], s[2][3] = s[2][2], s[2][3], s[2][0], s[2][1]
    s[3][0], s[3][1], s[3][2], s[3][3] = s[3][3], s[3][0], s[3][1], s[3][2]
    return s


def inv_shift_rows(s):
    s[1][0], s[1][1], s[1][2], s[1][3] = s[1][3], s[1][0], s[1][1], s[1][2]
    s[2][0], s[2][1], s[2][2], s[2][3] = s[2][2], s[2][3], s[2][0], s[2][1]
    s[3][0], s[3][1], s[3][2], s[3][3] = s[3][1], s[3][2], s[3][3], s[3][0]
    return s


def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] = xor(s[i][j], k[i][j])
    return s


def textToHex(Text):
    Hex = ""
    for char in Text:
        Hex += hex(ord(char)).lstrip("0x").rstrip("L")
    return Hex


def hexToText(Hex):
    bytes_object = bytes.fromhex(Hex)
    Text = bytes_object.decode("ASCII")
    return Text


def hexToBin(Hex):
    Binary = bin(int(Hex, 16))[2:].zfill(len(Hex) * 4)
    return Binary


def binToHex(Binary):
    Hex = format(int(Binary, 2), 'x').upper()
    while (len(Hex) != len(Binary) / 4):
        Hex = '0' + Hex
    return Hex


def xor(a, b):
    r = ""
    a = hexToBin(a)
    b = hexToBin(b)
    for i in range(len(a)):
        if a[i] == b[i]:
            r += '0'
        else:
            r += '1'
    r = binToHex(r)
    return r


def makeBlock(x):
    y = [[], [], [], []]
    c = 0
    for i in range(4):
        for j in range(4):
            y[j].append(x[c:c + 2])
            c += 2
    return y


def unBlocker(y):
    x = ''
    for i in range(4):
        for j in range(4):
            x += y[j][i]
    return x


def HexToSymbolic(hex_number):
    number = hexToBin(hex_number)
    binary_number = number[::-1]
    # print(number)
    one_index = list()
    for item in range(0, len(binary_number)):
        if binary_number[item] == '1':
            one_index.append(item)
    return one_index


def SymbolicToHex(symbol):
    number = [0] * 8
    for item in symbol:
        number[item] = 1
    a = map(str, number[::-1])
    binary = ''.join(a)

    return binToHex(binary)


def SymbolicToHex15(symbol):
    number = [0] * 15
    for item in symbol:
        number[item] = 1
    a = map(str, number[::-1])
    binary = ''.join(a)

    return binToHex(binary), binary


def MixedCol(s):
    state = [[], [], [], []]
    c = makeBlock(constant_matrix)
    for i in range(4):
        for j in range(4):
            temp = '00'
            for k in range(4):
                mul = GaloisMultiplication(c[j][k], s[k][i])
                temp = xor(temp, mul)
            state[j].append(temp)
    s = state
    return s


def inv_MixedCol(s):
    state = [[], [], [], []]
    c = makeBlock(inverse_constant_matrix)
    for i in range(4):
        for j in range(4):
            temp = '00'
            for k in range(4):
                if c[j][k] == '09':
                    mul = GaloisMultiplication('02', s[k][i])
                    mul = GaloisMultiplication('02', mul)
                    mul = GaloisMultiplication('02', mul)
                    mul = xor(mul, s[k][i])
                elif c[j][k] == '0B':
                    mul = GaloisMultiplication('02', s[k][i])
                    mul = GaloisMultiplication('02', mul)
                    mul = GaloisMultiplication('02', mul)
                    mul = xor(mul, s[k][i])
                    mul = xor(mul, GaloisMultiplication('02', s[k][i]))
                elif c[j][k] == '0D':
                    mul = GaloisMultiplication('02', s[k][i])
                    mul = GaloisMultiplication('02', mul)
                    mul = GaloisMultiplication('02', mul)
                    mul = xor(mul, s[k][i])
                    t = GaloisMultiplication('02', s[k][i])
                    mul = xor(mul, GaloisMultiplication('02', t))
                elif c[j][k] == '0E':
                    mul = GaloisMultiplication('02', s[k][i])
                    mul = GaloisMultiplication('02', mul)
                    mul = GaloisMultiplication('02', mul)
                    t = GaloisMultiplication('02', s[k][i])
                    mul = xor(mul, GaloisMultiplication('02', t))
                    mul = xor(mul, GaloisMultiplication('02', s[k][i]))
                temp = xor(temp, mul)
            state[j].append(temp)
    s = state
    return s


# Hamming distance
def hammingDist(str1, str2):
    index = 0
    count = 0

    while index < len(str1):
        if str1[index] != str2[index]:
            count += 1
        index += 1
    return count


def GaloisMultiplication(a, b):
    symbol_a = HexToSymbolic(a)
    symbol_b = HexToSymbolic(b)

    multiply_symbols = list()
    for i in symbol_a:
        for j in symbol_b:
            multiply_symbols.append(i + j)
    multiply_symbols.sort()
    temp = copy.deepcopy(multiply_symbols)
    for item in temp:
        counter = multiply_symbols.count(item)
        # remove if counter is not odd
        if counter % 2 == 0:
            while counter != 0:
                multiply_symbols.remove(item)
                counter -= 1

    # print(multiply_symbols)

    ext1 = [0] * 15

    for i in multiply_symbols:
        ext1[i] = 1

    ext2 = [1, 0, 0, 0, 1, 1, 0, 1, 1]
    x = np.array(ext1[::-1])
    y = np.array(ext2)
    quotient, remainder = np.polydiv(x, y)

    final = list()
    remainder = remainder[::-1]
    for i in range(len(remainder)):
        if remainder[i] != 0:
            final.append(i)
    # print(final)
    return SymbolicToHex(final)


def encrypt(p, k):
    k = keyExpansion(k.upper())
    p = p.upper()
    s = makeBlock(p)
    key = makeBlock(k[0])
    s = add_round_key(s, key)
    for i in range(1, 10):
        key = makeBlock(k[i])
        s = sub_bytes(s)
        s = shift_rows(s)
        s = MixedCol(s)
        s = add_round_key(s, key)
    key = makeBlock(k[10])
    s = sub_bytes(s)
    s = shift_rows(s)
    s = add_round_key(s, key)
    cypher = unBlocker(s)
    return cypher


def decrypt(c, k):
    k = keyExpansion(k.upper())
    c = c.upper()
    s = makeBlock(c)
    key = makeBlock(k[10])
    s = add_round_key(s, key)
    for i in range(1, 10):
        key = makeBlock(k[10 - i])
        s = inv_shift_rows(s)
        s = inv_sub_bytes(s)
        s = add_round_key(s, key)
        s = inv_MixedCol(s)
    key = makeBlock(k[0])
    s = inv_shift_rows(s)
    s = inv_sub_bytes(s)
    s = add_round_key(s, key)
    plain = unBlocker(s)
    return plain


if __name__ == '__main__':
    key = '00000000000000000000000000000000'

    # 1
    print("\nSOAL 1:")
    name = "Mohammad Ghorbanpoor Aref Yazdkhasti Project AES"
    hex_name = textToHex(name)
    print("name: ", name)
    print("HEX name : ", hex_name)
    cypher = ''
    for i in range (int(len(hex_name) / 32)):
        hex1 = hex_name[i*32:i*32+32]
        cypher += encrypt(hex1, key)

    print("cypher: ", cypher)

    plain = ''
    for i in range(int(len(cypher) / 32)):
        hex1 = cypher[i * 32:i * 32 + 32]
        plain += decrypt(hex1, key)

    print("plain: ", plain)

    print("main text:", hexToText(plain))

    print("Hamming Dis between main HEX text and cipher: ", hammingDist(hexToBin(hex_name), hexToBin(cypher)))


    # 2
    print("\nSOAL 2:")
    name2 = "Mohammad Ghorbanpoor Bref Yazdkhasti Project AES"
    hex_name2 = textToHex(name2)
    print("name: ", name2)
    print("HEX name : ", hex_name2)
    cypher2 = ''
    for i in range(int(len(hex_name2) / 32)):
        hex2 = hex_name2[i*32:i*32+32]
        cypher2 += encrypt(hex2, key)

    print("cypher: ", cypher2)

    plain2 = ''
    for i in range(int(len(cypher2) / 32)):
        hex2 = cypher2[i * 32:i * 32 + 32]
        plain2 += decrypt(hex2, key)

    print("plain: ", plain2)

    print("main text:", hexToText(plain2))

    print("Hamming Dis between main HEX text and cipher: ", hammingDist(hexToBin(hex_name2), hexToBin(cypher2)))

    # 3
    print("\nSOAL 3:")
    print("Hamming Dis between cypher 1 , 2: ", hammingDist(hexToBin(cypher), hexToBin(cypher2)))
    
    # 4
    print("\nSOAL 4:")
    key_list1 = keyExpansion('00000000000000000000000000000000')
    key_list2 = keyExpansion('00000000000000000000000000000001')

    for index in range(10):
        print("round", index, "Key:", key_list1[index])
        print("round", index, "Key':", key_list2[index])
        print("Hamming Distance between ", index, "round keys", hammingDist(hexToBin(key_list1[index]), hexToBin(key_list2[index])))


    # 5
    print("\nSOAL 5:")
    text = 'Mohammad Ghorbanpoor Aref Yazdkhasti Project AES'
    hexText = textToHex(text)
    cypher = ''
    plainHex = ''
    for i in range(int(len(hexText) / 32)):
        temp = encrypt(hexText[32 * i:32 * i + 32], key)
        temp = hexToBin(temp)
        temp = temp[:23] + '0' + temp[24:]
        temp = binToHex(temp)
        cypher += temp
        plainHex += decrypt(temp, key)
    print('name:', text)
    print('HEX name:', hexText)
    print('cypher:', cypher)
    print('plain:', plainHex)
    print('Hamming Dis between main binary text and binary plain:', hammingDist(hexToBin(hexText), hexToBin(plainHex)))

