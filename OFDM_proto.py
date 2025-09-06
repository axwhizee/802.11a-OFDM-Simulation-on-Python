import numpy as np
import matplotlib.pyplot as plt
from math import log2
from commpy.channelcoding.convcode import viterbi_decode, Trellis, conv_encode, puncturing, depuncturing
from commpy.channels import awgn

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']       # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False                  # 正常显示负号

# 参数：trs_rate=36Mbps | 16QAM | code_rate=3/4 | Nbpsc=4 | Ncbps=192 | Ndbps=144

# 用户定义参数
Ns = 64             # 子载波总数
Nsd = 48            # 数据子载波数
code_rate = 3/4     # 编码速率，可以是1/2、2/3、3/4
sample_rate = 20e6  # 采样率，单位Hz
modulate = "16QAM"  # 调制方式
pilotValue = 3+3j   # 导频格式
pilotCarrier = np.array([11, 25, 38, 52])       # 导频子载波位置
scmb_seed = np.array([1, 0, 1, 1, 1, 0, 1])     # 扰码种子（初始值）

# 子载波分布
allCarriers = np.arange(Ns)                     # 载波序列
spareCarrier = np.concatenate((allCarriers[:6], allCarriers[-6:]))      # 空闲子载波位置
dataCarriers = np.delete(allCarriers, np.concatenate((pilotCarrier, spareCarrier)))

# 计算参数
# Nsp = Ns - Nsd      # 4个导频子载波
# Nst = Nsp + Nsd # 共有52个有效子载波
Nbpsc = {"BPSK": 1,
         "QPSK": 2,
         "16QAM": 4,
         "64QAM": 6}[modulate]   # 每子载波中的比特数
psy_rate = sample_rate * Nsd // Ns      # 物理传输速率
# 传输速率，$Rate=Psy\_rate\times{R}\times{mod}\times0.8$，0.8为数据占比3.2/4.0（0.8us保护间隔）
trs_rate = psy_rate * code_rate * Nbpsc * 0.8
Ncbps = Nbpsc * Nsd                     # 每个OFDM符号中经过编码的比特数
Ndbps = Ncbps * code_rate               # 每个OFDM符号的比特数

MAC_header = np.array([0x04, 0x02, 0x00, 0x2E, 0x00, 0x60, 0x08, 0xCD,
                    0x37, 0xA6, 0x00, 0x20, 0xD6, 0x01,\
                    0x3C, 0xF1, 0x00, 0x60, 0x08, 0xAD,\
                    0x3B, 0xAF, 0x00, 0x00], dtype = np.uint8)    # MAC报头（例子）
message = "Joy, bright spark of divinity,\x0a\
Daughter of Elysium,\x0aFire-insired we trea"
# d\x0aThy sanctuary.\x0aThy magic power re-unites\x0a\
# All that custom has divided,\x0aAll men become broth\
# ers\x0aUnder the sway of thy gentle wings.."                          # 消息
CRC_checksum = np.array([0x67, 0x33, 0x21, 0xB6], dtype = np.uint8)       # CRC校验和（例子）

# 生成Data比特流
def data_packer():
    """
    打包生成data原始数据

    :return data_tx: 生成的data原始数据
    """
    global MAC_header       # 调用参数
    global message          # 调用参数
    global CRC_checksum     # 调用参数
    global Ndbps            # 调用参数
    MAC_header = np.unpackbits(MAC_header)      # 将每个字节转换为 8 位的二进制序列
    message = np.frombuffer(message.encode('ascii'), dtype=np.uint8)    # 将消息转为ascii码
    message = np.unpackbits(message)    # 将每个字节转换为 8 位的二进制序列
    CRC_checksum = np.unpackbits(CRC_checksum)      # 将每个字节转换为 8 位的二进制序列

    PSDU_size = MAC_header.size + message.size + CRC_checksum.size      # PSDU长度
    data_size = int(np.ceil((PSDU_size + 22) / Ndbps) * Ndbps)      # 基于Ndbps计算data长度
    pad_size= data_size - PSDU_size - 16    # 计算Tail尾及附加段的长度
    # 组成DATA：SERVICE(16b)、MAC报头(192b)、消息(~b)、CRC校验和(32b)、Tail尾(6b)、附加(~b)
    data_tx = np.concatenate((np.zeros(16), MAC_header, message, CRC_checksum, np.zeros(pad_size)))
    return data_tx.astype(np.uint8)     # 将Data变为整型，便于后续调用

def gen_signal(data):
    """
    Signal段生成器

    :param data: data比特流
    :return signal: Signal段比特流
    """
    Rate = np.array({6:[1, 1, 0, 1], 9:[1, 1, 1, 1],
                    12:[0, 1, 0, 1], 18:[0, 1, 1, 1],
                    24:[1, 0, 0, 1], 36:[1, 0, 1, 1],
                    48:[0, 0, 0, 1], 54:[0, 0, 1, 1]}[int(trs_rate // 10e5)])
    data_len = "".join(reversed(bin(data.size)[2:].zfill(12)))
    data_len = np.array(list(data_len), dtype = np.int8)                # 数据长度位
    signal = np.concatenate((Rate, np.zeros(1), data_len))
    check = np.sum(signal) % 2                                          # 校验和
    signal = np.concatenate((signal, np.array([check]), np.zeros(6)))
    return signal.astype(np.int8)

# 生成前导序列
def gen_Training_Symbol():
    """
    生成前导复数训练序列

    :return training_symbol: 生成的前导训练复数序列
    """
    global Ns
    short_num = 10      # 短训练符号数量
    long_num = 2        # 长训练符号数量
# 短训练符号（Short Training Sequence）,将倒数第三位变正，再将0~31位与32~63位互换位置
# short_seq = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0,\
#                       1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0,\
#                       0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0,\
#                       -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0])  # 64位
    short_seq = np.array([0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0,\
                        +1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0,\
                        1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0])
    short_seq = short_seq * np.sqrt(13 / 6)     # 归一化
    short_train = np.fft.ifft(short_seq, Ns)
    short_train = np.tile(short_train[-16:], short_num)       # 取最后16个采样点作为循环前缀重复10次，长度160

# 长训练符号（Long Training Sequence）,操作同上
# long_seq = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,\
#                      1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,\
#                      0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,\
#                      1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    long_seq = np.array([0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,\
                        1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0,\
                        0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,\
                        1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])
    long_train = np.fft.ifft(long_seq, Ns)
    long_train = np.tile(long_train, long_num)      # 循环前缀重复2次，长度128
    long_train = np.concatenate((long_train[-32:], long_train))     # 最后32个采样点作为循环前缀拼接，长度160

    long_train = np.append(long_train,long_train[0])    # 长序列补1
    long_train[0] = (short_train[0] + long_train[0]) * 0.5      # 耦合长短序列
    training_symbol = np.concatenate((short_train,long_train))      # 拼接形成完整前导训练序列
    training_symbol[0] *= 0.5
    training_symbol[-1] *= 0.5      # 整体加窗
    return training_symbol

####################################################################################################

def scrambler(data_tx, scmb_seed):
    """
    扰码器

    :param data_tx: 原始data比特流
    :param scmb_seed: 加扰器初始值
    :return data_scmd: 解码后比特流
    """
    data_size = data_tx.size                            # 提前求数据长度，减少计算量
    data_scmd = np.zeros(data_size, dtype = np.int8)    # 初始化一个空数组用于存储生成的扰码序列
    for i in range(data_size):      # 处理data_size次
        # x0 = x4 + x7 (异或运算)结果放入扰码序列
        data_scmd[i] = scmb_seed[3] ^ scmb_seed[6]      # 结果输出
        scmb_seed = np.roll(scmb_seed, 1)       # 将寄存器左移一位
        scmb_seed[0] = data_scmd[i]     # 将异或结果放回0位
        data_scmd[i] = data_tx[i] ^ data_scmd[i]    # 输出数据与原始数据异或
    return data_scmd

def descrambler(data_rx):
    """
    解扰器

    :param data_dcvlt: 维特比译码后比特流
    :return data_dscmd: 解扰后比特流
    """
    data_size = data_rx.size                        # 提前求数据长度，减少计算量
    dscmb_s_seed = data_rx[0:7]                     # 取data中Service段前7位，用以计算初始状态
    dscmb_seed = np.zeros(7, dtype = np.int8)       # 存放还原的初始状态
    for i in range(7):      # 处理7次,还原初始状态
        dscmb_seed[i] = dscmb_s_seed[0] ^ dscmb_s_seed[4]       # 异或运算，结果放入dscmb_seed
        dscmb_s_seed = np.roll(dscmb_s_seed, -1)    # 将寄存器右移一位
        dscmb_s_seed[6] = dscmb_seed[i]     # 将异或结果放回6位

    data_dscmd = np.zeros(data_size, dtype = np.int8)       # 初始化一个空数组用于存储生成的解扰码序列
    for i in range(data_size):      # 处理data_dcvtd.size次（）
        data_dscmd[i] = dscmb_seed[3] ^ dscmb_seed[6]       # 异或计算，结果放入data_dscmd
        dscmb_seed = np.roll(dscmb_seed, 1)     # 将寄存器左移一位
        dscmb_seed[0] = data_dscmd[i]       # 将异或结果放回0位
        data_dscmd[i] = data_rx[i] ^ data_dscmd[i]      # 输出数据与扰码数据异或
    return data_dscmd

# 卷积与解卷积
L = 7                                       # 约束长度
generator_matrix = np.array([[109, 79]])    # 生成矩阵
trellis = Trellis(np.array([L-1]), generator_matrix)    # 卷积编码器寄存器
puncture_martix = np.array([1, 1, 1, 0, 0, 1])          # 删余矩阵

def convoluter(data_tx):
    """
    卷积码编码器

    :param data_scmd: 加扰后比特流
    :return data_cvlt: 卷积编码并删余后比特流
    """
    # 卷积编码
    data_size = data_tx.size                                    # 求数据长度，减少计算量
    data_m_cvlt = np.zeros(data_size * 2, dtype = np.int8)      # 存放编码生成的两倍data长的数据
    cvtd_seed = np.zeros(6, dtype = np.int8)                    # 编码寄存器长度6
    for i in range(data_size):      # 处理data_size次
        # 异或运算结果放入扰码序列
        data_m_cvlt[2 * i] = data_tx[i] ^ cvtd_seed[1] ^ cvtd_seed[2] ^ cvtd_seed[4] ^ cvtd_seed[5]     # 偶数位
        data_m_cvlt[2 * i + 1] = data_tx[i] ^ cvtd_seed[0] ^ cvtd_seed[1] ^ cvtd_seed[2] ^ cvtd_seed[5] # 奇数位
        cvtd_seed = np.roll(cvtd_seed, 1)       # 将寄存器右移一位
        cvtd_seed[0] = data_tx[i]       # 将data输入寄存器0位
    # 删余
    data_cvlt = np.array([], dtype = np.int8)       # 创建空数据以便存储
    if (code_rate == 3/4):      # 3/4编码速率
        dlt_len = 6     # 用于删余
        for i in range(data_size * 2):
            if (i % dlt_len != 3) and (i % dlt_len != 4):       # 仅在不为第4或第5位时，传递数据
                data_cvlt = np.append(data_cvlt,[data_m_cvlt[i]])       # 拼接数据
    return data_cvlt

def deconvoluter(data_rx):
    """
    Viterbi解码器

    :param data_rx: 解交织后比特流
    :return data_dcvlt: 补哑元并用Viterbi解码后比特流
    """
    # 补哑元
    data_m_dcvlt = np.array([], dtype = np.int8)      # 创建空数据以便储存
    if (code_rate == 3/4):      # 3/4编码速率
        add_len = 4     # 用于补哑元
        for i in range(data_rx.size):
            data_m_dcvlt = np.append(data_m_dcvlt, [data_rx[i]])
            if (i % add_len == 2):      # 第3位时，插入两哑元
                data_m_dcvlt = np.append(data_m_dcvlt, [0, 0])
    # return viterbi_decoder(data_m_dcvlt, 4)     # 最后进行Viterbi译码
    global generator_matrix
    global trellis
    tb_depth = 5*(np.array([L-1]).sum() + 1)    # 回溯深度
    return viterbi_decode(data_m_dcvlt, trellis, tb_depth)

def viterbi_decoder(data_rx, accuracy = 4):
    """
    Viterbi译码器

    :param data_rx: 接收到的比特流
    :param accuracy: 译码精度（码距计算深度），越大越高

    :return data_decoded: 解码后的比特序列

    尚未完成
    """
    data_rx = data_rx.tolist()          # 使用原生python数据结构以便高效运行（怀疑）
    data_decod = []                     # 储存解码结果
    dst_buf = [0]                       # 缺省码距容器，初始值为0
    dst_num = 1                         # 码距容器中有效码距的数量，初始值为1
    state_buf = [[0, 0, 0, 0, 0, 0]]    # 缺省寄存器状态容器，初始值为全0
    buffer = [0, 0, 0, 0]               # 4位数组，高2位用于存储数据，低2位储存码距
    for d in range(0, len(data_rx), 2):     # 每两位数据为一次循环，解码出一位数据
        buffer[2:4] = [data_rx[d], data_rx[d + 1]]      # 储存当前数据
        for t in range(dst_num):    # 处理所有已储存码距对应的状态
            state_tmp = state_buf[t + dst_num - 1]      # 取出寄存器状态
            for i in [0, 1]:    # 每个状态处理两次，分别输入0和1
                # 下面计算卷积，同时计算码距
                buffer[0] = state_tmp[1] ^ state_tmp[2] ^ state_tmp[4] ^ state_tmp[5] ^ i ^ buffer[2]
                buffer[1] = state_tmp[0] ^ state_tmp[1] ^ state_tmp[2] ^ state_tmp[5] ^ i ^ buffer[3]
                dst_buf += [buffer[0] + buffer[1] + dst_buf[(len(dst_buf) - 1) // 2]]     # 累积码距（与上级相加）
                state_buf += [[i] + state_tmp[1:6]]     # 储存状态，保持状态数和码距数对齐
        dst_num *= 2    # 码距数量翻倍，为当前含有可能解码结果的状态的数量（树状图最外侧）
        dst_buf[:(dst_num - 1)] = (dst_num - 1) * [accuracy * 2]    # 令之前的码距失效，即赋accuracy * 2
        if (dst_num >= accuracy) or (d + 2 == len(data_rx)):        # 若有效码距数达到精度
            dst_min = dst_buf.index(min(dst_buf[-dst_num:]))    # 在最终dst_num个码距中找出最小码距的索引
            # 取出相应的解码结果dst_min行的前np.log2(dst_num)
            data_decod += (state_buf[dst_min])[int(log2(dst_num)) - 7:-7:-1]       # 取出译码结果
            state_buf = [state_buf[dst_min]]    # 完成解码后，将最小码距相应状态保留
            dst_buf = [0]       # 缺省码距容器
            dst_num = 1     # 码距数量恢复为1
    data_decod = np.array(data_decod, dtype = np.int8)      # 转回numpy数据结构
    return data_decod

def cpmmpy_convcode(data_tx):
    global generator_matrix     # 生成矩阵
    global trellis              # 寄存器结构
    global puncture_martix
    # , puncture_matrix = np.array([[1, 1, 1, 0, 0, 1]])
    data_m_tx = conv_encode(data_tx, trellis, termination = "cont")
    # data_cvlt = puncturing(data_m_tx, puncture_martix)    # 删余后长度1152
    # return data_cvlt
    return data_m_tx

def cpmmpy_deconvcode(data_rx):
    global generator_matrix     # 生成矩阵
    global trellis              # 寄存器结构
    global puncture_martix
    tb_depth = 5*(np.array([L-1]).sum() + 1)    # 回溯深度
    # data_m_rx = depuncturing(data_rx, puncture_martix, data_rx.size * 6 // 4)
    # data_decvlt = viterbi_decode(data_m_rx, trellis, tb_depth)# BER:0.4212962962962963
    data_decvlt = viterbi_decode(data_rx, trellis, tb_depth)# BER:0.4212962962962963
    return data_decvlt

def interleaver(data_tx):
    """
    交织器

    :param data_tx: 卷积编码后的比特流
    :return data_itlv: 交织后的比特流
    """
    global Ncbps                                        # 调用参数
    global Nbpsc                                        # 调用参数
    data_size = data_tx.size                            # 求数据长度，减少计算量
    Ncbps_d16 = Ncbps // 16                             # 预处理，减少计算量
    s = max(Nbpsc // 2, 1)                              # 求s，减少计算量
    num = data_size // Ncbps                            # 求处理批数
    data_itlv = np.zeros(data_size, dtype = np.int8)    # 交织结果
    itlv_list = np.zeros(Ncbps, dtype = np.int16)       # 交织映射表
    itlv_m_list = np.zeros(Ncbps, dtype = np.int16)     # 交织映射中间表
    for k in range(Ncbps):      # 根据计算式求一次交织映射表
        itlv_m_list[k] = np.int16(Ncbps_d16 * (k % 16) + k // 16)
    for i in range(Ncbps):      # 根据计算式求交织映射表
        itlv_list[i] = itlv_m_list[np.int16(s * (i // s) + (i + Ncbps - i // Ncbps_d16) % s)]
        # 根据计算式求二次交织映射表
        # itlv_m_list[i] = int(s * (i // s) + (i + Ncbps - i // Ncbps_d16) % s)
    for t in range(num):
        num = t * Ncbps     # 求批次索引基数
        for i in range(Ncbps):
            data_itlv[num + i] = data_tx[num + itlv_list[i]]     # 按批数与映射表进行交织映射
    return data_itlv

def deinterleaver(data_rx):
    """
    解交织器

    :param data_rx:   接收到的比特数据流
    :return data_deitlv: 解交织后的比特数据流
    """
    global Ncbps                                            # 调用参数
    global Nbpsc                                            # 调用参数
    data_size = data_rx.size                                # 求数据长度，减少计算量
    Ncbps_d16 = Ncbps // 16                                 # 预处理，减少计算量
    s = max(Nbpsc // 2, 1)                                  # 求s，减少计算量
    num = data_size // Ncbps                                # 求处理批数
    data_deitlv = np.zeros(data_size, dtype = np.int8)      # 解交织结果
    deitlv_list = np.zeros(Ncbps, dtype = np.int16)         # 解交织映射表
    deitlv_m_list = np.zeros(Ncbps, dtype = np.int16)       # 解交织映射中间表
    for j in range(Ncbps):      # 根据计算式求一次解交织映射表
        deitlv_m_list[j] = int(s * (j // s) + (j + j // Ncbps_d16) % s)
    for i in range(Ncbps):      # 根据计算式求解交织映射表
        deitlv_list[i] = deitlv_m_list[int(16 * i - (Ncbps - 1) * (i // Ncbps_d16))]
        # 根据计算式求二次解交织映射表
        #deitlv_m_list = int(16 * i - (Ncbps - 1) * (i // Ncbps_d16))
    for t in range(num):
        num = t * Ncbps     # 求批次索引基数
        for i in range(Ncbps):
            data_deitlv[num + i] = data_rx[num + deitlv_list[i]]     # 按批数与映射表进行解交织映射
    return data_deitlv

def modulater(data_tx):
    """
    星座映射调试器

    :param data_tx: 交织映射的比特流
    :return data_mapped: 星座映射后的复数序列
    """
    global modulate                 # 调取参数
    data_size = data_tx.size        # 求数据长度，减少计算量
    data_tx = np.bool(data_tx)      # 转化为bool值，以便计算
    data_tx = ~ data_tx             # 对所有数据取反，便于后续映射计算
    if (modulate == "BPSK"):
        data_mapped = np.array([], dtype = np.complex64)     # 存放BPSK映射结果
        for i in range(data_size):
            data_mapped[i] = (-1) ** data_tx    # BPSK映射关系
    elif (modulate == "QPSK"):
        data_mapped = np.array([], dtype = np.complex64)    # 存放QPSK映射结果
        for i in range(0, data_size, 2):
            # QPSK映射计算式：$(-1)^{i_0}+(-1)^{i_1}{j}$
            data_mapped = np.append(data_mapped, [(-1) ** data_tx[i] + (-1) ** data_tx[i + 1] * 1j])
        # data_mapped /= np.sqrt(2)       # 归一化
    elif (modulate == "16QAM"):
        data_mapped = np.array([], dtype = np.complex64)    # 存放16QAM映射结果
        for i in range(0, data_size, 4):
            # 16QAM映射计算式：$(-1)^{i_0}\times3^{i_1} + (-1)^{i_2}\times3^{i_3}{j}$
            data_mapped = np.append(data_mapped, [(-1) ** data_tx[i] * 3 ** data_tx[i + 1] \
            + (-1) ** data_tx[i + 2] * 3 ** data_tx[i + 3] * 1j])
        # data_mapped /= np.sqrt(10)      # 归一化
    elif (modulate == "64QAM"):
        data_mapped = np.array([], dtype = np.complex64)    # 存放64QAM映射结果
        for i in range(0, data_size, 6):
            # 64QAM映射计算式：
            # $(-1)^{i_0}\times(4\times{\bar{i_1}}+3^{i_2})+(-1)^{i_3}+\times(4\times{\bar{i_4}}+3^{i_5}{j})$
            data_mapped = np.append(data_mapped, [\
            (-1) ** data_tx[i] * (4 * (not data_tx[i + 1]) + 3 ** data_tx[i + 2]) \
            + (-1) ** data_tx[i + 3] * (4 * (not data_tx[i + 4]) + 3 ** data_tx[i + 5]) * 1j])
        # data_mapped /= np.sqrt(42)      # 归一化
    else:
        print("ERROR 调制错误")
    return data_mapped

def demodulater(data_rx):
    """
    星座映射解调器

    :param data_rx: 接收到的复数序列
    :return data_demapped: 解映射的的比特流
    """
    global modulate     # 调取参数
    data_size = data_rx.size    # 求数据长度，减少计算量
    if (modulate == "BPSK"):
        data_demapped = np.zeros(data_size, dtype = np.bool)
        for i in range(data_size):
            data_demapped[i] = (data_rx[i] > 0)     # 1为True（1），否则（-1）为False（0）
    elif (modulate == "QPSK"):
        data_demapped = np.zeros(data_size * 2, dtype = np.bool)
        for i in range(data_size):
            j = i * 2
            data_demapped[j] = (data_rx[i].real > 0)
            data_demapped[j + 1] = (data_rx[i].imag > 0)
    elif (modulate == "16QAM"):
        data_demapped = np.zeros(data_size * 4, dtype = np.bool)
        for i in range(data_size):
            j = i * 4
            data_demapped[j] = (data_rx[i].real > 0)
            data_demapped[j + 1] = (abs(data_rx[i].real) < 2)       # 实部绝对值小于2即True
            data_demapped[j + 2] = (data_rx[i].imag > 0)
            data_demapped[j + 3] = (abs(data_rx[i].imag) < 2)       # 虚部绝对值小于2即True
    elif (modulate == "64QAM"):
        data_demapped = np.zeros(data_size * 6, dtype = np.bool)
        for i in range(data_size):
            j = i * 6
            data_demapped[j] = (data_rx[i].real > 0)
            data_demapped[j + 1] = (abs(data_rx[i].real) < 2)
            data_demapped[j + 2] = (8 - (abs(data_rx[i].real)) > 2)     # 实部绝对值-4大于2即True***
            data_demapped[j + 3] = (data_rx[i].imag > 0)
            data_demapped[j + 4] = (abs(data_rx[i].imag) < 2)
            data_demapped[j + 5] = (8 - (abs(data_rx[i].imag)) > 2)     # 虚部绝对值-4大于2即True
    else:
        print("ERROR 调制错误")
    return np.int8(data_demapped)       # 将结果转换回int8

####################################################################################################

def OFDM_packer(data_tx):
    """
    OFDM符号打包

    :param data_tx: 发送的复数序列
    :return symbols: 生成的OFDM符号二维表
    """
    global pilotCarrier
    global dataCarriers
    global pilotValue
    global Ns
    global Nsd
    data_size = data_tx.size
    symbol = np.zeros(Ns, dtype = np.complex64)     # 单OFDM符号
    symbol[pilotCarrier] = pilotValue               # 提前在导频位置插入导频
    symbols = np.zeros((0, Ns))                     # 二维OFDM符号序列
    for i in range(data_size // Nsd):
        symbol[dataCarriers] = data_tx[i * Nsd:(i + 1) * Nsd]       # 在数据位置插入数据
        symbols = np.vstack((symbols, symbol))      # 拼接OFDM符号序列
    return symbols

def IDFT_adcp(symbols):
    """
    OFDM符号进行逆快速傅里叶变换生成发送信号

    :param symbols: OFDM符号
    :return signal: 发送信号
    """
    symbol_num = symbols.shape[0]               # 获取符号数
    signal_tx = np.zeros(0, dtype = np.int8)    # 发送比特流
    for i in range(symbol_num):
        data = np.fft.ifft(symbols[i], Ns)          # 对每个OFDM符号进行逆傅里叶变换
        signal_tx = np.append(signal_tx, data)      # 拼接生成的比特序列
    return signal_tx

def DFT_rmcp(signal_rx):
    """
    发送信号进行快速傅里叶变换生成OFDM符号

    :param signal_rx: 接收信号
    :return symbols: OFDM符号
    """
    global Ns
    # data_rx = data_rx[16:(16+64)]       # 去循环前缀
    data_size = signal_rx.size      # 获取符号数
    symbols = np.zeros((0, Ns))     # 二维OFDM符号序列
    for i in range(data_size // 64):
        data = np.fft.fft(signal_rx[i * Ns:(i + 1) * Ns], Ns)       # 每64一次傅里叶变换
        symbols = np.vstack((symbols, data))    # 拼接转换后的OFDM符号
    return symbols

def OFDM_loader(symbols):
    """
    读取OFDM符号

    :param symbols: OFDM符号
    :return data_rx: 接收的复数序列
    """
    global dataCarriers
    global Nsd
    symbol_num = symbols.shape[0]                               # 获得OFDM符号数
    data_rx = np.zeros(symbol_num * Nsd, dtype = np.complex64)  # 从OFDM符号恢复的数据
    for i in range(symbol_num):
        data_rx[i * Nsd:(i + 1) * Nsd] = symbols[i][dataCarriers]       # 从数据位读取数据
    return data_rx

def frame_sync(received_signal, Trn_symbols):
    """
    前导序列进行同步，通过自相关计算检测帧的起始位置
    """
    correlation = np.correlate(received_signal, Trn_symbols, mode='full')       # 计算自相关
    peak_index = np.argmax(correlation)     # 找出自相关值的峰值位置
    # return start_index, correlation
    return peak_index + 1       # 输出数据起点

####################################################################################################

def bit_reader(bits_array):
    """
    比特阅读器，将比特流转十六进制字节

    :param bits_array: 读取的比特流
    """
    bytes_array = np.packbits(bits_array)
    hex_array = np.array([f"{byte:02X}" for byte in bytes_array])
    print(hex_array, f"summing: {hex_array.size * 8} bits")
    return 0

def ber_counter(data_tx, data_rx):
    """
    比特误码率计算器

    :param data_tx: 发送端比特流
    :return data_rx: 接收端比特流
    """
    data_tx_size = data_tx.size
    data_rx_size = data_rx.size
    if (data_tx_size >= data_rx_size):      # 若发送端数据量大于接收端
        data_ecr = data_tx[:data_rx_size] ^ data_rx     # 取发送端部分与接收端异或
        be_rate = np.sum(data_ecr) / data_rx_size       # 结果求和，计算误码率
    else:
        data_ecr = data_rx[:data_tx_size] ^ data_tx
        be_rate = np.sum(data_ecr) / data_tx_size
    print(f"BER:{be_rate.item()}")
    return be_rate

def FFT_frg(data, n, stitle):
    """
    绘制比特流的频谱图
    :param data: 比特流数据
    :param n: 绘图位置， 范围1~4
    :param stitle: 图表标题
    """
    plt.subplot(2, 2, n)                    # 绘制子图
    # FFT采样周期为64*0.05us（1/20MHz=0.05us）
    freq = np.abs(np.fft.fft(data, 64))     # 64点快速傅里叶变换
    # 设信号时长为20s，采样20MHz。则信号间隔1ms
    x = np.fft.fftfreq(freq.size, d = 1/trs_rate) / 1000    # 生成频率序列/kHz
    plt.plot(x, freq)
    plt.xlabel("频域 /kHz")
    plt.ylabel("幅值")
    plt.title(stitle)
    plt.tight_layout()      # 自动调整子图布局
    plt.savefig("./frq")

####################################################################################################
# 系统仿真
Data_src = data_packer()
signal = gen_signal(Data_src)
FFT_frg(Data_src, 1, "原始信号")                    # 原始信号幅频响应
# 完整OFDM符号拼接？
data_scrambled = scrambler(Data_src, scmb_seed)     # 扰码
FFT_frg(data_scrambled, 2, "扰码信号")              # 扰码信号幅频响应
data_convoluted = cpmmpy_convcode(data_scrambled)   # 卷积编码
signal_convoluted = cpmmpy_convcode(signal)
FFT_frg(data_convoluted, 3, "卷积编码信号")         # 卷积编码信号幅频响应
data_interleaved = interleaver(data_convoluted)     # 交织
signal_interleaved = interleaver(signal_convoluted)
FFT_frg(data_interleaved, 4, "交织信号")            # 交织信号幅频响应
data_tx = modulater(data_interleaved)               # 星座调制
signal_sec = interleaver(signal_interleaved)

symbol_tx = OFDM_packer(data_tx)                    # 将数据转化为OFDM符号
signal_tx = IDFT_adcp(symbol_tx)                    # 对每个OFDM符号IDFT变为比特流信号
trn_symbol = gen_Training_Symbol()
signal_tx = np.concatenate((trn_symbol, signal_tx))  # 与前导码拼接

snr_list = np.arange(-3, 1, 0.3)                    # 取不同SNR的值
BER_list = np.zeros(snr_list.size, dtype = np.float16)
for i in range(snr_list.size):
    signal_rx = awgn(signal_tx, i)                              # 加性高斯信道
    signal_rx = signal_rx[frame_sync(signal_rx, trn_symbol):]   # 同步定位，截取数据

    symbol_rx = DFT_rmcp(signal_rx)                             # 比特流转符号
    data_rx = OFDM_loader(symbol_rx)                            # 符号获得数据

    data_demapped = demodulater(data_rx)                        # 解调
    data_deinterleaved = deinterleaver(data_demapped)           # 解交织
    data_deconvoluted = cpmmpy_deconvcode(data_deinterleaved)   # 维特比译码
    data_descrambled = descrambler(data_deconvoluted)           # 解扰码
    # bit_reader(data_descrambled)                                # 输出接收数据
    BER_list[i] = ber_counter(Data_src, data_descrambled)       # 计算比特误码率

# plt.close()
# plt.scatter(np.real(data_tx), np.imag(data_tx), color ='blue')
# plt.xlabel("I 轴")                              # x轴名称
# plt.ylabel("Q 轴")                              # y轴名称
# plt.title("16QAM调制信号星座图")                # 图像名称
# plt.grid(True)                                  # 显示网格
# plt.savefig("./modulate.png")                   # 保存为图像
# plt.close()

plt.close()
plt.plot(snr_list, np.log10(BER_list), "o-")
plt.xlabel("SNR /dB")                           # x轴名称
plt.ylabel("BER /lg")                               # y轴名称
plt.title("系统性能")                           # 图像名称
plt.savefig("./BER.png")                        # 保存为图像
plt.close()

# 原始数据备份
# ['00' '00' '04' '02' '00' '2E' '00' '60' '08' 'CD' '37' 'A6' '00' '20'
#  'D6' '01' '3C' 'F1' '00' '60' '08' 'AD' '3B' 'AF' '00' '00' '4A' '6F'
#  '79' '2C' '20' '62' '72' '69' '67' '68' '74' '20' '73' '70' '61' '72'
#  '6B' '20' '6F' '66' '20' '64' '69' '76' '69' '6E' '69' '74' '79' '2C'
#  '0A' '44' '61' '75' '67' '68' '74' '65' '72' '20' '6F' '66' '20' '45'
#  '6C' '79' '73' '69' '75' '6D' '2C' '0A' '46' '69' '72' '65' '2D' '69'
#  '6E' '73' '69' '72' '65' '64' '20' '77' '65' '20' '74' '72' '65' '61'
#  '67' '33' '21' 'B6' '00' '00' '00' '00' '00' '00'] 864 bits
