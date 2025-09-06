# 基于Python的802.11a标准OFDM系统仿真

## 介绍
基于Python的IEEE 802.11a标准OFDM系统的部分仿真，包括前导码生成、扰码、卷积编码、交织、映射，及其相应的逆过程（接收方）。

## 依赖

1. numpy
2. matplotlib
3. commpy

## 测试用字符串

```python
# 出自《欢乐颂》
MAC_header = np.array([0x04, 0x02, 0x00, 0x2e, 0x00, 0x60, 0x08, 0xcd,
                    0x37, 0xa6, 0x00, 0x20, 0xd6, 0x01,\
                    0x3c, 0xf1, 0x00, 0x60, 0x08, 0xad,\
                    0x3b, 0xAf, 0x00, 0x00], dtype=np.uint8)        # MAC报头（例子）
message = "Joy, bright spark of divinity,\x0a\
Daughter of Elysium,\x0aFire-insired we trea"
# d\x0aThy sanctuary.\x0aThy magic power re-unites\x0a\
# All that custom has divided,\x0aAll men become broth\
# ers\x0aUnder the sway of thy gentle wings.."                      # 消息
CRC_checksum = np.array([0xDa, 0x57, 0x99, 0xed], dtype=np.uint8)   # CRC校验和（例子）
```

## 参数参考

### 不同速率参数

|Rate   |数据速率   |调制   |编码速率   |BPSC   |CBPS   |DBPS   |
|:-:    |:-:        |:-:    |:-:        |:-:    |:-:    |:-:    |
|1101   |6          |BPSK   |1/2        |1      |48     |24     |
|1111   |9          |BPSK   |3/4        |1      |48     |36     |
|0101   |12         |QPSK   |1/2        |2      |96     |48     |
|0111   |18         |QPSK   |3/4        |2      |96     |72     |
|1001   |24         |16QAM  |1/2        |4      |192    |96     |
|1011   |36         |16QAM  |3/4        |4      |192    |144    |
|0001   |48         |64QAM  |2/3        |6      |288    |192    |
|0011   |54         |64QAM  |3/4        |6      |288    |216    |

### 数据速率Rate

$$
Rate=Psy \_ rate\times{R}\times{mod}\times0.8
$$

1. $Psy \_rate =\frac{1}{20MHz}\times\frac{48}{64}$（64子载波中48个数据子载波）
2. $R$ 为卷积编码删余后的编码率，$\frac{3}{6-2}$或$\frac{2}{4-1}$或$\frac{1}{2}$
3. $mod$ 为相应调制方式下符号所含比特数（1、2、4、6）

$\LARGE0.8 = \LARGE\frac{4.0us-3.2us}{4.0us}$

引用源：

表格：  80211-2000->Page:2810  
数据：  80211-2000->Page:4161 | 802.11a-1999->Page:64

### 星座调制映射计算式

> 输入$i_0、i_1、i_2、i_3$……其中$i_n$均已取$\bar{i_n}$

1. BPSK计算式：$(-1)^{i_0}$
2. QPSK计算式：$(-1)^{i_0}+(-1)^{i_1}{j}$
3. 16QAM计算式：$(-1)^{i_0}\times3^{i_1} + (-1)^{i_2}\times3^{i_3}{j}$
4. 64QAM计算式：$(-1)^{i_0}\times(4\times{\bar{i_1}}+3^{i_2})+(-1)^{i_3}+\times(4\times{\bar{i_4}}+3^{i_5}){j}$

归一化分别除：$1、\sqrt{2}、\sqrt{10}、\sqrt{42}$
