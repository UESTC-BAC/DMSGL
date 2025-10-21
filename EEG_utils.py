import hdf5storage
import os
from typing import Optional
import numpy as np
from scipy import signal


class EEGUtils:

    def __init__(self, folder: Optional[str] = None, freq_band=None, fs=1000):
        # 数据存放路径
        if freq_band is None:
            freq_band = [[4, 8], [8, 12], [12, 30], [30, 75]]
        self.folder = folder
        self.freq_band = freq_band
        self.fs = fs

    def readMat(self, file: str, key: str):
        """
        从.mat文件读取数据
        :param file: 文件名
        :param key: 文件名想要读取的关键字
        :return: 对应关键字数据
        """
        # 判断传入的参数是否字符
        check_type(file, str)
        check_type(key, str)
        data_dict = hdf5storage.loadmat(os.path.join(self.folder, file))
        data = data_dict[key]
        return data

    def _calculateMeanPSD(self, data, nperseg, noverlap=None, window=None, detrend=None):
        """
        输入1D信号，计算各个频段的平均功率谱
        :param data: 1D信号
        :param nperseg: 窗长
        :param window: 窗类型
        :param detrend: 去趋势的类型
        :return: 各个频段的平均功率谱

        example:
        PSD = EEG_utils._calculateMeanPSD(data, 2000, 'hamming', 'linear') 等价于matlab中自己的pwelch函数
        matlab中的pwelch都是hamming窗
        """
        if nperseg is not None:
            check_type(nperseg, int)
        if noverlap is not None:
            check_type(noverlap, int)
        if window is not None:
            check_type(window, str)


        meanPSDfeature = []
        # 利用welch法计算功率谱
        f, Pxx_den = signal.welch(data, self.fs, noverlap=noverlap, window=window, nperseg=nperseg, detrend=detrend)
        # 计算每个频段内的平均功率谱
        for freq in self.freq_band:
            # 频率分量下限的索引值
            ind1 = np.where(f >= freq[0])[0][0]
            # 频率分量上限的索引值
            ind2 = np.where(f >= freq[1])[0][0]
            # 将这个区间的功率谱密度进行平均
            meanPSD = np.mean(Pxx_den[ind1:ind2])
            meanPSDfeature.append(meanPSD)

        meanPSDfeature = np.array(meanPSDfeature)
        return meanPSDfeature

    def _calculateDE(self, data, nperseg, noverlap=None, window=None, detrend=None):
        """
        输入1D信号，使用log2PSD来近似微分熵，计算各个频段的微分熵特征
        :param data: 1D信号
        :param nperseg: 窗长
        :param window: 窗类型
        :param detrend: 去趋势的类型
        :return: 各个频段的微分熵特征

        example:
        DE = EEG_utils._calculateDE(data, 2000, 'hamming', 'linear')
        """
        meanPSD = self._calculateMeanPSD(data, nperseg, noverlap, window, detrend)
        # 使用log2PSD近似计算微分熵特征
        DE = np.log2(meanPSD)
        return DE

    def changeFolder(self, newFolder: str):
        # 更改默认的路径目录
        check_type(newFolder, str)

        self.folder = newFolder

    def calculatePLV(self, data):
        """
        通过hilbert变换计算脑电信号的锁相值
        :param data: 输入的2D信号
        :return: PLV特征
        """
        M, N = data.shape[0], data.shape[1]
        if M > N:
            channel = N
            data = data.T
        else:
            channel = M
        # 希尔伯特变换构建解析表达式
        data = signal.hilbert(data)
        # 获取信号的瞬时相位
        data_theta = np.unwrap(np.angle(data))
        # 创建plv矩阵
        plvArray = np.zeros((channel, channel))
        # 计算plv
        for i in range(channel - 1):
            for j in range(i + 1, channel):
                plvArray[i, j] = phaseSI(data_theta[i, :], data_theta[j, :])
                plvArray[j, i] = plvArray[i, j]
        plvArray = plvArray + np.eye(channel)
        return plvArray

    def eegFilt(self, data, srate, locutoff=None, hicutoff=None, filtorder=None, revfilt=False):
        """
        改自matlab中的eegfilt函数，使用双向最小二乘 FIR 滤波对数据进行滤波
        :param data: 输入2D的数据，通道*时间点
        :param srate: 采样频率
        :param locutoff: 高通频率
        :param hicutoff: 低通频率
        :param filtorder: 滤波器阶数
        :return: 滤波后的数据
        """
        check_type(data, np.ndarray)
        check_dim(data, 2)

        channel, frames = data.shape[0], data.shape[1]
        nyq = srate * 0.5
        MINFREQ = 0

        minfac = 3  # this many(lo)cutoff - freq cycles in filter
        min_filtorder = 15  # %minimum filter length
        trans = 0.15  # fractional width of transition zones

        if filtorder is None or filtorder == 0:
            if locutoff > 0:
                filtorder = minfac * int(srate / locutoff)
            elif hicutoff > 0:
                filtorder = minfac * int(srate / hicutoff)
            if filtorder < min_filtorder:
                filtorder = min_filtorder

        if filtorder * 3 > frames:
            raise ValueError(f"filter order is {filtorder}, data lengths is {frames}must be at least 3 times the "
                             f"filtorder.")

        if locutoff is not None and hicutoff is not None:
            if locutoff > 0 and 0 < hicutoff < locutoff:
                raise ValueError('locutoff > hicutoff, hicutoff must larger than locutoff')

            if locutoff < 0 or hicutoff < 0:
                raise ValueError('locutoff or hicutoff < 0 locutoff or hicutof must larger than 0')

            if locutoff > nyq:
                raise ValueError('Low cutoff frequency cannot be > srate/2')

            if hicutoff > nyq:
                raise ValueError('High cutoff frequency cannot be > srate/2')

            if (1 + trans) * hicutoff / nyq > 1:
                raise ValueError('high cutoff frequency too close to Nyquist frequency')

            if locutoff > 0 and hicutoff > 0:
                band = [MINFREQ, (1 - trans) * locutoff / nyq, locutoff / nyq, hicutoff / nyq, (1 + trans) * hicutoff / nyq,
                        1]
                # print(f'eegfilt() - low transition band width is {(band[3] - band[2]) * srate / 2 :1.1f} Hz; high trans. band width,{(band[5] - band[4]) * srate / 2 :1.1f} Hz.')
                desired = [0, 0, 1, 1, 0, 0]
                if revfilt:
                    desired = [1, 1, 0, 0, 1, 1]

        elif hicutoff is None:
            if locutoff > nyq:
                raise ValueError('Low cutoff frequency cannot be > srate/2')

            if locutoff <= 0:
                raise ValueError('locutoff  < 0 locutoff  must larger than 0')

            if locutoff > 0:
                if locutoff / nyq < MINFREQ:
                    raise ValueError(f'eegfilt() - highpass cutoff freq must be > {MINFREQ * nyq} Hz')
                band = [MINFREQ, locutoff * (1 - trans) / nyq, locutoff / nyq, 1]
                # filt_hz = (band[3] - band[2]) * srate / 2
                # print(f'eegfilt() - highpass transition band width is {filt_hz :1.1}Hz.')
                desired = [0, 0, 1, 1]

        elif locutoff is None:
            if hicutoff <= 0:
                raise ValueError('hicutoff < 0 hicutof must larger than 0')

            if hicutoff > nyq:
                raise ValueError('High cutoff frequency cannot be > srate/2')

            if (1 + trans) * hicutoff / nyq > 1:
                raise ValueError('high cutoff frequency too close to Nyquist frequency')

            if hicutoff > 0:
                if hicutoff / nyq < MINFREQ:
                    raise ValueError(f'eegfilt() - lowpass cutoff freq must be > {MINFREQ * nyq} Hz')
                band = [MINFREQ, hicutoff / nyq, hicutoff * (1 + trans) / nyq, 1]
                # filt_hz = (band[3] - band[2]) * srate / 2
                # print(f'eegfilt() - lowpass transition band width is {filt_hz:1.1f}Hz.')
                desired = [1, 1, 0, 0]

        else:
            raise ValueError('high and low cutoff frequency is needed')

        if filtorder % 2 == 0:
            raise ValueError(f'filtorder number is {filtorder}, expected odd')

        fir_firls = signal.firls(filtorder, band, desired)  # get FIR filter coefficients
        smooth_data = np.zeros((channel, frames))
        for i in range(channel):
            smooth_data[i] = signal.filtfilt(fir_firls, 1, data[i], padtype='odd', padlen=3 * (max(len(fir_firls), len([1])) - 1))
        return smooth_data

    def changeFreqBand(self, freq_band: list):
        check_type(freq_band, list)

        self.folder = freq_band

    def changesamplefs(self, fs: int):
        check_type(fs, int)

        self.fs = fs

    def calculateMeanPSD(self, data, nperseg, noverlap=None, window=None, detrend=None):
        """
        输入2D的数据，输出2D的平均功率谱，输出每一行信号的平均功率谱
        :param data: 2D信号
        :param nperseg: 窗长
        :param window: 窗类型
        :param detrend: 去趋势
        :return: 2D的平均功率谱

        example:
        PSD = EEG_utils.calculateMeanPSD(data, 2000, 'hamming', 'linear')
        """
        check_type(data, np.ndarray)
        check_dim(data, 2)

        channel, frames = data.shape[0], data.shape[1]
        MeanPSD = []
        for i in range(channel):
            aChannelPSD = self._calculateMeanPSD(data[i, :], nperseg, noverlap, window, detrend)
            MeanPSD.append(aChannelPSD)
        MeanPSD = np.array(MeanPSD)
        return MeanPSD

    def calculateDE(self, data, nperseg, noverlap=None, window=None, detrend=None):
        """
        输入2D的数据，输出2D的微分熵特征，输出每一行信号的微分熵特征
        :param data: 2D信号
        :param nperseg: 窗长
        :param window: 窗类型
        :param detrend: 去趋势
        :return: 2D的微分熵特征

        example:
        DE = EEG_utils.calculateDE(data, 2000, 'hamming', 'linear')
        """
        MeanPSD = self.calculateMeanPSD(data, nperseg, noverlap, window, detrend)
        DE = np.log2(MeanPSD)
        return DE

    def readRawData(self, file: str):
        """
        从.mat文件读取数据
        :param file: 文件名
        :param key: 文件名想要读取的关键字
        :return: 对应关键字数据
        """
        # 判断传入的参数是否字符
        check_type(file, str)
        data_dict = hdf5storage.loadmat(os.path.join(self.folder, file))
        return data_dict

def phaseSI(xr, yr):
    # 瞬时相位差
    xy_theta = xr - yr
    # 相位锁相值
    complex_phase_diff = np.exp(complex(0, 1) * (xy_theta))
    plv = np.abs(np.sum(complex_phase_diff)) / len(xr)
    return plv


def check_type(value, expected_type):
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected type {expected_type}, got {type(value)}")


def check_dim(array, ndim):
    if array.ndim != ndim:
        raise Exception(f"Expected array ndim {ndim}, got {array.ndim}")

