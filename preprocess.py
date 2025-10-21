import numpy as np
from scipy import signal
from EEG_utils import EEGUtils


class Preprocess(EEGUtils):
    def __init__(self):
        super().__init__()

    def eegFilt(self, data, srate, locutoff=None, hicutoff=None, filtorder=None, revfilt=False):
        # 滤波函数
        channel, frames = data.shape[0], data.shape[1]
        nyq = srate * 0.5
        MINFREQ = 0

        trans = 0.15  # fractional width of transition zones

        band = [MINFREQ, (1 - trans) * locutoff / nyq, locutoff / nyq, hicutoff / nyq, (1 + trans) * hicutoff / nyq, 1]

        desired = [0, 0, 1, 1, 0, 0]
        if revfilt:
            desired = [1, 1, 0, 0, 1, 1]

        # get FIR filter coefficients
        fir_firls = signal.firls(filtorder, band, desired)
        smooth_data = np.zeros((channel, frames))
        for i in range(channel):
            smooth_data[i] = signal.filtfilt(fir_firls, 1, data[i], padtype='odd',
                                             padlen=3 * (max(len(fir_firls), len([1])) - 1))
        return smooth_data

    def preprocessing(self, data):
        # 带通滤波 0.5-75
        data = self.eegFilt(data, 1000, 0.5, 75, 101)
        # 陷波函数 50
        data = self.eegFilt(data, 1000, 49, 51, 101, True)
        # 去除基线漂移
        data = signal.detrend(data, axis=1)
        return data

    def get_DE(self, data):
        DE = self.calculateDE(data, 1000, None, 'hamming', 'linear')
        return DE

    def get_PLV(self, data, b):
        filt_data = self.eegFilt(data, 1000, b[0], b[1], 101)
        plv = self.calculatePLV(filt_data)
        plv = np.expand_dims(plv, -1)
        return plv

    def calculate_feature(self, data):
        # 特征工程得到DE特征以及PLV脑网络
        DE = self.get_DE(data)
        # 获取每个频段下的脑网络特征
        for b in self.freq_band:
            plv = self.get_PLV(data, b)
            if b == self.freq_band[0]:
                PLV_Feature = plv
            else:
                PLV_Feature = np.append(PLV_Feature, plv, axis=-1)
        return DE, PLV_Feature

    def process_data(self, eeg_data):
        # 预处理得到干净的信号
        processed_data = self.preprocessing(eeg_data)  # 0.02-0.04s
        # 特征工程获取特征
        de, plv = self.calculate_feature(processed_data)
        return de, plv
