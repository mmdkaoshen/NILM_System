# coding=gbk
import numpy as np


onThreadMap = {
    0: 0.02,
    1: 0.8,
    2: 0.04,
    3: 1.0,
    4: 0.02,
}

offThreadMap = {
    0: -0.02,
    1: -0.4,
    2: -0.06,
    3: -1.0,
    4: -0.03,
}


def getInstantaneousPower(voltage, current):
    """
    计算瞬时功率
    :param voltage: 输入电压
    :param current: 输入电流
    :return: 瞬时功率
    """
    timeSeq = len(voltage)
    instantaneousPower = [abs(v * c / timeSeq) for v, c in zip(voltage, current)]
    return instantaneousPower


def getPowerConsumption(voltage, current):
    """
    计算有功功率
    :param voltage: 输入电压
    :param current: 输入电流
    :return: 消耗电量
    """
    instantaneousPower = getInstantaneousPower(voltage, current)

    return sum(instantaneousPower)


def getSlideWindowDetectResult(idx, current, winNum, winWidth) -> int:
    onFlag = False
    offFlag = False
    win_len = winWidth * winNum
    window = [0] * int(win_len / 2)
    res = []
    current = window + current

    for i in range(len(current) - win_len):
        window = current[i:(i + win_len)]
        pre_avg = sum(window[:winWidth]) / winWidth
        aft_avg = sum(window[(winNum - 1) * winWidth:]) / winWidth
        delta = aft_avg - pre_avg
        res.append(delta)

    if max(res) > onThreadMap[idx]:
        onFlag = True
    if min(res) < offThreadMap[idx]:
        offFlag = True
    if onFlag and not offFlag:
        return 1
    elif offFlag and not onFlag:
        return -1
    return 0


def getActiveCurrent(current, sampleFreq, baseFreq) -> list:
    current = list(map(abs, current))
    seqLength = int(sampleFreq / baseFreq)  # 20
    zeroPadding = [0.0] * int(seqLength / 2)
    paddingCurrent = zeroPadding + current + zeroPadding
    activeCurrent = []
    for i in range(int(seqLength / 2), len(paddingCurrent) - int(seqLength / 2)):
        activeCurrent.append(sum(paddingCurrent[i - int(seqLength / 2):i + int(seqLength / 2)]) / seqLength)

    return activeCurrent


def detectSwitchStatus(idx: int, current: list) -> int:
    activeCurrent = getActiveCurrent(current, 1000, 50)
    status = getSlideWindowDetectResult(idx, activeCurrent, 4, 20)

    return status
