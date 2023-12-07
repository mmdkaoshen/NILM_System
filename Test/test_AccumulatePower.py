import pytest

from System.AccumulatePower import AccumulatePower


class TestAccumulatePower:
    @pytest.mark.parametrize(
        "resultData, deviceNum, times",
        [
            ([1, 1, 1, 1, 1], 5, 10),
        ]
    )
    def test_tc01(self, resultData: list, deviceNum: int, times: int):
        accumulatePower = AccumulatePower(deviceNum)
        for _ in range(times):
            accumulatePower.accumulateTotalPower(resultData)

        assert accumulatePower.totalPower == [x * times for x in resultData]
