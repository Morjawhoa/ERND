import torch
import numpy as np
from scipy.stats import norm


class RunningStats:
    def __init__(self):
        self.n = 0          # 数据点数量
        self.mean = 0.0      # 当前均值
        self.M2 = 0.0        # 二阶中心矩累积量
        self.maximum = 0.0
        self.minimum = 100

    def _update(self, x):
        """ 更新统计量，严格使用样本方差定义（除以n） """
        self.n += 1
        delta_pre = x - self.mean
        self.mean += delta_pre / self.n  # 更新均值
        delta_post = x - self.mean       # 与新均值的差
        self.M2 += delta_pre * delta_post  # 关键递推公式

        self.maximum = max(self.maximum, x)
        self.minimum = min(self.minimum, x)

    def update_multi(self, nums):
        for num in nums:
            self._update(num)

    @property
    def variance(self):
        """ 样本方差（始终除以n） """
        return self.M2 / self.n if self.n else 0.0

    @property
    def standard_deviation(self):
        """ 样本方差（始终除以n） """
        return (self.M2 / self.n if self.n else 0.0) ** 0.5


# 使用示例
if __name__ == "__main__":
    stats = RunningStats()
    nums = [1,2,3,4,5]
    stats.update_multi(nums)

    nums_norm = (np.array(nums) - stats.mean) / stats.standard_deviation
    nums_norm = norm.cdf(nums_norm)

    print(f"数据点 {nums} => 均值: {stats.mean:.1f}, 方差: {stats.variance:.2f}")

    print(nums_norm)

    # 验证输出
    # 最终结果应与 np.var([3,5,7,9,11]) 相同
