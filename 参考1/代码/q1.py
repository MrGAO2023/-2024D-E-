import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# 参数定义
L = 100  # 潜艇长度
W = 20   # 潜艇宽度
H = 25   # 潜艇高度
sigma = 120  # 水平定位标准差
z0 = 150  # 潜艇深度定位值
kill_radius = 20  # 深弹杀伤半径

# 水平命中概率函数
def P_xy(xd, yd, sigma):
    Px = norm.cdf((L / 2 + xd) / sigma) - norm.cdf((-L / 2 + xd) / sigma)
    Py = norm.cdf((W / 2 + yd) / sigma) - norm.cdf((-W / 2 + yd) / sigma)
    return Px * Py

# 深度命中概率函数
def P_z(z_d, z0, H, kill_radius):
    if z_d < z0 - H / 2:
        return 1.0  # 触发引信引爆
    else:
        return norm.cdf(kill_radius / sigma)

# 总命中概率函数
def P_hit(params):
    xd, yd, zd = params
    return -P_xy(xd, yd, sigma) * P_z(zd, z0, H, kill_radius)

# 优化问题
initial_guess = [0, 0, z0]
result = minimize(P_hit, initial_guess, bounds=[(-L/2, L/2), (-W/2, W/2), (z0-H/2, z0+kill_radius)],
                  method='L-BFGS-B')

# 输出结果
xd_opt, yd_opt, zd_opt = result.x
max_P_hit = -result.fun

print(f"最优投弹平面坐标: ({xd_opt:.2f}, {yd_opt:.2f})")
print(f"最优引爆深度: {zd_opt:.2f} m")
print(f"最大命中概率: {max_P_hit:.5f}")