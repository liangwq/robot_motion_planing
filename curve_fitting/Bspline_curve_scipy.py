import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# 输入控制点
control_points = np.array([[0, 0], [1, 3], [2, -1], [3, 2], [4, 0]])

# B样条曲线的阶数
degree = 3

# 创建B样条曲线对象
bspline = BSpline(np.arange(len(control_points) + degree + 1) - degree, control_points, degree)

# 生成曲线上的点
t = np.linspace(0, len(control_points) - degree, 1000)
curve_points = bspline(t)

# 绘制B样条曲线
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
plt.legend()
plt.show()
