import numpy as np
import matplotlib.pyplot as plt

def basis_function(i, p, u, knots):
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i+1] else 0.0
    else:
        denom1 = knots[i+p] - knots[i]
        denom2 = knots[i+p+1] - knots[i+1]

        term1 = 0.0 if denom1 == 0 else ((u - knots[i]) / denom1) * basis_function(i, p-1, u, knots)
        term2 = 0.0 if denom2 == 0 else ((knots[i+p+1] - u) / denom2) * basis_function(i+1, p-1, u, knots)

        return term1 + term2

def nurbs_curve(control_points, weights, degree, knots, num_samples=100):
    u_values = np.linspace(knots[degree], knots[-(degree+1)], num_samples)
    curve_points = np.zeros((num_samples, len(control_points[0])))

    for i in range(len(u_values)):
        u = u_values[i]
        numerator = np.zeros(len(control_points[0]))
        denominator = 0.0

        for j in range(len(control_points)):
            basis = basis_function(j, degree, u, knots)
            weighted_point = weights[j] * basis * np.array(control_points[j])
            numerator += weighted_point
            denominator += weights[j] * basis

        curve_points[i] = numerator / denominator

    return curve_points

# Example usage
degree = 3
control_points = np.array([[0, 54.493, 52.139], [0, 55.507, 52.139], [0, 56.082, 49.615], [0, 56.780, 44.971], [0, 69.575, 51.358],
               [0, 77.786, 58.573], [0, 90.526, 67.081], [0, 105.973, 63.801], [0, 100.400, 47.326],
               [0, 94.567, 39.913],
               [0, 92.369, 30.485], [0, 83.440, 33.757], [0, 91.892, 28.509], [0, 89.444, 20.393], [0, 83.218, 15.446],
               [0, 87.621, 4.830], [0, 80.945, 9.267], [0, 79.834, 14.535], [0, 76.074, 8.522], [0, 70.183, 12.550],
               [0, 64.171, 16.865], [0, 59.993, 22.122], [0, 55.680, 36.359], [0, 56.925, 24.995], [0, 59.765, 19.828],
               [0, 54.493, 14.940], [0, 49.220, 19.828], [0, 52.060, 24.994], [0, 53.305, 36.359], [0, 48.992, 22.122],
               [0, 44.814, 16.865], [0, 38.802, 12.551], [0, 32.911, 8.521], [0, 29.152, 14.535], [0, 28.040, 9.267],
               [0, 21.364, 4.830], [0, 25.768, 15.447], [0, 19.539, 20.391], [0, 17.097, 28.512], [0, 25.537, 33.750],
               [0, 16.602, 30.496], [0, 14.199, 39.803], [0, 8.668, 47.408], [0, 3.000, 63.794], [0, 18.465, 67.084],
               [0, 31.197, 58.572], [0, 39.411, 51.358], [0, 52.204, 44.971], [0, 52.904, 49.614], [0, 53.478, 52.139],
               [0, 54.493, 52.139]])
weights = np.array([1.00, 1.00, 1.00, 1.20, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 2.00, 1.00, 1.00, 5.00, 3.00, 1.00,
         1.10, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.000, 1.00, 1.10,
         1.00, 3.00, 5.00, 1.00, 1.00, 2.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.20, 1.00, 1.00, 1.00])
knots = np.array([0, 0, 0, 0, 0.0083, 0.0150, 0.0361, 0.0855, 0.1293, 0.1509, 0.1931, 0.2273, 0.2435, 0.2561,
                      0.2692, 0.2889, 0.3170, 0.3316, 0.3482, 0.3553, 0.3649, 0.3837, 0.4005, 0.4269, 0.4510,
                      0.4660, 0.4891, 0.5000, 0.5109, 0.5340, 0.5489, 0.5731, 0.5994, 0.6163, 0.6351, 0.6447,
                      0.6518, 0.6683, 0.6830, 0.7111, 0.7307, 0.7439, 0.7565, 0.7729, 0.8069, 0.8491, 0.8707,
                      0.9145, 0.9639, 0.9850, 0.9917, 1, 1, 1, 1])
curve_points = nurbs_curve(control_points, weights, degree, knots)

# Plot the NURBS curve
plt.plot( curve_points[:, 1], curve_points[:, 2], label='NURBS Curve')
plt.scatter( control_points[:, 1], control_points[:, 2], c='red', label='Control Points')
plt.legend()
plt.show()
