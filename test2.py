import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from basics import Point, Vector, ParametricCurve

# Define the parametric curve by equations
# and its derivatives

def x(t, x0=1):
    return 0.5 * x0 * t**2

def y(t, y0=1):
    return y0 * t

def dxdt(t, x0=1):
    return x0 * t

def dydt(t, y0=1):
    return y0 * t**0


# Point to project
q = Point(2, 5)
# The scope of the parameter definition and parameter values
t0, tT = 0, 2 * np.pi
t = np.linspace(t0, tT, 101)

# Create the instance of parametric curve
parabola = ParametricCurve(x, y, dxdt, dydt)
# Find the parameter t of projection point
t_ = q.projection_loop(3, parabola)
# Calculate coorditanes of projecton point by parameter value
q_pr = Point(*parabola.value(t_))
# Find the normal vector of projection
normal = np.vstack([q.coords, q_pr.coords]).T
# Find the tangent line in projection point
tline = parabola.TangLine(t_)

# Test: if projection calculated good the scalar product is close to zero 
v1 = q_pr - q
v2 = Point(*tline.value(t_)) - Point(*tline.value(0))
print('Scalar product equal to: ', v1 * v2)
cos = v1 * v2 / (np.linalg.norm(v1.coords) * np.linalg.norm(v2.coords))
print('Angle is: ', 180 * np.arccos(cos) / np.pi)

# Plot the objects
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(*parabola.value(t), color='b', lw=0.6)
ax.plot(*tline.value(t), color='r', lw=0.6)
ax.plot(*normal, color='k', lw=0.6, linestyle='--')
ax.scatter(*parabola.value(t_), s=20, color='#00FF00')
ax.scatter(*q.coords, s=20, color='k')
ax.scatter(*q_pr.coords, s=20, color='r')
plt.show()