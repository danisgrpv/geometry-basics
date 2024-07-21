import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from basics import Point, Vector, ParametricCurve

# Define the parametric curve by equations
# and its derivatives

def x(t, r=1):
    return r * np.cos(t)

def y(t, r=1):
    return r * np.sin(t)

def dxdt(t, r=1):
    return -r * np.sin(t)

def dydt(t, r=1):
    return r * np.cos(t)


# Create the instance of parametric curve
circle = ParametricCurve(x, y, dxdt, dydt)
# Parameter value for point of tangency
ttang = 1
# Tangent line
tline = circle.TangLine(ttang)

# The scope of the parameter definition
# and parameter values
t0, tT = 0, 2 * np.pi
t = np.linspace(t0, tT, 101)

# Point to project
q = Point(1, 2)
# Point projection to tangent
q_pr = q.projection(tline, 1, 2)
# Projection vector
normal = np.vstack([q.coords, q_pr.coords]).T

# If projection calculated good the scalar product is zero 
v1 = q - q_pr
v2 = Point(*tline.value(t0)) - Point(*tline.value(0))
print('Scalar product equal to: ', v1 * v2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(*circle.value(t), color='b', lw=0.6)
ax.plot(*tline.value(t), color='r', lw=0.6)
ax.plot(*normal, color='k', lw=0.6, linestyle='--')
ax.scatter(*circle.value(ttang), s=20, color='#00FF00')
ax.scatter(*q.coords, s=20, color='k')
ax.scatter(*q_pr.coords, s=20, color='k')
plt.show()