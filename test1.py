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


# Point to project
q = Point(1, 1)
# The scope of the parameter definition and parameter values
t0, tT = 0, 2 * np.pi
t = np.linspace(t0, tT, 101)

# Create the instance of parametric curve
circle = ParametricCurve(x, y, dxdt, dydt)
# Find the parameter t of projection point
t_ = q.projection_loop(1, circle)
# Calculate coorditanes of projecton point by parameter value
q_pr = Point(*circle.value(t_))
# Find the normal vector of projection
normal = np.vstack([q.coords, q_pr.coords]).T
# Find the tangent line in projection point
tline = circle.TangLine(t_)

# Test: if projection calculated good the scalar product is close to zero 
v1 = q_pr - q
v2 = Point(*tline.value(t_)) - Point(*tline.value(0))
print('Scalar product equal to: ', v1 * v2)

# Plot the objects
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(*circle.value(t), color='b', lw=0.6)
ax.plot(*tline.value(t), color='r', lw=0.6)
ax.plot(*normal, color='k', lw=0.6, linestyle='--')
ax.scatter(*circle.value(t_), s=20, color='#00FF00')
ax.scatter(*q.coords, s=20, color='k')
ax.scatter(*q_pr.coords, s=20, color='r')
plt.show()