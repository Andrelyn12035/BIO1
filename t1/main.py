
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource

import matplotlib.animation as animation
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(15, 9))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-5, 5, .12)
Y = np.arange(-5, 5, .12)
X, Y = np.meshgrid(X, Y)

Z = 0.5 * (X**4 -16 * X**2 + 5*X + Y**4 -16 * Y**2 + 5*Y )
ls = LightSource(70, 55)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(Z, cmap=cm.turbo, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False)

ax.set_zlim(-50.01, 250.01)
ax.set_xlim(-5.01, 5.01)
ax.set_ylim(-5.01, 5.01)



#plotting plt contour semitranslucent 
#show axes on plot
#set height of the plot to 0.5
sub = fig.add_subplot(1, 2, 2)
contour = sub.contourf(X, Y, Z, levels=50, cmap=cm.turbo, alpha=0.9)
sub.set_xlim(-5.01, 5.01)
sub.set_ylim(-5.01, 5.01)




def get_next_step(current_x, current_y, step_size=0.01):
    # Example gradient descent step (simplified)
    # In a real application, this would compute the gradient of the function
    # and take a step in the negative gradient direction.
    x_gradient, y_gradient = get_gradient(current_x, current_y)
    next_x = current_x - step_size * x_gradient
    next_y = current_y - step_size * y_gradient
    return next_x, next_y

def get_gradient(current_x, current_y):
    x_gradient = 2*current_x**3 - 16*current_x +2.5
    y_gradient = 2*current_y**3 - 16*current_y +2.5
    return x_gradient, y_gradient

x_steps = [-4]
y_steps = [4]
z_steps = [0.5 * (x_steps[0]**4 -16 * x_steps[0]**2 + 5*x_steps[0] + y_steps[0]**4 -16 * y_steps[0]**2 + 5*y_steps[0] )]
scatter, = ax.plot(x_steps, y_steps, z_steps,  linestyle="", marker="o", color="red", markersize=5)

scatter_2d, = sub.plot(x_steps, y_steps, linestyle="", marker="o", color="red", markersize=5)

for i in range(10000000):
    current_x, current_y = x_steps[-1], y_steps[-1]
    next_x, next_y = get_next_step(current_x, current_y)
    x_steps.append(next_x)
    y_steps.append(next_y)
    z_steps.append((0.5 * (next_x**4 -16 * next_x**2 + 5*next_x + next_y**4 -16 * next_y**2 + 5*next_y ))+ 1)
    print(f"Step {i+1}: Moving to ({next_x:.2f}, {next_y:.2f})")
    limit = 0.0001
    #euclidean distance between current point and next point
    tolerance = np.sqrt((next_x - current_x)**2 + (next_y - current_y)**2)
    if tolerance < limit:
        print(f"Convergence reached at step {i+1}.")
        break


def update(frame):
    # for each frame, update the data stored on each artist.
    x = x_steps[:frame]
    y = y_steps[:frame]
    z = z_steps[:frame] 
    # update the scatter plot:
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
    #                     linewidth=0, antialiased=False)
    scatter.set_data(x, y)
    scatter.set_3d_properties(z)
    scatter_2d.set_data(x, y)
    return scatter, scatter_2d


ani = animation.FuncAnimation(fig, update, frames=len(x_steps), interval=100, blit=True)
plt.show()
