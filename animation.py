import matplotlib.pyplot as plt
import matplotlib.animation as animation

positions = []

# Read the particle positions from the file
with open('posiciones_particulas.txt', 'r') as file:
    lines = file.readlines()
    temp_positions = []
    for line in lines:
        if line.strip() == '':  # Check for an empty line indicating a new time step
            positions.append(temp_positions)
            temp_positions = []
        else:
            temp_positions.append(list(map(float, line.split())))

# Prepare the figure
fig, ax = plt.subplots()
ax.set_xlim(-100, 100)  # Set x-axis limits as needed
ax.set_ylim(-100, 100)  # Set y-axis limits as needed
particles, = ax.plot([], [], 'bo')  # Plot the particles as blue dots

# Function to update the plot for each frame of the animation
def update(frame):
    x_data = [pos[0] for pos in positions[frame]]
    y_data = [pos[1] for pos in positions[frame]]
    particles.set_data(x_data, y_data)
    return particles,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=1, blit=True)

# Show the animation
plt.show()
