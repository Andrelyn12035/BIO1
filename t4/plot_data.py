import matplotlib.pyplot as plt
import pandas as pd



X_RANGE = (0.1, 10.0)
Y_RANGE = (0.0, 1.0)

DIR = "./plots/xyz_data_points_2026-04-12_00-28-57.csv"

#read data from csv file
def read_csv(file_path):
    data_points = []
    with open(file_path, 'r') as f:
        next(f)  # Skip header
        bets_fitness = 0
        for line in f:
            x, y, fitness = map(float, line.strip().split(','))
            
            if X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]:
                if fitness < bets_fitness:
                    bets_fitness = fitness
                data_points.append((x, y, fitness))
    print(f"Best fitness: {bets_fitness}")
    return data_points


df = pd.read_csv(DIR, names=['X', 'Y', 'Fitness'], skiprows=1)
print(df.head())
#max and min of fitness
print(f"Max fitness: {df['Fitness'].max()}")
print(f"Min fitness: {df['Fitness'].min()}")



xyz_data_points = read_csv(DIR)


#plot and save the xyz_data_points to a csv file
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = [point[0] for point in xyz_data_points]
ys = [point[1] for point in xyz_data_points]
zs = [point[2] for point in xyz_data_points]
ax.scatter(xs, ys, zs, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')
#SHOW THE PLOT
plt.show()