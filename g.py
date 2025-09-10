import matplotlib 
import numpy as np
import matplotlib.pyplot as plt

def pxy(x,y,m,c):
    x_min = min(x) - 1
    x_max = max(x) + 1
    x_line = np.linspace(x_min, x_max, 150)
    y_line = m * x_line + c
    plt.plot(x_line, y_line, color = 'red', label = f'y = {m}x + {c}')
    plt.scatter(x, y, color = 'blue', label='Data Points')
    plt.title("Simple x vs y Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid()
    plt.savefig('plot.png')
    print("Plot saved as plot.png")

