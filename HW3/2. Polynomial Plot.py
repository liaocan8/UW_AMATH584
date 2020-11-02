import matplotlib.pyplot as plt
import numpy as np

def p1(x):
    return (x**9)-18*(x**8)+144*(x**7)-672*(x**6)+2016*(x**5)-4032*(x**4)+5376*(x**3)-4608*(x**2)+2304*x-512

def p2(x):
    return (x-2)**9

delta_x = 0.001
start_x = 1.920
end_x = 2.080
num_step = int((end_x - start_x)/delta_x)+1
x_scale = np.linspace(1.920,2.080,num=num_step)
p1_plot = [p1(i) for i in x_scale]
p2_plot = [p2(i) for i in x_scale]

plt.figure("Plot")
plt.plot(x_scale,p1_plot)
plt.plot(x_scale,p2_plot)
plt.legend(("Foiled","Unfoiled"))
plt.show()
