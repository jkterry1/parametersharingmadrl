import numpy as np
import matplotlib.pyplot as plt

# log((1-v)*.05/(1-v - g/n))/log(1- k/(n-1) +
# (k*g*(n-1)/(n*(n-1)))/((k/(n-1)+1)(1-v) + v/(n-1))) for v=0.1, n=20, g=.01, k=0 to 0.00001

plt.rcParams["font.family"] = "serif"
plt.figure(figsize=(4,3))

x = np.linspace(0, 0.00001, 40)


def func(val):
    c = 0.1
    I = 0.01
    e = 0.001
    n = 3

    y = np.log(((1-c)*e)/(1-c-(I/n)))/np.log(1 - (val/(n-1)) + val*I/(n*((1+(val/(n-1))) * (1 - c) + (c/(n-1)))))

    return y


y_arr = np.array([func(i) for i in x])

plt.plot(x, y_arr)
plt.xlabel("$K_{total}$")
plt.ylabel("Number of Steps")
plt.tight_layout()
plt.title("Convergence Rates Over K Values")
plt.xticks(ticks=[0,.000002,.000004,.000006,.000008,.00001],labels=['0','$2\\cdot10^{-6}$','$4\\cdot10^{-6}$','$6\\cdot10^{-6}$','$8\\cdot10^{-6}$','$10\\cdot10^{-6}$'])
plt.yticks(ticks=[0,1e7,2e7,3e7,4e7,5e7],labels=['0','$1\\cdot10^{7}$','$2\\cdot10^{7}$','$3\\cdot10^{7}$','$4\\cdot10^{7}$','$5\\cdot10^{7}$'])
plt.savefig("kplot_camera.png", bbox_inches='tight', pad_inches=.025)
