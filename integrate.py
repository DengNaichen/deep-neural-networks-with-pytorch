#%%
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

x = np.linspace(0.1, 2 * np.pi, 100000)
f = np.square(np.sin(x))
plt.plot(x,f)
# plt.show()


# %%
## check first term
print ("first term")
y1 = f * np.log (3.5 * f)
I1 = integrate.trapz(y1, x)
print ("integration is: ", I1)
avg1 = np.mean (y1) * 2 * np.pi
print ("average is: ", avg1)
## 基本一致, 可是要乘2pi
#%%
print ("second term")
y2 = 

#%%
# check first term
y3 = f
print ("third term")
I3 = integrate.trapz(y3, x)
print ("integration is: ", I3)

Avg3 = np.mean (y3) * 2 * np.pi
print (Avg3)