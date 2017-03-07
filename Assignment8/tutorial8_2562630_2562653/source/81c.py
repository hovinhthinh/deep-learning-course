from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

f = [3, 1, 8, 6, 3, 9, 5, 1]
s1 = [1.5, 2, 4.5, 7, 4.5, 6, 7, 3, 0.5]
s2 = [0.75, 1.75, 3.25, 5.75, 5.75, 5.25, 6.5, 5, 1.75, 0.25]
a = np.arange(len(f))
plt.plot(a, f, 'b-', label='f', linewidth=2)
b = np.arange(len(s1))
plt.plot(b, s1, 'r-', label='s1', linewidth=2)
c = np.arange(len(s2))
plt.plot(c, s2, 'g-', label='s2', linewidth=2)
plt.legend()
plt.xlabel('time')
plt.show()
