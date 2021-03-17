import numpy as np
import torch
a = np.ones((4,4))

print(a)

b= torch.from_numpy(a)

print(b)

b.mul_(2)

print(b)

print(a)
