import torch.optim as optim
import model
import numpy as np
import torch
from torch.autograd import Variable # torch 中 Variable 模块


learning_rate = 0.1
x = np.random.rand(3, 200, 176)
x = torch.from_numpy(x).to(torch.float32)
x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
mymodel = model.my_model()
out = mymodel(x)
# print(type(out))
print(out.shape)
heatmap = out.view(out.size(0), -1)
print(heatmap.shape)
creiteron = model.My_L2loss()
loss = creiteron(out, [1.0, 1.0])
print(loss)
optimizer = optim.Adam(mymodel.parameters(), lr=learning_rate, weight_decay=0.0005)
label = [50.0, 50.0]
loss = creiteron(out, label)
optimizer.zero_grad()
loss.backward()
optimizer.step()