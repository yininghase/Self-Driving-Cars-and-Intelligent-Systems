import torch
import matplotlib.pyplot as plt

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
scheduler_2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
s = torch.optim.lr_scheduler.ChainedScheduler([scheduler_2, scheduler_1])
lrs = []


for i in range(200):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    s.step()

plt.plot(lrs)
plt.show()