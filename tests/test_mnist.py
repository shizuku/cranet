from src import dpln
from src.dpln import nn, optim

model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.Linear(64, 32),
    nn.Linear(32, 10),
)

print(model)

model_p = list(model.parameters())
model_optm = optim.SGD(model.parameters(), 1)

inp = dpln.uniform((10, 784), 0, 1, True)

model_optm.zero_grad()
out = model(inp)
out.backward(dpln.ones((10, 10)))
model_optm.step()
model_optm.zero_grad()
print(out.shape)
