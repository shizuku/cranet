from src import dpln
from src.dpln import nn, optim

encoder = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.Linear(64, 32),
    nn.Linear(32, 10),
)

decoder = nn.Sequential(
    nn.Linear(10, 32),
    nn.Linear(32, 64),
    nn.Linear(64, 28 * 28),
)

auto_encoder = nn.Sequential(
    encoder,
    decoder,
)

ae_p = list(auto_encoder.named_parameters())
ae_optm = optim.SGD(auto_encoder.parameters(), 0.001)

inp = dpln.uniform((10, 784), 0, 1, True)

ae_optm.zero_grad()
out = auto_encoder(inp)
out.backward(dpln.ones_like(out))
ae_optm.step()
ae_optm.zero_grad()
print(out.shape)
