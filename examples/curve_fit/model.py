import cranet
from cranet import nn


from typing import(
    Iterable,
)


class FNN(nn.Module):
    def __init__(self, features: Iterable[int]):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(features[0], features[1]),
            nn.Tanh(),
            nn.Linear(features[1], features[2]),
        )

    def forward(self, x):
        return self.layer(x)
    
    def fit(self, curvedata, epochs, lr=1e-2, opt_func=cranet.optim.SGD, loss_func=cranet.nn.loss.MSELoss):
        history = []
        optimizer = opt_func(self.parameters(), lr=lr)
        loss_func = loss_func()
        for epoch in range(epochs):
            # Training Phase
            self.train()
            train_losses = []
            for x, y in curvedata:
                pred = self(x)
                loss = loss_func(pred, y)
                train_losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            result = cranet.stack(train_losses).mean().numpy()
            if epoch % 10 == 0:
                print(f"{epoch}/{epochs} loss:{result}")
            history.append(result)
            
        return history