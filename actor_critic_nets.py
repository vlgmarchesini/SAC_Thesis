import torch.nn as nn
import torch
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters

# Regular model that predicts 20 steps ahead
class BoatModelSigmoid(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20,xhist=4,n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))


class BoatModelRELUSigmoid(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20,xhist=4,n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))


class BoatModelRELUSigmoidBias(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20,xhist=4,n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))


class BoatModelRELUSigmoidBiasMultiple(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20,xhist=4,n_cmd=2):
        super().__init__()

        self.xhist = xhist
        self.n_state = n_state
        self.steps_ahead = steps_ahead
        self.n_cmd = n_cmd

        output_dim = n_state
        input_dim = n_state*xhist + xhist*n_cmd

        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.n_state))


class BoatModelTanhBias(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20, xhist=4, n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, output_dim, bias=True),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))


class HydrofoilTanhBias(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20, xhist=4, n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))
        
class HydrofoilTanhBiasEx(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20, xhist=4, n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))

class BoatModelRBFNN(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=256, n_state=7, steps_ahead=20, xhist=4, n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead*n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, input_dim, bias=True),
            RBFNN(input_dim,input_dim),
            nn.Linear(input_dim, input_dim, bias=True),
            RBFNN(input_dim,input_dim),
            nn.Linear(input_dim, output_dim, bias=True),
            #RBFNN(input_dim,input_dim),
            #nn.Linear(input_dim, hidden_dim, bias=True),
            #RBFNN(hidden_dim, hidden_dim),
            #nn.Linear(hidden_dim, hidden_dim, bias=True),
            #RBFNN(hidden_dim, hidden_dim),
            #nn.Linear(hidden_dim, output_dim, bias=True),
            )

class BoatModelRBFNNShort(nn.Module):
    def __init__(self, input_shape=74, hidden_dim=32, n_state=7, steps_ahead=20, xhist=4, n_cmd=2):
        super().__init__()

        self.n_state = n_state
        self.steps_ahead = steps_ahead
        output_dim = steps_ahead * n_state

        input_dim = torch.prod(torch.tensor(input_shape))
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(input_dim, hidden_dim, bias=True),
            RBFNN(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x):
        y_hat = self.model(x)
        return torch.reshape(y_hat, (x.size(0), self.steps_ahead, self.n_state))


model_dic = {
    'BoatModelSigmoid': BoatModelSigmoid,
    'BoatModelRELUSigmoid': BoatModelRELUSigmoid,
    'BoatModelRELUSigmoidBias': BoatModelRELUSigmoidBias,
    'BoatModelTanhBias': BoatModelTanhBias,
    'BoatModelRELUSigmoidBiasMultiple': BoatModelRELUSigmoidBiasMultiple,
    'BoatModelRBFNN': BoatModelRBFNN,
    'BoatModelRBFNNShort': BoatModelRBFNNShort,
    'HydrofoilTanhBias': HydrofoilTanhBias,
    'HydrofoilTanhBiasEx': HydrofoilTanhBiasEx,
}
