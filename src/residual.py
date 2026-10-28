import torch
from torch import nn
import pennylane as qml

MODEL_TYPES = {'Classical MLP':'none', 'Classical Residual':'classical', 'Quantum Residual':'quantum'}

class ClassicalBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,1))
    def forward(self,x):
        return self.network(x).squeeze(-1)

class ClassicalResidualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4,3),nn.Tanh(),nn.Linear(3,1))
    def forward(self,x):
        return self.network(x).squeeze(-1)

class QuantumResidualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        dev = qml.device('default.qubit', wires=4)
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs,weights):
            for i in range(4):
                qml.RY(inputs[i],wires=i)
            for layer in range(2):
                for i in range(4):
                    qml.RY(weights[layer,i,0],wires=i)
                    qml.RZ(weights[layer,i,1],wires=i)
                for i in range(3):
                    qml.CNOT(wires=[i,i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        self.quantum = qml.qnn.TorchLayer(circuit, {'weights':(2,4,2)})
        self.output = nn.Linear(4,1)
    def forward(self,x):
        return self.output(torch.stack([self.quantum(sample) for sample in x])).squeeze(-1)

class ResidualClassifier(nn.Module):
    def __init__(self, branch_type='none', scale=0.5):
        super().__init__()
        self.base = ClassicalBase()
        self.scale = float(scale)
        if branch_type == 'none': self.residual = None
        elif branch_type == 'classical': self.residual = ClassicalResidualBranch()
        elif branch_type == 'quantum': self.residual = QuantumResidualBranch()
        else: raise ValueError(branch_type)
    def forward(self,x):
        base = self.base(x)
        return base if self.residual is None else base + self.scale * self.residual(x)
