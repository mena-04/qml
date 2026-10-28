import torch
from src.residual import ResidualClassifier

def test_models():
    x=torch.rand(3,4,dtype=torch.float64)
    for branch,n in [('none',25),('classical',44),('quantum',46)]:
        m=ResidualClassifier(branch).double()
        assert sum(p.numel() for p in m.parameters())==n
        assert m(x).shape==(3,)
        if branch=='quantum':
            m(x).sum().backward()
            assert torch.isfinite(m.residual.quantum.weights.grad).all()
