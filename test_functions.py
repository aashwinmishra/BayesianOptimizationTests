import numpy as np
import torch
import math


class Forrester:
    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (isinstance(d, int) and (not d < 0)), "The dimension d must be None or a positive integer"
        return d == 1

    def __init__(self, d=1):
        self.d = d
        self.input_domain = torch.tensor([[0.0],[1.0]])

    def get_params(self):
        return {}

    def get_global_minimum(self):
        x = np.array([0.757249])
        return x, self(x)

    def __call__(self, points):
        result = []
        for x in points:
            result.append(np.square(6.0*x[0] -2.0)*np.sin(12.0*x[0]-4.0))
        return torch.tensor(result).unsqueeze(-1)


class Ackley:
    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (isinstance(d, int) and (not d < 0)), "The dimension d must be None or a positive integer"
        return (d is None) or (d  > 0)

    def __init__(self, d=2, a=20.0, b=0.2, c=2*np.pi):
        self.d = d
        self.input_domain = torch.tensor([[-32.768]*d, [32.768]*d])#np.array([[-32.768, 32.768] for _ in range(d)])
        self.a = a
        self.b = b
        self.c = c

    def get_params(self):
        return {'a': self.a, 'b': self.b, 'c': self.c}

    def get_global_minimum(self, d):
        x = torch.zeros((1,d))
        return x, self(x)

    def __call__(self, points):
        assert points.shape[1]==self.d
        num = points.shape[0]
        result = []
        for i in range(num):
            x = points[i].clone().detach()
            t1 = -self.a*torch.exp(-self.b*torch.sqrt(torch.mean(torch.square(x))))
            t2 = -torch.exp(torch.mean(torch.cos(self.c*x))) + self.a + torch.exp(torch.tensor(1.0))
            result.append(t1+t2)
        return torch.tensor(result).unsqueeze(-1)


class Branin:
    def __init__(self, d=2, a=1.0, b=5.1/(4*math.pi**2), c=5/math.pi, r=6.0, t=1/(8*math.pi), s=10.0):
        self.d = d
        self.input_domain = torch.tensor([[-5.0, 0.0], [10, 15]])  #[[l1,l2,l3...], [u1,u2,u3...]]
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.t = t
        self.s = s

    def get_params(self):
        return {'a': self.a, 'b': self.b, 'c': self.c, 'r': self.r, 's': self.s, 't': self.t}

    def get_global_minimum(self):
        x = torch.tensor([[-math.pi, 12.275],[math.pi, 2.275],[9.42478, 2.475]])
        return x, self(x)

    def __call__(self, points):
        assert points.shape[1] == self.d
        num = points.shape[0]
        result = []
        for i in range(num):
            x = points[i].clone().detach()
            res = self.a*(x[1] - self.b * x[0]**2 + self.c*x[0] - self.r)**2 + self.s*(1-self.t)*torch.cos(x[0]) +self.s
            result.append(res)
        return torch.tensor(result).unsqueeze(-1)


class BraninModified:
    def __init__(self, d=2, a=1.0, b=5.1/(4*math.pi**2), c=5/math.pi, r=6.0, t=1/(8*math.pi), s=10.0):
        self.d = d
        self.input_domain = torch.tensor([[-5.0, 0.0], [10, 15]])  #[[l1,l2,l3...], [u1,u2,u3...]]
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.t = t
        self.s = s

    def get_params(self):
        return {'a': self.a, 'b': self.b, 'c': self.c, 'r': self.r, 's': self.s, 't': self.t}

    def get_global_minimum(self):
        x = torch.tensor([[-math.pi, 12.275],[math.pi, 2.275],[9.42478, 2.475]])
        return x, self(x)

    def __call__(self, points):
        assert points.shape[1] == self.d
        num = points.shape[0]
        result = []
        for i in range(num):
            x = points[i].clone().detach()
            res = self.a*(x[1] - self.b * x[0]**2 + self.c*x[0] - self.r)**2 + self.s*(1-self.t)*torch.cos(x[0]) + self.s + 5.0*x[0]
            result.append(res)
        return torch.tensor(result).unsqueeze(-1)


class AlpineN1:
    def __init__(self, d=2):
        self.d = d
        self.input_domain = torch.tensor([[0.0]*d, [10.0]*d])

    def get_params(self):
        return {}

    def get_global_minimum(self, d):
        x = torch.tensor([[0.0]*d])
        return x, self(x)

    def __call__(self, points):
        assert points.shape[1] == self.d
        num = points.shape[0]
        result = []
        for i in range(num):
            x = points[i].clone().detach()
            res = torch.sum(torch.abs(x*torch.sin(x) + 0.1*x))
            result.append(res)
        return torch.tensor(result).unsqueeze(-1)


class AlpineN2:
    def __init__(self, d=2):
        self.d = d
        self.input_domain = torch.tensor([[0]*d, [10]*d])

    def get_params(self):
        return {}

    def get_global_minimum(self, d):
        x = torch.tensor([[7.917]*d])
        return x, self(x)

    def __call__(self, points):
        assert points.shape[1] == self.d
        num = points.shape[0]
        result = []
        for i in range(num):
            x = points[i].clone().detach()
            res = -1.0*torch.sum(torch.sqrt(x)*torch.sin(x))
            result.append(res)
        return torch.tensor(result).unsqueeze(-1)


class Bird:
    def __init__(self, d=2):
        self.d = d
        self.input_domain = torch.tensor([[-2*math.pi] * d, [2*math.pi] * d])

    def get_params(self):
        return {}

    def get_global_minimum(self, d):
        x = torch.tensor([[4.70104,3.15294], [-1.58214,-3.13024]])
        return x, self(x)

    def __call__(self, points):
        assert points.shape[1] == self.d
        num = points.shape[0]
        result = []
        for i in range(num):
            x = points[i].clone().detach()
            res = torch.sin(x[0])*torch.exp((1-torch.cos(x[1]))**2) + torch.cos(x[1])*torch.exp((1-torch.sin(x[0]))**2) + (x[0] - x[1])**2
            result.append(res)
        return torch.tensor(result).unsqueeze(-1)



