import torch

from src.rl.architecture import MultilayerPerceptron


def concat_experiment1():
    u = torch.tensor([0, 1])
    u = u.unsqueeze(0)
    A = torch.tensor([[1, 2], [3, 4]])
    u_exp = u.expand(A.shape)
    print(u_exp)
    print(A)

    scalar = torch.Tensor([[-1], [0]])

    X = torch.cat((u_exp, A), dim=1)
    print(X)

    Y = torch.cat((u_exp, A, scalar), dim=1)
    print(Y)


def concat_experiment2():
    A = torch.tensor([[1, 2], [3, 4]])
    a_sum = A.sum(dim=0)
    print(A)
    print(a_sum)
    b = torch.Tensor([1, -1])

    print(torch.cat((a_sum, b)))


def softmax_experiment():
    """
    The purpose of this code is to test stacking vectors [u | v | w],
    applying an MLP to them to produce a score for each vector [f(u) f(v) f(w)]
    and then sampling indices according to these scores using softmax.
    """
    u = torch.Tensor([1, 2])
    v = torch.Tensor([-1, -2])
    w = torch.Tensor([-3.5, 1.75])
    stack = torch.stack([u, v, w])

    mlp = MultilayerPerceptron([2, 64, 1])

    res = mlp(stack)
    print(res)

    p = torch.softmax(res, dim=0)
    print(p)
    print(p.sum())

    samples = torch.multinomial(p.T, 10, replacement=True)
    print(samples)


if __name__ == '__main__':
    softmax_experiment()
