import sympy as sym
import IPython


def skew3(x):
    return sym.Matrix([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def skew6(x):
    a = x[:3]
    b = x[3:]
    return sym.Matrix([[skew3(b), sym.zeros(3, 3)], [skew3(a), skew3(b)]])


def skew6_matrices():
    Ss = []
    for i in range(6):
        E = sym.zeros(6, 1)
        E[i] = 1
        Ss.append(skew6(E))
    return Ss


def lift3(x):
    # fmt: off
    return sym.Matrix([
        [x[0], x[1], x[2], 0, 0, 0],
        [0, x[0], 0, x[1], x[2], 0],
        [0, 0, x[0], 0, x[1], x[2]]])
    # fmt: on

def ALB(A, B):
    return skew6(A) * lift6(B)


def VLV(V):
    return ALB(V, V)


def lift6(x):
    a = sym.Matrix(x[:3])
    b = sym.Matrix(x[3:])
    return sym.Matrix([
        [a, skew3(b), sym.zeros(3, 6)],
        [sym.zeros(3, 1), -skew3(a), lift3(b)]])


def lift6_matrices():
    Ls = []
    for i in range(6):
        E = sym.zeros(6, 1)
        E[i] = 1
        Ls.append(lift6(E))
    return Ls


v = sym.Matrix(sym.symbols("vx,vy,vz"))
ω = sym.Matrix(sym.symbols("ωx,ωy,ωz"))
V = sym.Matrix.vstack(v, ω)

VVT = V * V.T
z = VVT.vec()
zh = VVT.vech()

Yv = VLV(V)
yv = Yv.vec()

Ss = skew6_matrices()
Ls = lift6_matrices()
Y_test = sym.zeros(6, 10)
Y_test2 = sym.zeros(6, 10)

As = []

for i in range(6):
    for j in range(6):
        As.append(Ss[i] * Ls[j])
        # Y_test += Ss[i] * Ls[j] * VVT[i, j]
        # Y_test += Ss[i] * Ls[j] * selector_vector(i, j, 6, 6).dot(z)

Ahs = []
for i in range(6):
    for j in range(i, 6):
        A = Ss[i] * Ls[j]
        if i != j:
            A += Ss[j] * Ls[i]
        Ahs.append(A)

for i in range(36):
    Y_test += As[i] * z[i]

for i in range(len(zh)):
    Y_test2 += Ahs[i] * zh[i]

nz_indices = []
for i in range(len(zh)):
    if Ahs[i] != sym.zeros(6, 10):
        nz_indices.append(i)
    print(f"Ahs[{i}] == 0 = {Ahs[i] == sym.zeros(6, 10)}")

# test vector
f = sym.Matrix([1, 2, 3, 4, 5, 6])

# matrix with rows of f.T * A[i]
# this is the linear representation required for the optimization problem
D = sym.Matrix([f.transpose() * A for A in As])
Dh = sym.Matrix([f.transpose() * A for A in Ahs])

assert f.transpose() * Yv == z.transpose() * D
assert f.transpose() * Yv == zh.transpose() * Dh

IPython.embed()
