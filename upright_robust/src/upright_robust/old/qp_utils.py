class QPUpdateMask:
    def __init__(
        self, P=True, q=True, A=True, b=True, G=True, h=True, lb=True, ub=True
    ):
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.G = G
        self.h = h
        self.lb = lb
        self.ub = ub

    def update(self, P, q, A=None, b=None, G=None, h=None, lb=None, ub=None):
        P = P if self.P else None
        q = q if self.q else None
        A = A if self.A else None
        b = b if self.b else None
        G = G if self.G else None
        h = h if self.h else None
        lb = lb if self.lb else None
        ub = ub if self.ub else None
        return P, q, A, b, G, h, lb, ub


