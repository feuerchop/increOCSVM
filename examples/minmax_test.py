__author__ = 'LT'
"""
Example of solving Mini-Max Problem
max { (x0-15)^2+(x1-80)^2, (x1-15)^2 + (x2-8)^2, (x2-8)^2 + (x0-80)^2 } -> min
Currently nsmm is single OO solver available for MMP
It defines function F(x) = max_i {f[i](x)}
and solves NSP F(x) -> min using solver ralg.
It's very far from specialized solvers (like MATLAB fminimax),
but it's better than having nothing at all,
and allows using of nonsmooth and noisy funcs.
This solver is intended to be enhanced in future.
"""
from numpy import *
from openopt import *
from DerApproximator import *
f1 = lambda x: (x[0]-15)**2 + (x[1]-80)**2
f2 = lambda x: (x[1]-15)**2 + (x[2]-8)**2
f3 = lambda x: (x[2]-8)**2 + (x[0]-80)**2
f = [f1, f2, f3]

# you can define matrices as numpy array, matrix, Python lists or tuples

#box-bound constraints lb <= x <= ub
lb = [0]*3# i.e. [0,0,0]
ub = [15,  inf,  80]

# linear ineq constraints A*x <= b
A = mat('4 5 6; 80 8 15')
b = [100,  350]

# non-linear eq constraints Aeq*x = beq
Aeq = mat('15 8 80')
beq = 90

# non-lin ineq constraints c(x) <= 0
c1 = lambda x: x[0] + (x[1]/8) ** 2 - 15
c2 = lambda x: x[0] + (x[2]/80) ** 2 - 15
c = [c1, c2]
#or: c = lambda x: (x[0] + (x[1]/8) ** 2 - 15, x[0] + (x[2]/80) ** 2 - 15)

# non-lin eq constraints h(x) = 0
h = lambda x: x[0]+x[2]**2 - x[1]

x0 = [0, 1, 2]
p = MMP(f,  x0,  lb = lb,  ub = ub,   A=A,  b=b,   Aeq = Aeq,  beq = beq,  c=c,  h=h, xtol = 1e-8)
# optional, matplotlib is required:
#p.plot=1
r = p.solve('nsmm')
print 'MMP result:',  r.ff