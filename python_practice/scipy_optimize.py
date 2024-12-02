import numpy as np
from scipy.optimize import linprog,minimize

# linprog实例
c = np.array([-72,-64])
a_ub = np.array([[1,1],[12,8],[3,0]])
b_ub = np.array([50,480,100])
res = linprog(c=c,A_ub=a_ub,b_ub=b_ub)
print(res.x)
print(res.fun)

# minimize实例
c_fun = np.array([222.6, 183.3, 261.8, 169.5])

def object_fun(x:np.ndarray) -> float:
    y1 = np.min([x[0]+2*x[1]+4*x[3], (10*x[0]+4*x[1]+16*x[2]+5*x[3])/2])
    y2 = x[0]+2*x[1]+4*x[3]-y1
    y3 = 10*x[0]+4*x[1]+16*x[2]+5*x[3]-2*y1
    return -.1*y1 + .001*(np.sum(c_fun * x) + 157.1*y2 + 19.6*y3)

cons = [{'type':'ineq', 'fun':lambda x:14.4-1.5*x[0]-2*x[1]-x[2]-3*x[3]},
        {'type':'ineq', 'fun':lambda x:5 - np.sum(x[0:2])},
        {'type':'ineq', 'fun':lambda x:2-x[3]}]
init_x = np.ones(4)
res = minimize(object_fun, init_x, constraints=cons)
print(res.success)
print(res.x)
print(res.fun)
