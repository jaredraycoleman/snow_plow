from typing import Callable
import numpy as np
import pandas as pd
from sympy import *
from sympy.solvers import solve
import plotly.express as px


def find_pos_scalable(distance: Callable[[Symbol], Expr], 
                      accumulator: Callable[[Symbol], Expr],
                      num_agents: int) -> np.ndarray:
    l = Symbol('l')
    r = Symbol('r')
    x = Symbol('x')

    f_fun = distance(x)
    g_fun = accumulator(x)

    cap_fun = f_fun.subs(x, r)*integrate(g_fun,(x, l, r))
    solutions = solve(cap_fun - 1, r)

    pos = np.zeros(num_agents+1)
    for i in range(1, num_agents+1):
        for sol in solutions:
            p = N(sol.subs(l, pos[i-1]))
            try:
                if p >= pos[-1]:
                    print(f"Candidate pos for agent {i}: {p}")
                    pos[i] = p
            except TypeError:
                pass

    pos /= pos[-1]

    # CHECK
    capacities = np.array([
        N(cap_fun.subs([(l, start), (r, end)]))
        for start, end in zip(pos[:-1], pos[1:])
    ])
    print(capacities)

    return pos 

def find_pos(distance: Callable[[Symbol], Expr], 
             accumulator: Callable[[Symbol], Expr],
             num_agents: int,
             epsilon: float) -> np.ndarray:
    l = Symbol('l')
    r = Symbol('r')
    x = Symbol('x')

    f_fun = distance(x)
    g_fun = accumulator(x)
    cap_fun = f_fun.subs(x, r)*integrate(g_fun,(x, l, r))

    print(cap_fun)

    c_low, c_high = 0, N(cap_fun.subs([(l, 0), (r, 1)]))
    while True:
        c = (c_high + c_low) / 2
        print("cap", c)        
        solutions = solve(cap_fun - c, r, numerical=True)

        pos = np.zeros(num_agents+1)
        for i in range(1, num_agents+1):
            for sol in solutions:
                p = N(sol.subs(l, pos[i-1]))
                print(p)
                try:
                    if p >= pos[-1]:
                        pos[i] = p
                except TypeError:
                    pass
            print()
            
        print(c, c_low, c_high, pos[-1])
        if np.abs(pos[-1] - 1) <= epsilon:
            break
        elif pos[-1] > 1:
            c_high = c
        else:
            c_low = c

    # CHECK
    capacities = np.array([
        N(cap_fun.subs([(l, start), (r, end)]))
        for start, end in zip(pos[:-1], pos[1:])
    ])
    print(capacities)

    return pos

def main():
    distance = lambda x: x # 1 + (x*0)
    accumulator = lambda x: 1 - x**2 # 1/(x+1)**2

    xs: np.ndarray = np.linspace(0, 1, num=100)
    df = pd.DataFrame(dict(pos=xs, distance=distance(xs), snow=accumulator(xs)))
    df = df.melt(id_vars="pos", value_vars=["distance", "snow"], var_name="function", value_name="value")

    pos = find_pos(
        distance=distance,
        accumulator=accumulator, # 1/(x-(1/2))**2- 4,
        num_agents=2,
        epsilon=1e-10
    )
    print(pos)

    df = pd.concat([df, pd.DataFrame({"pos": pos, "function": "robot", "value": 1})])

    fig = px.scatter(
        df, x="pos", y="value", facet_row="function",
        template="simple_white"
    )
    
    fig.data[0].update(mode='lines')
    fig.data[1].update(mode='lines')
    fig.show()

if __name__ == "__main__":
    main()