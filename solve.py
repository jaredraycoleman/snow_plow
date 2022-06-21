import logging
from typing import Callable, Optional
import numpy as np
import pandas as pd
from sympy import *
from sympy.solvers import solve
from sympy.calculus.singularities import is_decreasing
import plotly.express as px


# def find_pos_scalable(distance: Callable[[Symbol], Expr], 
#                       accumulator: Callable[[Symbol], Expr],
#                       num_agents: int) -> np.ndarray:
#     l = Symbol('l')
#     r = Symbol('r')
#     x = Symbol('x')

#     f_fun = distance(x)
#     g_fun = accumulator(x)

#     cap_fun = f_fun.subs(x, r)*integrate(g_fun,(x, l, r))
#     solutions = solve(cap_fun - 1, r)

#     pos = np.zeros(num_agents+1)
#     for i in range(1, num_agents+1):
#         for sol in solutions:
#             p = N(sol.subs(l, pos[i-1]))
#             try:
#                 if p >= pos[-1]:
#                     print(f"Candidate pos for agent {i}: {p}")
#                     pos[i] = p
#             except TypeError:
#                 pass

#     pos /= pos[-1]

#     # CHECK
#     capacities = np.array([
#         N(cap_fun.subs([(l, start), (r, end)]))
#         for start, end in zip(pos[:-1], pos[1:])
#     ])
#     print(capacities)

#     return pos 

def find_pos_endpoint(f_fun: Expr, 
                      G_fun: Expr, 
                      G_inv: Expr, 
                      x: Symbol, 
                      y: Symbol,
                      num_agents: int,
                      epsilon: float) -> np.ndarray:
    c_low, c_high = 0, N(f_fun.subs(x, 1)*G_inv.subs(y, 1))
    iteration = 0
    while c_high - c_low > 2*epsilon:
        iteration += 1
        logging.debug(f'Interval: {(c_low, c_high)}')
        c = (c_low + c_high) / 2
        logging.debug(f'Iteration {iteration}: capacity {c}')
        pos = np.ones(num_agents)
        for i in range(num_agents - 1)[::-1]:
            pos[i] = N(G_inv.subs(y, G_fun.subs(x, pos[i+1]) - c/f_fun.subs(x, pos[i+1])))
            if pos[i] <= 0 or N(f_fun.subs(x, pos[i]) * G_fun.subs(x, pos[i])) <= c:
                logging.debug(f'Coverable with {num_agents-i}/{num_agents} agent(s). Capacity {c} is feasible')
                c_high = c
                break
            logging.debug(f'Agent {i}: {pos[i]}')
        if c < c_high:
            c_low = c
    return pos

def find_pos_midpoint(f_fun: Expr, 
                      G_fun: Expr, 
                      G_inv: Expr, 
                      x: Symbol, 
                      y: Symbol,
                      num_agents: int,
                      epsilon: float,
                      destination: float) -> np.ndarray:
    c_low, c_high = 0, N(f_fun.subs(x, 1)*G_inv.subs(y, 1))
    iteration = 0

    def cap(x1, x2) -> float:
        if x1 > x2:
            x1, x2 = x2, x1

        if x2 <= destination: # both on left
            d1 = destination - x1 
            d2 = destination - x2
            return N(f_fun.subs(x, d1) * (G_fun.subs(x, d1) - G_fun.subs(x, d2)))
        elif x1 >= destination: # both on right
            d1 = x1 - destination
            d2 = x2 - destination
            return N(f_fun.subs(x, d2) * (G_fun.subs(x, d2) - G_fun.subs(x, d1)))
        else:
            d1 = destination - x1
            d2 = x2 - destination
            return N(f_fun.subs(x, d1+d2) * max(G_fun.subs(x, d1), G_fun.subs(x, d2)))
           
    def h(x_next: float) -> float:
        if x_next <= destination:
            d = destination - x_next
            return destination - N(G_inv.subs(y, G_fun.subs(x, d) - c/f_fun.subs(x, d)))
        else:
            d = x_next - destination
            return destination + N(G_inv.subs(y, G_fun.subs(x, d) - c/f_fun.subs(x, d)))

    while c_high - c_low > 2*epsilon:
        iteration += 1
        logging.debug(f'Interval: {(c_low, c_high)}')
        c = (c_low + c_high) / 2
        logging.debug(f'Iteration {iteration}: capacity {c}')
        
        left = [0]
        right = [1]
        has_middle = False
        covered = False
        if cap(destination, left[-1]) <= c and cap(destination, right[-1]) <= c:
            logging.debug(f"c={c:0.4f} is feasible with just two robots")
            c_high = c
            continue

        for i in range(num_agents - 1)[::-1]:
            logging.debug(f'Assigning Agent {i}')
            if destination-left[-1] <= right[-1]-destination:
                right.append(h(right[-1]))
            else:
                left.append(h(left[-1]))
            logging.debug(f'Left: {left}')
            logging.debug(f'Right: {right}')
            
            if cap(destination, right[-1]) <= c and cap(destination, left[-1]) <= c:
                covered = True
                break
        
        if not covered:
            logging.debug('even with extra robot, c is infeasible')
            c_low = c
        elif len(left) + len(right) <= num_agents:
            logging.debug('did not need extra robot - c is feasible')
            c_high = c
        else:
            has_middle = True            
            c1 = cap(left[-1], right[-1])
            logging.debug(f"final interval: ({left[-1], right[-1]}), cap={c1}")
            if c1 < c:
                c_high = c
            else:
                c_low = c

    pos = np.array(left + right[::-1])
    logging.debug(f'Middle Robot: {has_middle}')
    if has_middle:
        for i in range(1, len(pos)):
            logging.debug(f'Agent {i} Capacity: {cap(pos[i-1], pos[i]):0.4f}')
    else:
        for i in range(len(pos)):
            if pos[i] < destination:
                other = destination if pos[i+1] > destination else pos[i+1]
            else:
                other = destination if pos[i-1] < destination else pos[i-1]
                
            logging.debug(f'Agent {i+1} Capacity: {cap(other, pos[i]):0.4f}')

    return pos 

def find_pos(distance: Callable[[Symbol], Expr], 
             accumulator: Callable[[Symbol], Expr],
             num_agents: int,
             epsilon: float,
             destination: float = 0.0) -> np.ndarray:
    y = Symbol('y')
    x = Symbol('x')

    if num_agents < 2 or not isinstance(num_agents, int):
        raise ValueError(f'num_agents must be an integer at least 2')

    if destination < 0 or destination > 1:
        raise ValueError(f'destination must be between 0 and 1')

    f_fun = distance(x)
    if is_decreasing(f_fun, interval=Interval(0, 1)):
        raise ValueError(f'distance function must be non-dereasing on [0, 1]')

    if N(f_fun.subs(x, 0)) < 0 or N(f_fun.subs(x, 1)) <= 0:
        raise ValueError(f'distance function must be positive on (0, 1]')

    g_fun = accumulator(x)
    G_fun = integrate(g_fun, x)

    if not is_increasing(G_fun, Interval(0, 1)):
        raise ValueError(f'resource function must be positive on [0, 1]')

    G_inv: Optional[Expr] = None
    logging.debug(f'G_inv = {G_inv}')
    for _sol in solve(G_fun - y, x):
        try:
            if N(_sol.subs(y, 1)) > 0:
                G_inv = _sol
                break
        except TypeError:
            pass
    
    if G_inv is None:
        raise ValueError(f'Could not find real, positive inverse for {G_fun}')

    if destination == 0.0:
        return find_pos_endpoint(
            f_fun=f_fun, G_fun=G_fun, G_inv=G_inv,
            x=x, y=y, num_agents=num_agents, epsilon=epsilon
        )
    else:
        pos = find_pos_midpoint(
            f_fun=f_fun, G_fun=G_fun, G_inv=G_inv,
            x=x, y=y, num_agents=num_agents, epsilon=epsilon,
            destination=destination
        )
        return pos



def main():
    logging.getLogger().setLevel(logging._nameToLevel['DEBUG'])
    distance = lambda x: x**2
    accumulator = lambda x: x+1
    destination = 0.5

    xs: np.ndarray = np.linspace(0, 1, num=100)
    df = pd.DataFrame(dict(pos=xs, distance=distance(xs), snow=accumulator(xs)))
    df = df.melt(id_vars="pos", value_vars=["distance", "snow"], var_name="function", value_name="value")

    pos = find_pos(
        distance=distance,
        accumulator=accumulator, # 1/(x-(1/2))**2- 4,
        num_agents=6,
        epsilon=1e-10,
        destination=destination
    )
    print(pos)

    # df = pd.concat([df, pd.DataFrame({"pos": pos, "function": "robot", "value": 1})])

    # fig = px.scatter(
    #     df, x="pos", y="value", facet_row="function",
    #     template="simple_white"
    # )
    
    # fig.data[0].update(mode='lines')
    # fig.data[1].update(mode='lines')
    # fig.show()

if __name__ == "__main__":
    main()