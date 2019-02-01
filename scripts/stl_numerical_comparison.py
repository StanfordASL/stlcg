# import matplotlib.pyplot as plt
# from importlib import reload
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join( os.pardir, os.pardir)))
import stl.stl_optimizer as opt
import stl.stl as stl

import numpy as np
from collections import OrderedDict
import sympy as sym
import copy
import time
from deap import base, creator, tools

import pandas as pd


a, b = np.pi*2, 0.25
t = np.arange(0, a, b)
x_values = np.sin(t - 0.5)
y_values = np.sin(0.75*t + 0.7)
z_values = (np.cos(t + 0.3)-0.9)
xyz_values = [[x, y, z] for (x, y, z) in zip(x_values, y_values, z_values)]
xyz_trace = stl.Trace.from_signals(xyz_values, var_name=[sym.symbols("x"), sym.symbols("y"), sym.symbols("z")])


def loss_fn(formula, val, tr):
    for (k,v) in val.items():
        formula = formula.subs(k, v)
    return formula.robustness(tr)**2


α, β, x, y, z = sym.symbols("α β x y z") 
I1 = stl.Interval(2, 5)
formula_1 = stl.And(stl.Always(stl.Comparison("<", x, α)) , stl.Always(stl.Comparison("<", y, β)))
formula_2 = stl.Or(stl.And(stl.Always(stl.Comparison("<", x, α)), stl.Always(stl.Comparison("<", y, β))), 
                   stl.Always(stl.Comparison("<", z, α)))

A = stl.Comparison(">", x, α)
B = stl.Comparison("<", y, β)
AA = stl.And(A, B)
lhs = stl.Eventually(stl.Always(AA, interval=I1))
rhs = stl.Always(stl.Comparison("<", z, α))
formula_3 = stl.And(lhs, rhs)


A = stl.Comparison(">", x, α + β)
B = stl.Comparison("<", y, α**2 - β)
AA = stl.And(A, B)
lhs = stl.Eventually(stl.Always(AA, interval=I1))
rhs = stl.Always(stl.Comparison("<", z, α+2*β**2))
formula_4 = stl.And(lhs, rhs)

def select_formula(n):
    if n == 1: return formula_1
    elif n == 2: return formula_2
    elif n == 3: return formula_3
    elif n == 4: return formula_4
    else: return None
    
    
for i in range(1,5):
    print(select_formula(i))

def gradient_descent_trials(n, xyz_trace, N=100, num_steps=1000):
    formula = select_formula(n)
    lr = 0.01
    gd = opt.STLGradientDescent(lr, verbose=False)
    num_iterations = []
    losses = []
    t0 = time.time()
    for run in range(N):
        print("RUN NUM: ", run)
        valuations = OrderedDict({α:2*np.random.rand()-1, β:2*np.random.rand()-1})
        for i in range(num_steps):
            if i % 100 == 0:
                print("num_steps = ", i)
            valuations, grad = gd.step(valuations, formula, loss_fn, xyz_trace)
            loss = loss_fn(formula, valuations, xyz_trace)
            if grad[α]**2 + grad[β]**2 < 1E-6 or loss < 1E-6:
                break
        num_iterations.append(i+1)
        losses.append(loss)
    t1 = time.time()
    total = t1-t0

    gd = {"total_time": total, "num_iterations": num_iterations, "losses": losses}
    np.save("gd_"+str(n) + ".npy", gd) 

def run_gradient_descent_trials(N=100, num_steps=1000):
    for i in range(1,5):
        gradient_descent_trials(i, xyz_trace, N=N, num_steps=num_steps)


def genetic_algorithm_trials(n, xyz_trace, N=100, ngen=20, pop_size=40):
    formula = select_formula(n)
    losses = []
    num_iterations = []
    settings = {'select': tools.selection.selTournament,
            'crossover': tools.crossover.cxSimulatedBinary,
            'mutate':tools.mutation.mutPolynomialBounded,
            'ngen': ngen,
            'tournsize': 10,
            'cxpb': 0.5,
            'indpb': 0.01,
            'mutpb': 0.2,
            'eta': 0.5,
            'lb': 0,
            'ub': 10}
    ga = opt.STLGeneticAlgorithm(settings, verbose=True)
    var = [α, β]
    t0 = time.time()
    for run in range(N):
        var = [α, β]
        pop = ga.create_population(var, pop_size, loss_fn, formula, xyz_trace)
        for step in range(settings['ngen']):
            if step % 5 == 0:
                print("step = ", step)
            pop = ga.step(pop)
            best = ga.select_best(pop)
            loss = loss_fn(formula, {α:best[0][0], β:best[0][1]}, xyz_trace)
            if loss < 1E-6:
                break
        best = ga.select_best(pop)
        loss = loss_fn(formula, {α:best[0][0], β:best[0][1]}, xyz_trace)
        losses.append(loss)
        num_iterations.append(step+1)
        print("run = ", run, "step = ", step, " loss = ", loss)
    t1 = time.time()
    total = t1 - t0
    ga = {"total_time": total,"losses": losses, "num_iterations": num_iterations}
    np.save("ga_"+str(n) + "_" + str(ngen) + "_" + str(pop_size) + ".npy", ga) 

def run_genetic_algorithm_trials(N=100, ngen=20, pop_size=40):
    for i in range(1,5):
        print("======= equation ", i, " ========")
        genetic_algorithm_trials(i, xyz_trace, N=N, ngen=ngen, pop_size=pop_size)



#  Computation graph stuff
# formula 1
α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
A = ln.LessThan("x", α)
B = ln.LessThan("y", β)
lhs = ln.STLModel(A, ln.Always())
rhs = ln.STLModel(B, ln.Always())

x_setup = lhs.setup(x_tensor, scale=scale)
lhs_outputs, _ = lhs.outer.run_cell(x_setup, lhs.outer.initialize_rnn_cell(x_setup), scale=scale)
lhs_output = lhs(x_tensor, scale=scale)

y_setup = rhs.setup(y_tensor, scale=scale)
rhs_outputs, _ = rhs.outer.run_cell(y_setup, rhs.outer.initialize_rnn_cell(y_setup), scale=scale)
rhs_output = rhs(x_tensor, scale=scale)
formula_1 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))
loss = formula_1(lhs_output, rhs_output, scale=scale)
loss


# formula 2
α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
A = ln.LessThan("x", α)
B = ln.LessThan("y", β)
C = ln.GreaterThan("z", α)
lhs = ln.STLModel(A, ln.Always())
rhs = ln.STLModel(B, ln.Always())

LHS = ln.And(ln.GreaterThan(str(lhs), 0), ln.GreaterThan(str(rhs), 0))
RHS = ln.STLModel(C, ln.Always())

x_setup = lhs.setup(x_tensor, scale=scale)
lhs_outputs, _ = lhs.outer.run_cell(x_setup, lhs.outer.initialize_rnn_cell(x_setup), scale=scale)
lhs_output = lhs(x_tensor, scale=scale)
y_setup = rhs.setup(y_tensor, scale=scale)
rhs_outputs, _ = rhs.outer.run_cell(y_setup, rhs.outer.initialize_rnn_cell(y_setup), scale=scale)
rhs_output = rhs(x_tensor, scale=scale)
RHS_output = RHS(z_tensor, scale=scale)
formula_2 = ln.Or(ln.GreaterThan(str(LHS), 0), ln.GreaterThan(str(RHS), 0))
loss = formula_2(LHS(lhs_output, rhs_output, scale=scale), RHS_output, scale=scale)




# formula 3
α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
A = ln.GreaterThan("x", α)
B = ln.LessThan("y", β)
AA = ln.And(A, B)
temp_inner = ln.Always(I1)
temp_outer = ln.Eventually()
temp = ln.TemporalComposition(temp_inner, temp_outer)
lhs = ln.STLModel(AA, temp)

inner = ln.LessThan("z", α)
outer = ln.Always()
rhs = ln.STLModel(inner, outer)

lhs_value = lhs(x_tensor, y_tensor, scale=scale)
rhs_value = rhs(z_tensor, scale=scale)
formula_3 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))


#  formula 4
α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
A = ln.GreaterThan("x", α + β)
B = ln.LessThan("y", α**2 - β)
AA = ln.And(A, B)
temp_inner = ln.Always(I1)
temp_outer = ln.Eventually()
temp = ln.TemporalComposition(temp_inner, temp_outer)
lhs = ln.STLModel(AA, temp)
inner = ln.LessThan("z", α + 2*β**2)
outer = ln.Always()
rhs = ln.STLModel(inner, outer)

lhs_value = lhs(x_tensor, y_tensor, scale=scale)
rhs_value = rhs(z_tensor, scale=scale)
formula_4 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))



def run_formula_1(N, scaling=True, verbose=True):
    NUM_EPOCH = N
    learning_rate = 0.01
    α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    A = ln.LessThan("x", α)
    B = ln.LessThan("y", β)
    lhs = ln.STLModel(A, ln.Always())
    rhs = ln.STLModel(B, ln.Always())
    optim = torch.optim.Adam([α,β], lr=learning_rate)
    for epoch in range(NUM_EPOCH):            
        # ------------------------------------------ exact loss -------------------------------------------------------- #
        scale = -1
        x_setup = lhs.setup(x_tensor, scale=scale)
        lhs_outputs, _ = lhs.outer.run_cell(x_setup, lhs.outer.initialize_rnn_cell(x_setup), scale=scale)
        lhs_output = lhs(x_tensor, scale=scale)
        y_setup = rhs.setup(y_tensor, scale=scale)
        rhs_outputs, _ = rhs.outer.run_cell(y_setup, rhs.outer.initialize_rnn_cell(y_setup), scale=scale)
        rhs_output = rhs(x_tensor, scale=scale)
        formula_1 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))
        exact_loss = formula_1(lhs_output, rhs_output, scale=scale)**2
        # -------------------------------------------------------------------------------------------------------- #
        if exact_loss <= loss_tol:
            print("EXACT LOSS IS SMALL ENOUGH")
            break
            
        if scaling:
            scale = scaling_schedule(epoch, NUM_EPOCH)

        # ------------------------------------------ LOSS -------------------------------------------------------- #
            x_setup = lhs.setup(x_tensor, scale=scale)
            lhs_outputs, _ = lhs.outer.run_cell(x_setup, lhs.outer.initialize_rnn_cell(x_setup), scale=scale)
            lhs_output = lhs(x_tensor, scale=scale)
            y_setup = rhs.setup(y_tensor, scale=scale)
            rhs_outputs, _ = rhs.outer.run_cell(y_setup, rhs.outer.initialize_rnn_cell(y_setup), scale=scale)
            rhs_output = rhs(x_tensor, scale=scale)
            formula_1 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))
            loss = formula_1(lhs_output, rhs_output, scale=scale)**2
        # -------------------------------------------------------------------------------------------------------- #
        else:
            loss = exact_loss

        loss.backward()
        with torch.no_grad():
            if epoch % 100 == 0:
                if verbose:
                    print("Epoch: ", epoch,
                          "\t β is : ", round(β.cpu().data.numpy()[0],3),
                          "\t α is : ", round(α.cpu().data.numpy()[0],3),
                          "\t Gradient: ", round(β.grad.data.numpy()[0]**2 + α.grad.data.numpy()[0]**2, 7),
                          "\t Loss: ", round(float(loss.data.numpy()), 3),
                          "\t Exact Loss: ", round(float(exact_loss.data.numpy()), 3),
                          "\t Scale: ", round(scale, 3))
#             β -= learning_rate * (β.grad + 0.001*torch.randn(1,  dtype=torch.float64))
#             α -= learning_rate * (α.grad + 0.001*torch.randn(1,  dtype=torch.float64))
            optim.step()
            if α.grad**2 + β.grad**2  < gradient_tol:
                print("Gradient IS SMALL ENOUGH")
                break
            β.grad.zero_()
            α.grad.zero_()
    return exact_loss, epoch


def run_formula_2(N, scaling=True, verbose=True):
    NUM_EPOCH = N
    learning_rate = 0.01
    α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    A = ln.LessThan("x", α)
    B = ln.LessThan("y", β)
    C = ln.GreaterThan("z", α)
    lhs = ln.STLModel(A, ln.Always())
    rhs = ln.STLModel(B, ln.Always())
    LHS = ln.And(ln.GreaterThan(str(lhs), 0), ln.GreaterThan(str(rhs), 0))
    RHS = ln.STLModel(C, ln.Always())
    optim = torch.optim.Adam([α,β], lr=learning_rate)
    for epoch in range(NUM_EPOCH):
        # ------------------------------------------ exact loss -------------------------------------------------------- #
        scale = -1
        x_setup = lhs.setup(x_tensor, scale=scale)
        lhs_outputs, _ = lhs.outer.run_cell(x_setup, lhs.outer.initialize_rnn_cell(x_setup), scale=scale)
        lhs_output = lhs(x_tensor, scale=scale)
        y_setup = rhs.setup(y_tensor, scale=scale)
        rhs_outputs, _ = rhs.outer.run_cell(y_setup, rhs.outer.initialize_rnn_cell(y_setup), scale=scale)
        rhs_output = rhs(x_tensor, scale=scale)
        RHS_output = RHS(z_tensor, scale=scale)
        formula_2 = ln.Or(ln.GreaterThan(str(LHS), 0), ln.GreaterThan(str(RHS), 0))
        exact_loss = formula_2(LHS(lhs_output, rhs_output, scale=scale), RHS_output, scale=scale)**2
        
        # -------------------------------------------------------------------------------------------------------- #
        if exact_loss <= loss_tol:
            print("EXACT LOSS IS SMALL ENOUGH")
            break
            
        if scaling:
            scale = scaling_schedule(epoch, NUM_EPOCH)
        
        # ------------------------------------------ LOSS -------------------------------------------------------- #
            x_setup = lhs.setup(x_tensor, scale=scale)
            lhs_outputs, _ = lhs.outer.run_cell(x_setup, lhs.outer.initialize_rnn_cell(x_setup), scale=scale)
            lhs_output = lhs(x_tensor, scale=scale)
            y_setup = rhs.setup(y_tensor, scale=scale)
            rhs_outputs, _ = rhs.outer.run_cell(y_setup, rhs.outer.initialize_rnn_cell(y_setup), scale=scale)
            rhs_output = rhs(x_tensor, scale=scale)
            RHS_output = RHS(z_tensor, scale=scale)
            formula_2 = ln.Or(ln.GreaterThan(str(LHS), 0), ln.GreaterThan(str(RHS), 0))
            loss = formula_2(LHS(lhs_output, rhs_output, scale=scale), RHS_output, scale=scale)**2
        else:
            loss = exact_loss
        # -------------------------------------------------------------------------------------------------------- #

        loss.backward()
        with torch.no_grad():
            if epoch % 100 == 0:
                if verbose:
                    print("Epoch number: ", epoch,
                          "\t β is : ", round(β.cpu().data.numpy()[0],3),
                          "\t α is : ", round(α.cpu().data.numpy()[0],3),
                          "\t Gradient is: ", round(β.grad.data.numpy()[0]**2 + α.grad.data.numpy()[0]**2, 7),
                          "\t Loss is: ", round(float(loss.data.numpy()), 3),
                          "\t Exact Loss is: ", round(float(exact_loss.data.numpy()), 3),
                          "\t Scale is: ", round(scale, 3))
#             β -= learning_rate * (β.grad + 0.001*torch.randn(1,  dtype=torch.float64))
#             α -= learning_rate * (α.grad + 0.001*torch.randn(1,  dtype=torch.float64))
            optim.step()
            if α.grad**2 + β.grad**2  < gradient_tol:
                print("Gradient IS SMALL ENOUGH")
                break
            β.grad.zero_()
            α.grad.zero_()
    return exact_loss, epoch




def run_formula_3(N, scaling=True, verbose=True):
    NUM_EPOCH = N
    learning_rate = 0.01
    α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    A = ln.GreaterThan("x", α)
    B = ln.LessThan("y", β)
    AA = ln.And(A, B)
    temp_inner = ln.Always(I1)
    temp_outer = ln.Eventually()
    temp = ln.TemporalComposition(temp_inner, temp_outer)
    lhs = ln.STLModel(AA, temp)
    inner = ln.LessThan("z", α)
    outer = ln.Always()
    rhs = ln.STLModel(inner, outer)
    formula_3 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))
    optim = torch.optim.Adam([α,β], lr=learning_rate)
    for epoch in range(NUM_EPOCH):
        # ------------------------------------------ exact loss -------------------------------------------------------- #
        scale = -1
        lhs_value = lhs(x_tensor, y_tensor, scale=scale)
        rhs_value = rhs(z_tensor, scale=scale)
        exact_loss = formula_3(lhs_value, rhs_value)**2
        # -------------------------------------------------------------------------------------------------------- #
        if exact_loss <= loss_tol:
            print("EXACT LOSS IS SMALL ENOUGH")
            break
            
        if scaling:
            scale = scaling_schedule(epoch, NUM_EPOCH)
        
        # ------------------------------------------ LOSS -------------------------------------------------------- #
            lhs_value = lhs(x_tensor, y_tensor, scale=scale)
            rhs_value = rhs(z_tensor, scale=scale)
            loss = formula_3(lhs_value, rhs_value)**2
        else:
            loss = exact_loss
        # -------------------------------------------------------------------------------------------------------- #

        loss.backward()
        with torch.no_grad():
            if epoch % 100 == 0:
                if verbose:
                    print("Epoch number: ", epoch,
                          "\t β is : ", round(β.cpu().data.numpy()[0],3),
                          "\t α is : ", round(α.cpu().data.numpy()[0],3),
                          "\t Gradient is: ", round(β.grad.data.numpy()[0]**2 + α.grad.data.numpy()[0]**2, 7),
                          "\t Loss is: ", round(float(loss.data.numpy()), 3),
                          "\t Exact Loss is: ", round(float(exact_loss.data.numpy()), 3),
                          "\t Scale is: ", round(scale, 3))
#             β -= learning_rate * (β.grad + 0.001*torch.randn(1,  dtype=torch.float64))
#             α -= learning_rate * (α.grad + 0.001*torch.randn(1,  dtype=torch.float64))
            optim.step()
            if α.grad**2 + β.grad**2  < gradient_tol:
                print("Gradient IS SMALL ENOUGH")
                break
            β.grad.zero_()
            α.grad.zero_()
    return exact_loss, epoch



# need to check this code 
def run_formula_4(N, scaling=True, verbose=True):
    NUM_EPOCH = N
    learning_rate = 0.01
    α = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    β = torch.randn(1,  dtype=torch.float64, requires_grad=True)
    A = ln.GreaterThan("x", α)
    B = ln.LessThan("y", α + β)
    AA = ln.And(A, B)
    temp_inner = ln.Always(I1)
    temp_outer = ln.Eventually()
    temp = ln.TemporalComposition(temp_inner, temp_outer)
    lhs = ln.STLModel(AA, temp)
    inner = ln.LessThan("z", α)
    outer = ln.Always()
    rhs = ln.STLModel(inner, outer)
    formula_3 = ln.And(ln.GreaterThan("lhs", 0), ln.GreaterThan("rhs", 0))
    optim = torch.optim.Adam([α,β], lr=learning_rate)
    for epoch in range(NUM_EPOCH):
        # ------------------------------------------ exact loss -------------------------------------------------------- #
        scale = -1
        lhs_value = lhs(x_tensor, y_tensor, scale=scale)
        rhs_value = rhs(z_tensor, scale=scale)
        exact_loss = formula_3(lhs_value, rhs_value)**2
        # -------------------------------------------------------------------------------------------------------- #
        if exact_loss <= loss_tol:
            print("EXACT LOSS IS SMALL ENOUGH")
            break
            
        if scaling:
            scale = ln.scale_schedule(epoch, NUM_EPOCH)
        
        # ------------------------------------------ LOSS -------------------------------------------------------- #
            lhs_value = lhs(x_tensor, y_tensor, scale=scale)
            rhs_value = rhs(z_tensor, scale=scale)
            loss = formula_3(lhs_value, rhs_value)**2
        else:
            loss = exact_loss
        # -------------------------------------------------------------------------------------------------------- #

        loss.backward()
        with torch.no_grad():
            if epoch % 100 == 0:
                if verbose:
                    print("Epoch number: ", epoch,
                          "\t β is : ", round(β.cpu().data.numpy()[0],3),
                          "\t α is : ", round(α.cpu().data.numpy()[0],3),
                          "\t Gradient is: ", round(β.grad.data.numpy()[0]**2 + α.grad.data.numpy()[0]**2, 7),
                          "\t Loss is: ", round(float(loss.data.numpy()), 3),
                          "\t Exact Loss is: ", round(float(exact_loss.data.numpy()), 3),
                          "\t Scale is: ", round(scale, 3))
#             β -= learning_rate * (β.grad + 0.001*torch.randn(1,  dtype=torch.float64))
#             α -= learning_rate * (α.grad + 0.001*torch.randn(1,  dtype=torch.float64))
            optim.step()
            if α.grad**2 + β.grad**2  < gradient_tol:
                print("Gradient IS SMALL ENOUGH")
                break
            β.grad.zero_()
            α.grad.zero_()
    return exact_loss, epoch

def choose_formula(n):
    if n == 1:
        return run_formula_1
    elif n == 2:
        return run_formula_2
    elif n == 3:
        return run_formula_3
    elif n == 4:
        return run_formula_4
    else: return 0
    


def run_cg_trials(cg):
    N = 50
    NUM_EPOCH = 2000
    verbose = False
    for n in range(1,4):
        run_formula = choose_formula(n)
        
        
        losses = []
        num_iterations = []
        times = []
        scaling = True
        t0 = time.time()
        for run in range(N):
            print("dcg run = ", run)
            t00 = time.time()
            loss, epoch = run_formula(NUM_EPOCH, scaling=scaling, verbose=verbose)
            t11 = time.time()
            print("loss = ", loss, "  epoch = ", epoch, "time = ", t11 - t00)
            losses.append(loss)
            num_iterations.append(epoch)
            times.append(t11-t00)
        t1 = time.time()
        total = t1-t0

        dcg = {"total_time": total, "num_iterations": num_iterations, "losses": losses, "times": times}
        np.save("dcg_"+str(n) + "_tol.npy", dcg) 

        losses = []
        num_iterations = []
        times = []
        scaling = False
        t0 = time.time()
        for run in range(N):
            print("cg run = ", run)
            t00 = time.time()
            loss, epoch = run_formula(NUM_EPOCH, scaling=scaling, verbose=verbose)
            t11 = time.time()
            print("loss = ", loss, "  epoch = ", epoch, "time = ", t11 - t00)
            losses.append(loss)
            num_iterations.append(epoch)
        t1 = time.time()
        total = t1-t0

        cg = {"total_time": total, "num_iterations": num_iterations, "losses": losses, "times": times}
        np.save("cg_"+str(n) + "_tol.npy", cg)

def make_bar_graph(data, formula, value_name = "x", method_names = ['1', '2', '3', '4']):
    df_list = []
    for (i,d) in enumerate(data):
        df = pd.DataFrame(OrderedDict({'Formula': formula, value_name: d}))
        df['method'] = str(i)
        df_list.append(df)

    DF = pd.concat(df_list)
    DFG = DF.groupby(['Formula', 'method'])
    DFGSum = DFG.sum().unstack(['method']).sum(axis=1,level=['method'])
    return DF, DFG, DFGSum


def make_comparison_bar_graph(means, stds, formula, kwargs={}):
    DF, DFG, DFGSum = make_bar_graph(means, formula)
    if stds is None:
        ERRGSum = None
    else:       
        ERR, ERRG, ERRGSum = make_bar_graph(stds, formula)
    DFGSum.plot(kind='bar', yerr=ERRGSum, **kwargs)




def IQR(data):
    a,b = np.percentile(data, [25, 75])
    return b - a

def get_mean_time(data, pop=1):
    total_time = data["total_time"]
    num_iters = sum(data["num_iterations"])
    return total_time/num_iters/pop

def get_mean_iterations(data):
    return [np.median(data["num_iterations"]), IQR(data["num_iterations"])]

def get_mean_loss(data):
    if type(data["losses"][0]) == torch.Tensor:
        losses_np = np.array([d.detach().numpy().item() for d in data["losses"]])
        q75, q25 = np.percentile(losses_np, [75 ,25])
        iqr = q75 - q25
        return [np.median(losses_np), iqr]
    q75, q25 = np.percentile(data["losses"], [75 ,25])
    iqr = q75 - q25
    return [np.median(data["losses"]), iqr]

def get_iqr(data):
    if type(data["losses"][0]) == torch.Tensor:
        losses_np = np.array([d.detach().numpy().item() for d in data["losses"]])
        q75, q25 = np.percentile(losses_np, [75 ,25])
        iqr = q75 - q25
        return q75, q25
    q75, q25 = np.percentile(data["losses"], [75 ,25])
    iqr = q75 - q25
    return q75, q25