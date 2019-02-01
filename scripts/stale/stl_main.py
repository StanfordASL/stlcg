import stl.stl_optimizer as opt
import stl.stl as stl
from features import *
from utils import *

import numpy as np
from collections import OrderedDict
import sympy as sym

import copy

params = {'a_min': -6,
          't_reaction': 0.3,
          'longitudinal_min': 5.5,
          'lateral_min': 2.25,
          'speed_limit': 16

         }


def optimize_gradient_descent(formula, opt, var, loss_fn, batch, 
                              initializer_fn=lambda var: {p:50*(np.random.ranf()-0.5) for p in var}, 
                              max_steps = 1000):
    losses = []
    valuations = OrderedDict(initializer_fn(var))
    step = 0
    loss = loss_fn(formula, valuations, batch)
    valuations, grad = opt.step(valuations, formula, loss_fn, batch)
    grad_norm = sum([v**2 for v in grad.values()])
    if grad_norm < 1E-4:
        while grad_norm < 1E-4:
            valuations = OrderedDict({p:50*(np.random.ranf()-0.5) for p in var})
            valuations, grad = opt.step(valuations, formula, loss_fn, batch)
            grad_norm = sum([v**2 for v in grad.values()])

    while step < max_steps and np.abs(loss) > 1E-3 and grad_norm > 1E-3:
        # opt.lr = np.exp(-step/10)
        print("STEP: " + str(step) + "    Loss:   " + str(round(loss, 3)) + "    Grad norm:  " + str(round(grad_norm,3)) + "   =======================================================================================")
        step += 1
        loss = loss_fn(formula, valuations, batch)
        losses.append(loss)
        valuations, grad = opt.step(valuations, formula, loss_fn, batch)
        grad_norm = sum([v**2 for v in grad.values()])    
        step += 1

    return valuations, losses


def optimize_genetic_algorithm(formula, opt, var, loss_fn, batch, settings,
                               initializer_fn=lambda x: 20*(np.random.ranf(x)-0.5)):
    losses = []
    bestes = []
    mem = settings['memory']
    pop = opt.create_population(var, settings['pop_size'], loss_fn, formula, batch, initializer_fn)
    best = opt.select_best(pop)[0]
    bestes.append(best)
    loss = opt.loss(best)[0]
    losses.append(-float(loss))
    string = "STEP: 0" + "    Loss:   " + str(round(loss, 3)) + "    "
    for (i, v) in enumerate(best):
        string += str(var[i]) + ":  " + str(round(v, 3)) + "     "
    print(string)
    step = 1
    while step < settings['ngen'] and np.abs(loss) > settings['tol']:
        pop = opt.step(pop)
        best = opt.select_best(pop)[0]
        bestes.append(best)
        loss = opt.loss(best)[0]
        losses.append(-float(loss))
        string = "STEP: " + str(step) + "    Loss:   " + str(round(loss, 3)) + "    "
        for (i, v) in enumerate(best):
            string += str(var[i]) + ":  " + str(round(v, 3)) + "     "
        print(string)
        step += 1
        if step < mem:
            continue
        # if the loss is stuck, create population again
        if np.abs(np.mean(losses[step-mem:step]) + loss) < 1E-3 and np.abs(loss) > settings['tol']:
            print("WARN: ******* GENOCIDE! *******")
            pop = opt.create_population(var, settings['pop_size'], loss_fn, formula, batch)

    best = opt.select_best(pop)[0]
    return {var[i]: v for (i,v) in enumerate(best)}, losses, bestes



def eval_variables(formula, valuation):
    _formula = copy.deepcopy(formula)
    for (p,v) in valuation.items():
        _formula = _formula.subs(p,v)
    return _formula



def default_loss_fn(formula, valuation, batch):
    _formula = eval_variables(formula, valuation)
    if type(batch) is not list:
        return _formula.robustness(batch)**2
    else:
        return sum([_formula.robustness(b)**2 for b in batch])



def front_stopping_distance_formula(fsdf,
                            fsdfl,
                            fsdfr,
                            dyfl,
                            dyfr,
                            direction,
                            min_lateral_gap=sym.simplify(3), 
                            soft=False):
    '''
    This computes the STL formula for the stopping 
    distance between the front vehicle and the front 
    adjacent vehicle depending on which direction 
    the ego vehicle is turning into.
    The formula is of the form:
    ⬓(fsdf > α) ∧ ♢ (⬓_[0,τ] dyfa < 2.8 ∧ fsdfa > β) 

    Inputs: fsdf, fsdfl, fsdfr: Front Stopping Distance Front/Front Left/Front Right variable/symbols
            dyf, dyfl, fyfr: y-distance from the ego and adjacent vehicle.
            direction: which direction Left or Right is the ego vehicle changing lanes into
    Output: the FSD formula
    '''
    (α, β, τ) = var = sym.symbols('α β τ')
    if direction == 'left':
        dyfa = dyfl
        fsdfa = fsdfl
    elif direction == 'right':
        dyfa = dyfr
        fsdfa = fsdfr
    else:
        raise ValueError(direction + " is not defined")
    A = stl.Comparison('<', dyfa, min_lateral_gap)
    B = stl.Comparison('>', fsdfa, β)
    adjacent = stl.And(A, B, soft=soft)
    always_adjacent = stl.Always(adjacent, stl.Interval(0, τ))
    adjacent_formula = stl.Eventually(always_adjacent)
    C = stl.Comparison('>', fsdf, α)
    return stl.Always(C), adjacent_formula, var

def cross_over_time_formula(cotf,
                            cotrl,
                            cotrr,
                            dyf,
                            dyrl,
                            dyrr,
                            direction,
                            min_lateral_gap=sym.simplify(3), 
                            soft=False):
    '''
    This computes the STL formula for the cross over 
    time between the front vehicle and the rear 
    adjacent vehicle depending on which direction 
    the ego vehicle is turning into.
    The formula is of the form:
    ⬓(cotf > α) ∧ ♢ (⬓_[0,τ] dyra < 2.8 ∧ cotra > β) 

    Inputs: cotf, cotrl, cotrr: Cross Over Time Front/Rear Left/Rear Right variable/symbols
            dyf, dyrl, fyrr: y-distance from the ego and adjacent vehicle.
            direction: which direction Left or Right is the ego vehicle changing lanes into
    Output: the COT formula
    '''
    (α, β, τ) = var = sym.symbols('α β τ')
    if direction == 'left':
        dyra = dyrl
        cotra = cotrl
    elif direction == 'right':
        dyra = dyrr
        cotra = cotrr
    else:
        raise ValueError(direction + " is not defined")
    A = stl.Comparison('<', dyra, min_lateral_gap)
    B = stl.Comparison('>', cotra, β)
    adjacent = stl.And(A, B, soft=soft)
    always_adjacent = stl.Always(adjacent, stl.Interval(0, τ))
    adjacent_formula = stl.Eventually(always_adjacent)
    C = stl.Comparison('>', cotf, α)
    return stl.Always(C), adjacent_formula, var

def lane_change_formula(dxrl,
                        dxrr,
                        cotrl,
                        cotrr,
                        lane_idx,
                        direction,
                        soft=False):
    (ε, δ, τ) = var = sym.symbols('ε δ τ')
    end_lane = sym.simplify(1) if direction == 'left' else sym.simplify(0)
    lane = stl.Comparison('==', lane_idx, end_lane)
    changed_lanes = stl.Next(stl.Always(lane))

    if direction == 'left':
        dxra = dxrl
        cotra = cotrl
    elif direction == 'right':
        dxra = dxrr
        cotra = cotrr
    else:
        raise ValueError(direction + " is not defined")

    A = stl.Comparison('>', dxra, ε)
    B = stl.Comparison('>', cotra, δ)
    lane_change = stl.And(A, B, soft=soft)
    lane_change_safety = stl.Always(lane_change, stl.Interval(0, τ))

    return stl.Eventually(stl.Until(lane_change_safety, changed_lanes)), var



