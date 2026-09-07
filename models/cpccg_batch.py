from gurobipy import GRB, quicksum
import gurobipy as gp
import numpy as np
import math
import sys
import yaml
from pathlib import Path
from datetime import datetime
import csv

from src.instance import Instance

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_master(ins, g_work):
    mp = gp.Model()
    mp.Params.OutputFlag = 0
    mp.Params.MIPGap = 0

    y = {}
    z = {}
    for i in range(ins.n_ware):
        y[i] = mp.addVar(vtype=GRB.BINARY, name=f'y_{i}')
        for k in range(ins.n_com):
            z[i, k] = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'z_{i}_{k}')
    eta = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name='eta')

    mp.setObjective(
        quicksum(ins.f[i] * y[i] for i in range(ins.n_ware)) +
        quicksum(ins.a[i, k] * z[i, k] for i in range(ins.n_ware) for k in range(ins.n_com)) + eta,
        GRB.MINIMIZE
    )

    for i in range(ins.n_ware):
        mp.addConstr(quicksum(z[i, k] for k in range(ins.n_com)) <= ins.Z[i] * y[i])

    x = {}

    for s in g_work.keys():
        for i in range(ins.n_ware):
            for j in range(ins.n_dem):
                for k in range(ins.n_com):
                    x[i, j, k, s] = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x_{i}_{j}_{k}_{s}')
        for j in range(ins.n_dem):
            for k in range(ins.n_com):
                mp.addConstr(quicksum(x[i, j, k, s] for i in range(ins.n_ware)) >=
                             ins.d[j, k] + ins.p[j, k] * g_work[s][j], name=f'xdem_{j}_{k}_{s}')
        for i in range(ins.n_ware):
            for k in range(ins.n_com):
                mp.addConstr(quicksum(x[i, j, k, s] for j in range(ins.n_dem)) <= z[i, k], name=f'xcap_{i}_{k}_{s}')
        mp.addConstr(eta >= quicksum(ins.c[i, j, k] * x[i, j, k, s]
                     for i in range(ins.n_ware) for j in range(ins.n_dem) for k in range(ins.n_com)), name=f'eta_{s}')
    mp.update()

    return mp, y, z, x, eta


def build_subproblem(ins, gamma_ratio):
    sp = gp.Model()
    sp.Params.OutputFlag = 0
    B = math.ceil(ins.c.max()) # set to a reasonable value

    g = {}
    lam = {}
    pi = {}
    w = {}
    for j in range(ins.n_dem):
        g[j] = sp.addVar(vtype=GRB.BINARY, name=f'g_{j}')
        for k in range(ins.n_com):
            lam[j, k] = sp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'lam_{j}_{k}')
            w[j, k] = sp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'w_{j}_{k}')
    for i in range(ins.n_ware):
        for k in range(ins.n_com):
            pi[i, k] = sp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'pi_{i}_{k}')

    sp.addConstr(quicksum(g[j] for j in range(ins.n_dem)) <= math.ceil(ins.n_dem * gamma_ratio))

    for i in range(ins.n_ware):
        for j in range(ins.n_dem):
            for k in range(ins.n_com):
                sp.addConstr(lam[j, k] - pi[i, k] <= ins.c[i, j, k])

    for j in range(ins.n_dem):
        for k in range(ins.n_com):
            sp.addConstr(w[j, k] <= lam[j, k])
            sp.addConstr(w[j, k] <= B * g[j])
            sp.addConstr(w[j, k] >= lam[j, k] - B * (1 - g[j]))
            sp.addConstr(lam[j, k] <= B)

    sp.update()

    return sp, g, lam, pi, w


def prune_scenes(ins, it, g_work, lam_cache, force_keep, prune_threshold, eps_inequality):
    m = len(g_work)
    if m <= force_keep + 1:
        return g_work.keys(), [], 0

    g_work_keys = sorted(g_work.keys())
    lam_cache_keys = sorted(lam_cache.keys())
    candidate_sce = g_work_keys[:-force_keep]
    forced_sce = g_work_keys[-force_keep:]

    val = np.zeros((it+1, it+1))
    for s in candidate_sce:
        demand_dev = ins.p * g_work[s].reshape(-1, 1)
        for l in lam_cache_keys:
            val[s, l] = np.sum(lam_cache[l] * demand_dev)
    max_val = val.max(axis=0)

    hit_sets = {}
    for l in lam_cache_keys:
        set_l = [s for s in candidate_sce if val[s, l] >= max_val[l] * prune_threshold * (1 - eps_inequality)]
        hit_sets[l] = set_l

    pa = gp.Model()
    pa.Params.OutputFlag = 0

    keep = {}
    for s in candidate_sce:
        keep[s] = pa.addVar(vtype=GRB.BINARY, name=f'keep_{s}')

    pa.setObjective(quicksum(keep[s] for s in candidate_sce), GRB.MINIMIZE)

    for l in lam_cache_keys:
        pa.addConstr(quicksum(keep[s] for s in hit_sets[l]) >= 1)

    pa.update()
    pa.optimize()

    kept_sce = [s for s in candidate_sce if round(keep[s].X) == 1] + forced_sce
    pruned_sce = [s for s in candidate_sce if round(keep[s].X) == 0]

    return kept_sce, pruned_sce, pa.Runtime


def main():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    out_dir = Path(__file__).parent / 'results'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_path = out_dir / f'{timestamp}_cpccg.csv'

    header_written = False
    field_names = None

    n_warehouse = config['n_warehouse']
    base_path = project_root / f'data/w{n_warehouse}'
    idx_min = config['idx_min']
    idx_max = config['idx_max']
    gamma_ratio_list = config['gamma_ratio_list']
    EPS_CCG = config['EPS_CCG']
    EPS_INEQUALITY = config['EPS_INEQUALITY']
    prune_interval_list = config['prune_interval_list']
    force_keep = config['force_keep']
    prune_threshold_list = config['prune_threshold_list']
    stop_prune_ratio_list = config['stop_prune_ratio_list']
    lb_degradation_tol_list = config['lb_degradation_tol_list']

    for idx in range(idx_min, idx_max + 1):
        ins_file = base_path / f'w{n_warehouse}_{idx}.txt'
        ins = Instance()
        ins.load_from_file(str(ins_file))

        for gamma_ratio in gamma_ratio_list:
            for prune_interval in prune_interval_list:
                for prune_threshold in prune_threshold_list:
                    for stop_prune_ratio in stop_prune_ratio_list:
                        for lb_degradation_tol in lb_degradation_tol_list:
                            if prune_interval == 5 and prune_threshold == 0.95 and stop_prune_ratio == 5 and lb_degradation_tol == 1.5:
                                continue
                            print('=' * 60)
                            print(f'CP-C&CG Start, {ins.name}, Gamma ratio = {gamma_ratio}')
                            print(f'Prune interval = {prune_interval}, force keep = {force_keep},\n'
                                  f'prune threshold = {prune_threshold}, stop prune ratio = {stop_prune_ratio}, '
                                  f'lb degradation tol = {lb_degradation_tol}')
                            print('=' * 60)
    
                            time_mp = 0
                            time_pa = 0
                            time_sp = 0
                            time_total = 0
    
                            LB = -GRB.INFINITY
                            UB = GRB.INFINITY
                            LB_last_prune = -GRB.INFINITY
                            gap = 1
    
                            g_work = {}  # working scenarios
                            g_cache = {}  # all scenarios
                            lam_cache = {}  # all dual directions
    
                            it = 0
    
                            mp, y, z, x, eta = build_master(ins, g_work)
                            sp, g, lam, pi, w = build_subproblem(ins, gamma_ratio)
    
                            next_prune_it = prune_interval
                            prune_interval_dyn = prune_interval
                            prune_threshold_dyn = prune_threshold
                            y_new_val = None
                            z_new_val = None
                            n_rollback = 0
    
                            while True:
                                it += 1
    
                                mp.optimize()
                                time_mp += mp.Runtime
                                time_total += mp.Runtime
    
                                LB = mp.ObjVal
    
                                y_val = np.array([y[i].X for i in range(ins.n_ware)])
                                z_val = np.array([[z[i, k].X for k in range(ins.n_com)] for i in range(ins.n_ware)])
    
                                obj = quicksum(ins.d[j, k] * lam[j, k] for j in range(ins.n_dem) for k in range(ins.n_com))
                                obj += quicksum(ins.p[j, k] * w[j, k] for j in range(ins.n_dem) for k in range(ins.n_com))
                                obj -= quicksum(z_val[i, k] * pi[i, k] for i in range(ins.n_ware) for k in range(ins.n_com))
                                sp.setObjective(obj, GRB.MAXIMIZE)
    
                                sp.update()
                                sp.optimize()
                                time_sp += sp.Runtime
                                time_total += sp.Runtime
    
                                UB = min(UB, mp.ObjVal - eta.X + sp.ObjVal)
                                gap = (UB - LB) / UB
    
                                g_opt = np.array([round(g[j].X) for j in range(ins.n_dem)])
                                lam_opt = np.array([[lam[j, k].X for k in range(ins.n_com)] for j in range(ins.n_dem)])
    
                                g_work[it] = g_opt
                                g_cache[it] = g_opt
                                if it > 1: # The first iteration produces a trivial dual direction; discard it
                                    lam_cache[it] = lam_opt
    
                                ni = int(mp.NumIntVars)
                                nc = int(mp.NumVars - mp.NumIntVars)
                                print(f'Iter {it}: U={UB:.0f}, L={LB:.0f}, gap={(UB-LB)/UB:.2%}, '
                                      f'#sce={len(g_work)}, '
                                      f'time(T/M/P/S)={time_total:.0f}/{time_mp:.0f}/{time_pa:.0f}/{time_sp:.0f}, '
                                      f'MP#vars(I/C)={ni}/{nc}, '
                                      f'MP#cons={mp.NumConstrs}')
    
                                if gap <= EPS_CCG:
                                    break
    
                                for i in range(ins.n_ware):
                                    for j in range(ins.n_dem):
                                        for k in range(ins.n_com):
                                            x[i, j, k, it] = mp.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                                       name=f'x_{i}_{j}_{k}_{it}')
                                for j in range(ins.n_dem):
                                    for k in range(ins.n_com):
                                        mp.addConstr(quicksum(x[i, j, k, it] for i in range(ins.n_ware))
                                                     >= ins.d[j, k] + ins.p[j, k] * g_opt[j], name=f'xdem_{j}_{k}_{it}')
                                for i in range(ins.n_ware):
                                    for k in range(ins.n_com):
                                        mp.addConstr(quicksum(x[i, j, k, it] for j in range(
                                            ins.n_dem)) <= z[i, k], name=f'xcap_{i}_{k}_{it}')
                                mp.addConstr(eta >= quicksum(ins.c[i, j, k] * x[i, j, k, it]
                                             for i in range(ins.n_ware) for j in range(ins.n_dem) for k in range(ins.n_com)), name=f'eta_{it}')
                                mp.update()
    
                                # Try to prune
                                if it == next_prune_it and gap > EPS_CCG * stop_prune_ratio:
                                    kept_sce, pruned_sce, pa_time = prune_scenes(
                                        ins, it, g_work, lam_cache, force_keep, prune_threshold_dyn, EPS_INEQUALITY)
                                    time_pa += pa_time
                                    time_total += pa_time
    
                                    g_work_new = {s: g_work[s] for s in kept_sce}
    
                                    mp_new, y_new, z_new, x_new, eta_new = build_master(ins, g_work_new)
    
                                    mp_new.optimize()
                                    time_mp += mp_new.Runtime
                                    time_total += mp_new.Runtime
    
                                    LB_new = mp_new.ObjVal
                                    gap_new = (UB - LB_new) / UB
    
                                    y_new_val = np.array([y_new[i].X for i in range(ins.n_ware)])
                                    z_new_val = np.array([[z_new[i, k].X for k in range(ins.n_com)]
                                                         for i in range(ins.n_ware)])
    
                                    print(f'  -> Prune: kept {len(kept_sce)}/{len(g_work)} scenarios')
    
                                    if LB_new < LB_last_prune * (1 + EPS_INEQUALITY) or gap_new > gap * lb_degradation_tol:
                                        prune_interval_dyn += 2
                                        prune_threshold_dyn = min(prune_threshold_dyn +
                                                                  (1.0 - prune_threshold_dyn) / 2, 1.0)
                                        print(f'  -> Rollback: LB degradation, degraded gap = {gap_new:.2%}')
                                        print(f'  -> Set prune threshold to {prune_threshold_dyn}')
                                        n_rollback += 1
                                    else:
                                        g_work = g_work_new
                                        mp, y, z, x, eta = mp_new, y_new, z_new, x_new, eta_new
                                        y_val, z_val = y_new_val, z_new_val
                                        LB = LB_new                                        
                                        gap = gap_new
                                        
                                    LB_last_prune = LB
                                    next_prune_it += prune_interval_dyn
    
                            y_sol = y_val.round().astype(int)
                            opened = [i for i in range(ins.n_ware) if y_sol[i] == 1]
                            y_sol_str = ''.join(map(str, y_sol))
    
                            print('=' * 60)
                            print('CP-C&CG Final Results')
                            print(f'Optimal value = {UB:.0f}')
                            print(
                                f'Time (total/MP/pruning/SP) = {time_total:.2f}/{time_mp:.2f}/{time_pa:.2f}/{time_sp:.2f}')
                            print(f'Total iterations = {it}')
                            print(f'Opened warehouses ({len(opened)}): {opened}')
                            print('=' * 60)
                            print()
    
                            row = {
                                'name': ins.name,
                                'n_warehouse': ins.n_ware,
                                'n_demand': ins.n_dem,
                                'n_commodity': ins.n_com,
                                'gamma_ratio': round(gamma_ratio, 3),
                                'alg': 'cpccg',
                                'prune_interval': prune_interval,
                                'force_keep': force_keep,
                                'prune_threshold': prune_threshold,
                                'stop_prune_ratio': stop_prune_ratio,
                                'lb_degradation_tol': lb_degradation_tol,
                                'time_total': round(time_total, 3),
                                'time_mp': round(time_mp, 3),
                                'time_pa': round(time_pa, 3),
                                'time_sp': round(time_sp, 3),
                                'iter_count': it,
                                'n_sce_final': len(g_work),
                                'n_rollback': n_rollback,
                                'n_mp_var_int': mp.NumIntVars,
                                'n_mp_var_cont': mp.NumVars - mp.NumIntVars,
                                'n_mp_constr': mp.NumConstrs,
                                'ub': round(UB, 3),
                                'lb': round(LB, 3),
                                'gap_abs': round(UB - LB, 3),
                                'y_sol_str': y_sol_str,
                            }
    
                            if not field_names:
                                field_names = config['field_names']
    
                            with open(csv_path, 'w' if not header_written else 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=field_names)
                                if not header_written:
                                    writer.writeheader()
                                    header_written = True
                                writer.writerow(row)


if __name__ == '__main__':
    main()
