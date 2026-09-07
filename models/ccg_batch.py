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


def main():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    out_dir = Path(__file__).parent / 'results'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_path = out_dir / f'{timestamp}_ccg.csv'

    header_written = False
    field_names = None

    n_warehouse = config['n_warehouse']
    base_path = project_root / f'data/w{n_warehouse}'
    print(base_path)
    idx_min = config['idx_min']
    idx_max = config['idx_max']
    gamma_ratio_list = config['gamma_ratio_list']
    EPS_CCG = config['EPS_CCG']

    for idx in range(idx_min, idx_max + 1):
        ins_file = base_path / f'w{n_warehouse}_{idx}.txt'
        ins = Instance()
        ins.load_from_file(str(ins_file))

        B = math.ceil(ins.c.max())

        for gamma_ratio in gamma_ratio_list:
            print('=' * 60)
            print(f'Original C&CG Start, {ins.name}, Gamma ratio = {gamma_ratio}')
            print('=' * 60)
            
            time_mp = 0
            time_sp = 0
            time_total = 0

            LB = -GRB.INFINITY
            UB = GRB.INFINITY
            
            scene_g_dict = {}

            it = 0

            mp = gp.Model()
            mp.Params.OutputFlag = 0
            mp.Params.MIPGap = 0

            y = {}
            z = {}
            x = {}
            for i in range(ins.n_ware):
                y[i] = mp.addVar(vtype=GRB.BINARY, name=f'y_{i}')                                   
                for k in range(ins.n_com):
                    z[i, k] = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'z_{i}_{k}')
                    for j in range(ins.n_dem): 
                        x[i, j, k, it] = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x_{i}_{j}_{k}_{it}')
            eta = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name='eta')

            mp.setObjective(
                quicksum(ins.f[i] * y[i] for i in range(ins.n_ware)) +
                quicksum(ins.a[i, k] * z[i, k] for i in range(ins.n_ware) for k in range(ins.n_com)) + eta,
                GRB.MINIMIZE
            )
            
            for i in range(ins.n_ware):
                mp.addConstr(quicksum(z[i, k] for k in range(ins.n_com)) <= ins.Z[i] * y[i])
            mp.update()
            
            sp = gp.Model()
            sp.Params.OutputFlag = 0

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

            x = {}

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

                g_opt = np.array([round(g[j].X) for j in range(ins.n_dem)])
                scene_g_dict[it] = g_opt

                UB = min(UB, mp.ObjVal - eta.X + sp.ObjVal)
                
                ni = int(mp.NumIntVars)
                nc = int(mp.NumVars - mp.NumIntVars)
                print(f'Iter {it}: U={UB:.0f}, L={LB:.0f}, gap={(UB-LB)/UB:.2%}, '
                      f'time(T/M/S)={time_total:.0f}/{time_mp:.0f}/{time_sp:.0f}, '
                      f'MP#vars(I/C)={ni}/{nc}, '
                      f'MP#cons={mp.NumConstrs}')       
                
                if (UB - LB) / UB <= EPS_CCG:
                    break

                for i in range(ins.n_ware):
                    for j in range(ins.n_dem):
                        for k in range(ins.n_com):
                            x[i, j, k, it] = mp.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'x_{i}_{j}_{k}_{it}')
                for j in range(ins.n_dem):
                    for k in range(ins.n_com):
                        mp.addConstr(quicksum(x[i, j, k, it] for i in range(ins.n_ware))
                                     >= ins.d[j, k] + ins.p[j, k] * g_opt[j])
                for i in range(ins.n_ware):
                    for k in range(ins.n_com):
                        mp.addConstr(quicksum(x[i, j, k, it] for j in range(ins.n_dem)) <= z[i, k])
                mp.addConstr(eta >= quicksum(ins.c[i, j, k] * x[i, j, k, it]
                             for i in range(ins.n_ware) for j in range(ins.n_dem) for k in range(ins.n_com)))
                mp.update()

            y_sol = y_val.round().astype(int)
            opened = [i for i in range(ins.n_ware) if y_sol[i] == 1]
            y_sol_str = ''.join(map(str, y_sol))

            print('=' * 60)
            print('Original C&CG Final Results')
            print(f'Optimal value = {UB:.0f}')
            print(f'Time (total/MP/SP) = {time_total:.2f}/{time_mp:.2f}/{time_sp:.2f}')
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
                'alg': 'ccg',
                'time_total': round(time_total, 3),
                'time_mp': round(time_mp, 3),
                'time_sp': round(time_sp, 3),
                'iter_count': it,
                'n_sce_final': len(scene_g_dict),
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
