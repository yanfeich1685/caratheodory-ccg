import numpy as np
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

n_ins = 20

n_warehouse = 30
n_demand = 15
n_commodity = 5

f_min, f_max = 500, 2000
a_min, a_max = 5, 30
c_min, c_max = 1, 50
d_min, d_max = 20, 200
dev_ratio_min, dev_ratio_max = 0.2, 0.6
Z_min, Z_max = 500, 2000

seed = 1077 # https://en.wikipedia.org/wiki/Road_to_Canossa


def generate_instance(seed, n_ins):
    np.random.seed(seed)
    for idx in range(n_ins):
        path = project_root / f'data/w{n_warehouse}/w{n_warehouse}_{idx+1}.txt'

        with open(path, 'w') as file:
            while True:
                f = np.random.randint(f_min, f_max + 1, size=n_warehouse).astype(int)
                Z = np.random.randint(Z_min, Z_max + 1, size=n_warehouse).astype(int)
                a = np.random.randint(a_min, a_max + 1, size=(n_warehouse, n_commodity)).astype(int)
                d = np.random.randint(d_min, d_max + 1, size=(n_demand, n_commodity)).astype(int)
                dev_ratio = dev_ratio_min + (dev_ratio_max - dev_ratio_min) * np.random.rand(n_demand, n_commodity)
                p = np.round(d * dev_ratio).astype(int)
                c = np.random.randint(c_min, c_max + 1, size=(n_warehouse, n_demand, n_commodity)).astype(int)
                if Z.sum() >= d.sum() + p.sum():
                    break

            file.write(f'{n_warehouse} {n_demand} {n_commodity}\n')
            file.write(' '.join(map(str, f)) + '\n')
            file.write(' '.join(map(str, Z)) + '\n')
            for i in range(n_warehouse):
                file.write(' '.join(map(str, a[i])) + '\n')
            for j in range(n_demand):
                file.write(' '.join(map(str, d[j])) + '\n')
            for j in range(n_demand):
                file.write(' '.join(map(str, p[j])) + '\n')
            for i in range(n_warehouse):
                for j in range(n_demand):
                    file.write(' '.join(map(str, c[i, j])) + '\n')

        print(f'Instance generated {path}')


class Instance:
    def __init__(self):
        self.filepath = None
        self.filename = None
        self.name = None
        self.idx = None

        self.n_ware = None
        self.n_dem = None
        self.n_com = None

        self.f = None
        self.a = None
        self.d = None
        self.c = None
        self.Z = None

    def load_from_file(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.name = os.path.splitext(self.filename)[0]

        with open(filepath, 'r') as f:
            self.n_ware, self.n_dem, self.n_com = map(int, f.readline().split())
            self.f = np.array([int(x) for x in f.readline().split()]).astype(int)
            self.Z = np.array([int(x) for x in f.readline().split()]).astype(int)
            self.a = np.zeros((self.n_ware, self.n_com))
            for i in range(self.n_ware):
                self.a[i] = np.array([int(x) for x in f.readline().split()]).astype(int)
            self.d = np.zeros((self.n_dem, self.n_com))
            for j in range(self.n_dem):
                self.d[j] = np.array([int(x) for x in f.readline().split()]).astype(int)
            self.p = np.zeros((self.n_dem, self.n_com))
            for j in range(self.n_dem):
                self.p[j] = np.array([int(x) for x in f.readline().split()]).astype(int)
            self.c = np.zeros((n_warehouse, self.n_dem, self.n_com))
            for i in range(self.n_ware):
                for j in range(self.n_dem):
                    self.c[i, j] = np.array([int(x) for x in f.readline().split()]).astype(int)


if __name__ == "__main__":
    generate_instance(seed, n_ins)
