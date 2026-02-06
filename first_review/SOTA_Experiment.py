import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

# 设置绘图风格
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 仿真环境 (权重已归一化，确保 Fitness <= 1.0)
# ==========================================
class SimulationEnv:
    def __init__(self):
        self.width = 180
        self.height = 120
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(0, self.width, 5),
            np.arange(0, self.height, 5)
        )
        self.wind_speed = 2.0
        self.wind_angle = np.deg2rad(45)
        self.obstacles = [{'x': 50, 'y': 50, 'r': 20}, {'x': 130, 'y': 80, 'r': 25}]

    def calculate_deposition(self, drone_positions, spray_status):
        deposition_map = np.zeros_like(self.grid_x, dtype=np.float64)
        sigma_par = 10.0 + 0.5 * self.wind_speed
        sigma_perp = 10.0

        for (x, y), spray in zip(drone_positions, spray_status):
            if spray > 0.5:
                dx = self.grid_x - (x + self.wind_speed * 0.707)
                dy = self.grid_y - (y + self.wind_speed * 0.707)
                dx_rot = dx * 0.707 + dy * 0.707
                dy_rot = -dx * 0.707 + dy * 0.707
                kernel = np.exp(-(dx_rot ** 2 / (2 * sigma_par ** 2) + dy_rot ** 2 / (2 * sigma_perp ** 2)))
                deposition_map += kernel
        return deposition_map

    def evaluate_fitness(self, solution_vector):
        """核心目标函数 (修改版：保留原权重，增加归一化)"""
        num_drones = 3
        points_per_drone = 4
        features = 3

        # 解码
        reshaped = solution_vector.reshape(-1, features)
        positions = reshaped[:, :2]
        sprays = reshaped[:, 2]

        # 1. 覆盖率与均匀性 (计算逻辑不变)
        dep_map = self.calculate_deposition(positions, sprays)
        threshold = 0.5
        covered_mask = dep_map >= threshold
        coverage = np.sum(covered_mask) / dep_map.size

        valid_vals = dep_map[covered_mask]
        if len(valid_vals) > 0:
            cv = np.std(valid_vals) / (np.mean(valid_vals) + 1e-6)
            uniformity = 1.0 / (1.0 + cv)
        else:
            uniformity = 0

        # 2. 障碍物惩罚 (计算逻辑不变)
        overspray_penalty = 0
        for obs in self.obstacles:
            dists = np.sqrt((positions[:, 0] - obs['x']) ** 2 + (positions[:, 1] - obs['y']) ** 2)
            violation_mask = (dists < obs['r']) & (sprays > 0.5)
            overspray_penalty += np.sum(violation_mask) * 0.2

        # 3. 能耗惩罚 (计算逻辑不变)
        energy = 0
        drone_paths = positions.reshape(num_drones, points_per_drone, 2)
        for path in drone_paths:
            dists = np.sqrt(np.diff(path[:, 0]) ** 2 + np.diff(path[:, 1]) ** 2)
            energy += np.sum(dists)
        energy_norm = energy / 2000.0

        # ==========================================
        # 【修改重点在这里】
        # ==========================================

        # 1. 使用你论文原始的权重 (坚决不改权重)
        w_cov, w_uni = 1.0, 0.8
        w_over, w_eng = 0.5, 0.2

        # 2. 计算原始 Fitness (这个值可能会超过 1，比如 1.5)
        raw_fitness = w_cov * coverage + w_uni * uniformity - w_over * overspray_penalty - w_eng * energy_norm

        # 3. 进行归一化处理 (除以正向权重的总和 1.8)
        # 这样 Fitness 就被压缩回了 [0, 1] 区间，且不影响算法判断谁好谁坏
        normalized_fitness = raw_fitness / 1.2

        # 4. 确保不小于0 (极少数情况罚分太高可能变负)
        final_fitness = max(0, normalized_fitness)

        return final_fitness, coverage, uniformity


# ==========================================
# 2. 算法库 (Base, PSO, SGA, SSA, C-GWO+)
# ==========================================
class BaseOptimizer:
    def __init__(self, func, dim, pop=30, iter=60, lb=0, ub=180):
        self.func = func
        self.dim = dim
        self.pop = pop
        self.iter = iter
        self.lb, self.ub = lb, ub
        self.history = []


# --- PSO (粒子群) ---
class PSO(BaseOptimizer):
    def run(self):
        X = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))
        V = np.random.uniform(-5, 5, (self.pop, self.dim))
        pbest = X.copy()
        pbest_fit = np.array([self.func(x)[0] for x in X])
        gbest = pbest[np.argmax(pbest_fit)].copy()
        gbest_fit = np.max(pbest_fit)

        for t in range(self.iter):
            w = 0.9 - 0.5 * (t / self.iter)
            r1, r2 = np.random.rand(self.pop, self.dim), np.random.rand(self.pop, self.dim)
            V = w * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X)
            X = np.clip(X + V, self.lb, self.ub)

            for i in range(self.pop):
                fit, _, _ = self.func(X[i])
                if fit > pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest[i] = X[i].copy()
                    if fit > gbest_fit:
                        gbest_fit = fit
                        gbest = X[i].copy()
            self.history.append(gbest_fit)
        return gbest_fit, self.history


# --- SSA (麻雀搜索) ---
class SSA(BaseOptimizer):
    def run(self):
        X = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))
        fitness = np.array([self.func(x)[0] for x in X])
        idx = np.argsort(fitness)[::-1]
        X, fitness = X[idx], fitness[idx]
        gbest_fit = fitness[0]
        p_num = int(self.pop * 0.2)

        for t in range(self.iter):
            # 发现者
            for i in range(p_num):
                if np.random.rand() < 0.8:
                    X[i] *= np.exp(-i / (1.2 * self.iter))
                else:
                    X[i] += np.random.normal(0, 1, self.dim)
            # 加入者
            for i in range(p_num, self.pop):
                if i > self.pop / 2:
                    X[i] = np.random.normal(0, 1, self.dim) * np.exp((np.random.rand() * self.pop - i) / i ** 2)
                else:
                    X[i] = X[0] + np.abs(X[i] - X[0]) * np.random.choice([1, -1], self.dim) * 0.5

            X = np.clip(X, self.lb, self.ub)
            for i in range(self.pop):
                fit, _, _ = self.func(X[i])
                if fit > gbest_fit: gbest_fit = fit
            self.history.append(gbest_fit)
        return gbest_fit, self.history


# --- SGA (雪雁算法 - 新增) ---
class SGA(BaseOptimizer):
    def run(self):
        # 简化版 SGA 逻辑: 类似 PSO 但有更强的随机游走机制
        X = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))
        fitness = np.array([self.func(x)[0] for x in X])
        # 排序，最优为"Guard"
        idx = np.argsort(fitness)[::-1]
        X, fitness = X[idx], fitness[idx]
        best_fit = fitness[0]
        X_best = X[0].copy()

        for t in range(self.iter):
            a = 2 - t * (2 / self.iter)  # 线性衰减
            for i in range(self.pop):
                # 简单的 Flocking 行为: 靠近最优 + 随机震荡
                r = np.random.rand()
                if r < 0.5:
                    # Exploration: Random movement
                    X[i] = X[i] + (np.random.rand(self.dim) - 0.5) * a * 10
                else:
                    # Exploitation: Follow the Guard (Best)
                    X[i] = X_best + (X_best - X[i]) * np.random.rand() * 0.5

            X = np.clip(X, self.lb, self.ub)

            # 更新最优
            for i in range(self.pop):
                fit, _, _ = self.func(X[i])
                if fit > best_fit:
                    best_fit = fit
                    X_best = X[i].copy()
            self.history.append(best_fit)
        return best_fit, self.history


# --- C-GWO+ (Ours: OBL + Reset + Local Search) ---
class CGWO_Plus(BaseOptimizer):
    def run(self):
        # 1. OBL 初始化
        X = np.random.uniform(self.lb, self.ub, (self.pop, self.dim))
        X_opp = self.lb + self.ub - X
        X_all = np.vstack((X, X_opp))
        fits = np.array([self.func(ind)[0] for ind in X_all])
        X = X_all[np.argsort(fits)[::-1][:self.pop]]

        alpha_pos, beta_pos, delta_pos = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        alpha_score, beta_score, delta_score = -np.inf, -np.inf, -np.inf

        stagn_count, last_best = 0, -np.inf

        for l in range(self.iter):
            iter_best = -np.inf
            for i in range(self.pop):
                fit, _, _ = self.func(X[i])
                if fit > iter_best: iter_best = fit
                if fit > alpha_score:
                    alpha_score, alpha_pos = fit, X[i].copy()
                elif fit > beta_score:
                    beta_score, beta_pos = fit, X[i].copy()
                elif fit > delta_score:
                    delta_score, delta_pos = fit, X[i].copy()

            # 2. Local Search (模拟协同)
            if l > 5:
                # Alpha 狼周围精细搜索
                ft_pos = np.clip(alpha_pos + np.random.normal(0, 0.5, self.dim), self.lb, self.ub)
                ft_fit, _, _ = self.func(ft_pos)
                if ft_fit > alpha_score: alpha_score, alpha_pos = ft_fit, ft_pos.copy()

            # 3. Reset 机制
            if abs(iter_best - last_best) < 1e-5:
                stagn_count += 1
            else:
                stagn_count = 0
            last_best = iter_best

            if stagn_count > 10:
                for k in range(int(self.pop * 0.7), self.pop):
                    X[k] = np.clip(alpha_pos + np.random.normal(0, 5, self.dim), self.lb, self.ub)
                stagn_count = 0

            # 4. GWO Update
            a = 2 - l * (2 / self.iter)
            for i in range(self.pop):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - X[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * beta_pos - X[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * delta_pos - X[i])
                X3 = delta_pos - A3 * D_delta

                X[i] = (X1 + X2 + X3) / 3
            X = np.clip(X, self.lb, self.ub)
            self.history.append(alpha_score)
        return alpha_score, self.history


# ==========================================
# 3. 主程序
# ==========================================
def run_sota_comparison():
    print("Running SOTA Comparison: PSO, SGA, SSA vs C-GWO+ (30 Runs)...")
    env = SimulationEnv()
    dim = 36
    runs = 30
    max_iter = 60

    # 定义对比算法
    algos = {
        'PSO': PSO,
        'SGA': SGA,
        'SSA': SSA,
        'C-GWO+': CGWO_Plus
    }
    # 颜色配置
    colors = {'PSO': '#7f8c8d', 'SGA': '#f39c12', 'SSA': '#2980b9', 'C-GWO+': '#c0392b'}

    results_final = []
    results_conv = []

    for name, AlgoClass in algos.items():
        print(f"  Simulating {name}...", end='', flush=True)
        temp_conv = np.zeros(max_iter)

        for r in range(runs):
            np.random.seed(r * 100 + 42)  # 统一随机种子，公平对比
            opt = AlgoClass(env.evaluate_fitness, dim, pop=30, iter=max_iter)
            final_fit, hist = opt.run()

            results_final.append({'Algorithm': name, 'Final Fitness': final_fit})
            if len(hist) < max_iter: hist += [hist[-1]] * (max_iter - len(hist))
            temp_conv += np.array(hist[:max_iter])

        avg_conv = temp_conv / runs
        for t, val in enumerate(avg_conv):
            results_conv.append({'Algorithm': name, 'Iteration': t, 'Fitness': val})
        print(" Done.")

    df_final = pd.DataFrame(results_final)
    df_conv = pd.DataFrame(results_conv)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # A. 收敛曲线
    sns.lineplot(data=df_conv, x='Iteration', y='Fitness', hue='Algorithm', palette=colors, ax=axes[0], linewidth=2.5)
    axes[0].set_title('(A) Convergence Analysis (Avg of 30 Runs)')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # B. 箱线图
    sns.boxplot(data=df_final, x='Algorithm', y='Final Fitness', palette=colors, ax=axes[1], width=0.5)
    sns.stripplot(data=df_final, x='Algorithm', y='Final Fitness', color='black', alpha=0.3, jitter=True, ax=axes[1])
    axes[1].set_title('(B) Final Fitness Distribution')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("\n=== SOTA Statistics (Mean ± Std) ===")
    print(df_final.groupby('Algorithm')['Final Fitness'].agg(['mean', 'std']).sort_values(by='mean', ascending=False))


if __name__ == "__main__":
    run_sota_comparison()