# Kinetics Calibration Service

中试反应动力学标定后端：存批次实验数据、积分模拟浓度曲线、用统一方法从多釜数据
标定各基元反应的阿伦尼乌斯参数（指前因子 A、活化能 Ea），并对数据集做版本化
管理。

* **框架**：Flask（Python 3.12）、NumPy 数组运算
* **积分 / 优化 / 统计量全部自写**（Dormand–Prince RK5(4)、Levenberg–
  Marquardt、Cholesky、Gram–Schmidt 秩判定、协方差/标准误差），不依赖 SciPy
* **数据库**：PostgreSQL 16；另有一份行为相同的内存仓储用于测试
* **测试**：pytest

---

## 1. 反应网络描述

```jsonc
{
  "components": ["A", "B", "C"],
  "reactions": [
    {
      "id": "r1",
      "stoichiometry": {"A": -1, "B": 1},   // 计量系数
      "orders": {"A": 1},                    // 正反应对各组分的级数
      "reversible": false                    // 可选，默认 false
    },
    {
      "id": "r2",
      "stoichiometry": {"B": -1, "C": 1},
      "orders": {"B": 1}
    }
  ]
}
```

* 每个反应展开成一个或两个**通道（channel）**：不可逆反应只有正反应通道；
  可逆反应另有逆反应通道，逆反应级数可用 `reverse_orders` 显式给出，否则默认
  等于正反应产物的化学计量数。
* 通道速率：

  ```
  r_j = k_j(T) · Π_i c_i ^ order_ij
  k_j(T) = A_j · exp(-Ea_j / (R T))
  ```
* 组分物料衡算：`dc_i/dt = Σ_j ν_ji · r_j`。正、逆反应分成独立通道（而不是
  一个相减的净速率），在接近平衡时积分更稳，每个方向的参数也都能单独估计。

一釜批次记录包含：恒定温度、初始浓度（未列出的组分按 0）、一组严格递增的取样
时刻及该时刻测得的组分浓度；**每次不要求测全所有组分**。

---

## 2. 模拟

`POST /api/networks/{id}/simulate` 给定网络、温度、初始浓度、终止时刻和各通道
参数（`a` 与 `ea`，Ea 单位 J/mol；也接受 `ln_a`），积分给出浓度随时间变化。

积分器是带局部误差控制的 **Dormand–Prince RK5(4)**（经典 `ode45` 同系数）：

* FSAL 复用、PI 风格的步长伸缩（safety 0.9）、最小步长保护（区间长度 ×
  1e-14，步长塌缩时报错而不是静默挂死）；
* 五阶**连续扩展（dense output）**在任意取样时刻取值，因此标定里即使有几十
  个取样点，积分仍按自然大步长走，不被取样时刻切碎；
* 返回中报告 **实际接受步数 `n_steps`、被拒步数 `n_rejected` 和估计误差
  `max_scaled_local_error`**（所有已接受步中最大的缩放局部误差 RMS，正常
  应 ≤ 1）。

默认公差 `rtol=1e-9, atol=1e-12`；标定时自动收紧到 `1e-11 / 1e-14`，使有限
差分雅可比不被积分误差污染。

---

## 3. 标定方法

### 3.1 参数重参数化（解决 A 与 Ea 的强相关）

直接拟合 ln A 与 Ea/R 高度相关、条件数很差。本服务对每个通道改在

```
ln k(T) = q + θ · (1/T_ref − 1/T)
q = ln k(T_ref),   θ = Ea/R
```

这组坐标下拟合：截距是“参考温度下速率常数的对数”，斜率是活化能/R。

**参考温度的选择**：`1/T_ref = mean_batches(1/T)`（各釜倒温度的平均）。在这
个选择下，截距与斜率的叉积项 Σ z_b ≈ 0（z_b = 1/T_ref − 1/T_b），两参数近
似正交，雅可比条件数显著改善。标定结束后再精确换算回原参数报告：

```
Ea = R · θ
ln A = q + θ / T_ref          （A = exp(ln A)）
```

### 3.2 优化

* 目标函数：普通最小二乘 `SSE = Σ (模拟值 − 实测值)²`，逐残差给出；
* **Levenberg–Marquardt**，列缩放后求阻尼高斯–牛顿步（自写 Cholesky 分解与
  三角回代），残差雅可比由前向有限差分给出（ODE 积分公差比统计精度紧一个
  数量级，差分截断误差可忽略）；
* **确定性多起点**：以“特征时间倒数”为中心在对数轴上取 5 个网格点
  （1e-3…1e2），筛出 SSE 最小的 3 个跑 LM；调用方给的初值（例如上一版标定
  结果）作为额外候选起点。因此**无论冷启动还是拿上一版结果热启动，同一数据
  版本得到的参数相同**（测试要求相对差 < 1e-6，实际常为机器精度一致）；
* **停止条件**：最多 200 次 LM 迭代（所有起点共享一个预算），或相邻两次目标
  函数相对变化 < 1e-10 时停。结果中 `stop_reason` 明确给出
  `"converged"` / `"max_iterations"`。

### 3.3 不可辨识参数（不返回任意值）

* **结构不可辨识**：若所有釜在同一温度，z_b ≡ 0，雅可比对 θ 的列为零，活化
  能（进而 A）无法从数据确定。此时 Ea/A 报告为 `null`、标准误差为 `null`、
  并给出原因；该温度下的 k 及其标准误差照常报告。
* **数值不可辨识**：在解点对雅可比做列选主元 Gram–Schmidt 判定秩
  （阈值 1e-9），若有列与其他列线性相关，冻结该列、在可辨识子空间上重新拟合，
  同样以 `null` + 原因报告。
* 标准误差由终值雅可比的近似协方差给出：

  ```
  Cov = σ² (JᵀJ)⁻¹,   σ² = SSE / (n_obs − n_free)
  ```

  并按 delta 方法换算到 Ea（线性）和 A（`SE(A) ≈ A · SE(ln A)`，含
  q–θ 协方差交叉项）。

### 3.4 数据集版本化

* 数据集是不可变版本组成的链：`POST .../datasets` 建 v1；
  `POST /api/datasets/{id}/versions`（给出完整 batch_id 列表）新增/剔除釜
  即产生下一版本，版本号递增；
* 每个标定结果**绑定它所用的数据集版本 id**；旧版本、旧标定永久保留；
* 在新版本上标定默认拿该数据集**上一版标定结果作热启动初值**（可用
  `use_previous_result: false` 关掉，也可显式传 `initial_parameters`）。

---

## 4. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/networks` | 建反应网络 |
| GET  | `/api/networks/{id}` | 取网络 |
| POST | `/api/networks/{id}/batches` | 存一釜实验数据 |
| GET  | `/api/networks/{id}/batches` | 列釜 |
| GET  | `/api/batches/{id}` | 取一釜 |
| POST | `/api/networks/{id}/datasets` | 建数据集（v1） |
| POST | `/api/datasets/{id}/versions` | 产生新版本 |
| GET  | `/api/datasets/{id}` / `.../versions` | 数据集与版本列表 |
| GET  | `/api/dataset-versions/{id}` | 版本详情（含各釜数据） |
| POST | `/api/networks/{id}/simulate` | 积分模拟 |
| POST | `/api/dataset-versions/{id}/calibrations` | 在该版本上标定 |
| GET  | `/api/dataset-versions/{id}/calibrations` | 该版本的全部标定 |
| GET  | `/api/calibrations/{id}` | 取标定结果 |
| GET  | `/health` | 健康检查 |

校验失败统一返回 `400 {"error":"validation_failed","details":[{"field":...}]}`，
字段路径精确到出问题的位置，例如：

* 浓度为负 / 非有限数 → `initial_concentrations.A`、
  `samples[0].observations.B`
* 温度不为正 → `temperature`
* 取样时刻不递增 → `samples[2].time`
* 反应引用未定义组分 → `reactions[0].stoichiometry.Z` /
  `reactions[0].orders.Q`

### 快速试用（A→B→C 串联）

```bash
# 1) 建网络
curl -s localhost:8000/api/networks -H 'Content-Type: application/json' -d '{
  "components": ["A","B","C"],
  "reactions": [
    {"id":"r1","stoichiometry":{"A":-1,"B":1},"orders":{"A":1}},
    {"id":"r2","stoichiometry":{"B":-1,"C":1},"orders":{"B":1}}
  ]}'

# 2) 模拟（k1=0.2, k2=0.1 min^-1）；输出时刻包含 t=0、各取样时刻和 t_end
curl -s localhost:8000/api/networks/$NID/simulate -H 'Content-Type: application/json' -d '{
  "temperature": 300,
  "initial_concentrations": {"A": 1},
  "t_end": 25,
  "sample_times": [6.93, 10],
  "parameters": {"per_channel": [{"a": 0.2}, {"a": 0.1}]}}'
```

取样时刻允许包含 `t = 0`（视为初始状态测量）；批次记录中未列出的初始组分按 0
处理。

标定结果结构（节选）：

```jsonc
{
  "reference_temperature": 319.59,
  "internal_parameters": {"q": [...], "theta": [...]},
  "channels": [
    {"reaction_id":"r1","direction":"forward",
     "a": 676445.03, "a_se": ..., "ea": 40000.0, "ea_se": ...,
     "k_at_reference": 0.2, "identifiable": true,
     "unidentifiable_reason": null}
  ],
  "objective": {"sse": ..., "rmse": ..., "n_observations": 90,
                "n_free_parameters": 4, "residual_variance": ...},
  "iterations": 8, "max_iterations": 200,
  "stop_reason": "converged", "converged": true,
  "residuals": [{"batch_id":..., "time":..., "component":"B",
                 "observed":..., "predicted":..., "residual":...}]
}
```

---

## 5. 运行

### Docker Compose（应用 + PostgreSQL 16）

```bash
docker compose up --build
# 应用 http://localhost:8000 ；数据库数据落在命名卷 pgdata，重启后版本与结果都在
```

未设置 `DATABASE_URL` 时服务退化为内存仓储（仅适合本地试玩，重启不保留）。

### 本地开发

```bash
uv venv --python 3.12 .venv
.venv/bin/activate
pip install -r requirements.txt

# 可选：指向一个 PostgreSQL 16
export DATABASE_URL="host=localhost port=5432 user=kinetics password=kinetics dbname=kinetics"
flask --app app.app run --debug
```

### 测试

```bash
pytest                                  # 内存仓储全套
# 附加 PostgreSQL 16 持久化/跨后端一致性测试：
export TEST_POSTGRES_DSN="host=localhost port=5432 user=postgres dbname=kinetics_test"
pytest
```

测试覆盖了题面给出的全部核对关系：

1. A→B→C 1:1 串联，三组分之和恒等于初始总量（相对误差 < 1e-8）；
2. 一级串联 k1=0.2、k2=0.1、初始只有 A 时，B 在 ≈ 6.93 min 达到最大
   （0.01 网格上 6.93，峰值 0.5）；
3. 已知参数生成无噪声多温度数据，A、Ea 还原相对误差 < 1e-4（实际 ~1e-9）；
4. 单一温度数据集明确报告 Ea（及 A）不可辨识；
5. 所有速率常数放大 c 倍，浓度曲线在时间轴上压缩为 1/c；
6. 冷启动 / 热启动结果相对差 < 1e-6（实际机器精度一致）；
7. 负浓度、非有限数、非正温度、不递增时刻、未定义组分引用都报具体字段；
8. 版本化：旧版本与旧标定保留；重启（新建数据库连接模拟服务重启）后仍在；
9. 内存仓储与 PostgreSQL 仓储标定结果一致。
