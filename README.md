# 动力学标定服务（Kinetic Calibration Service）

存储每一釜间歇反应数据，用统一的方法拟合反应网络的阿伦尼乌斯参数（指前因子 A、活化能 Ea）。
技术栈：Flask + Python 3.12 + NumPy（积分器/优化器/统计量全部自写，不依赖 SciPy）+ PostgreSQL 16。

## 反应网络描述

```json
{
  "species": ["A", "B", "C"],
  "reactions": [
    {
      "name": "r1",
      "stoichiometry": {"A": -1, "B": 1},
      "orders": {"A": 1.0},
      "reversible": false,
      "reverse_orders": {}
    }
  ]
}
```

- `stoichiometry`：每个反应对各组分的净计量系数（反应物为负、产物为正）。
- `orders` / `reverse_orders`：正/逆反应对各组分的反应级数（允许分数、零）。
  缺省取化学计量：正向默认 `-min(stoich,0)`，逆向默认 `max(stoich,0)`。
- `reversible: true` 时该反应拥有独立的正、逆两个速率常数。
- 每个速率常数都满足 `k(T) = A · exp(-Ea / (R T))`，R = 8.314462618 J/(mol·K)。

## 运行

```bash
docker compose up --build
# 应用: http://localhost:8000  数据库: postgres:16-alpine
```

测试（需可访问的 PostgreSQL；默认读 `TEST_DATABASE_URL`，回退
`postgresql://kinetics:kinetics@127.0.0.1:5433/kinetics`）：

```bash
pip install -r requirements.txt
pytest
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/networks` | 创建反应网络（定义即校验） |
| GET  | `/api/networks/{id}` | 读取网络 |
| POST | `/api/networks/{id}/simulate` | 给定参数/T/初始浓度/时刻，积分并报告浓度、步数、误差 |
| POST | `/api/networks/{id}/datasets` | 在某网络下创建数据集 |
| GET  | `/api/datasets/{id}` | 数据集元信息 |
| POST | `/api/datasets/{id}/batches` | **新增一釜 → 产生新版本**（克隆旧批次后追加） |
| DELETE | `/api/datasets/{id}/batches/{index}` | **剔除一釜 → 产生新版本** |
| GET  | `/api/datasets/{id}/versions` | 列出所有版本（旧版本保留） |
| GET  | `/api/dataset-versions/{id}` | 某版本及其全部釜 |
| POST | `/api/dataset-versions/{id}/calibrate` | 在该版本上标定，结果绑定该版本 |
| GET  | `/api/calibrations/{id}` | 读取某次标定结果 |
| GET  | `/api/dataset-versions/{id}/calibrations` | 列出该版本上的标定 |

一釜数据形如：

```json
{
  "temperature": 300.0,
  "initial_concentrations": {"A": 1.0},
  "samples": [
    {"time": 0.5, "concentrations": {"A": 0.905, "B": 0.091}},
    {"time": 1.0, "concentrations": {"B": 0.172}}
  ]
}
```

允许每次只测部分组分（缺测即留空）。标定请求可带：

- `cold_start: true`：不用历史结果，从统一默认初值（k(T_ref)=1 min⁻¹、Ea=0）开始；
- `previous_calibration_id`: 显式指定拿哪次标定结果作初值；
- 缺省：若该数据集有更早版本，自动用最近版本的最近一次标定作热启动初值。

## 模拟方法

- 自适应 **Dormand–Prince RK5(4)**（FSAL），按缩放误差范数控制步长。
- 请求的每个输出时刻都是积分器的**精确着陆节点**（强制步边界），采样值即五阶节点值；
  节点之间用三次 Hermite 稠密输出（例如找 B 的峰值时刻）。
- 返回 `accepted_steps` / `rejected_steps` / `steps` 和 `estimated_error`
  （各接受步的最大缩放局部误差范数）。
- 若出现非有限状态且步长缩到最小仍失败，返回明确错误。

## 标定方法

### 参数化（解决 A–Ea 强相关）

直接拟合 (A, Ea) 时二者在阿伦尼乌斯指数里强耦合，雅可比接近奇异。改为对每个速率常数拟合

```
beta = [ln k_ref, gamma],  gamma = Ea / R,
ln k(T) = ln k_ref + gamma · (1/T_ref − 1/T)
```

- **参考温度 T_ref 取数据中不同温度的算术平均**（实验温窗中心）。在 T_ref 处
  `ln k_ref` 描述速率水平、`gamma` 描述温度敏感性，二者近似正交。
- 拟合完成后再换算回 A、Ea 报告；A、Ea 的近似标准误差由协方差经参数变换
  （delta 方法，含 ln k_ref 与 gamma 的协方差）得到。

### 优化器

- 自写 **Levenberg–Marquardt**（Marquardt 对角归一化阻尼，增益比驱动 λ 调整），
  收敛后再做一次 **Gauss–Newton 抛光**，保证无论冷热启动都停在同一最小二乘驻点。
- 雅可比用**解析灵敏度方程**（与状态联立的增广系统 `dS/dt = f_c S + f_β`），
  一次增广积分同时拿到残差与全部偏导；线搜索试探点只做廉价的纯状态积分。
- 停止条件：目标函数相对变化 `< 1e-10`，或迭代达 200 次。结果中
  `termination` 为 `converged` / `max_iterations`，`stop_reason` 给出具体原因。

### 可辨识性

- 单一温度 T0 时，数据只确定 `k(T0)`；gamma（即 Ea）方向的灵敏度结构性为零。
  服务只拟合 `ln k(T0)`，并在 `non_identifiable` 中对每个速率常数明确报告
  **Ea 与 A 均不可辨识**（任何落在该阿伦尼乌斯线上的 A/Ea 组合都拟合），
  `se_A/se_Ea` 为 `null`，`identifiable` 中 `A/Ea` 标志为 `false`
  （而 `ln_k_ref`、即 k(T0) 仍可辨识并准确报告），不会给一个任意值。
- 多温度时再检查列缩放后雅可比的数值秩（SVD，阈值 `max(m,n)·σ1·1e-10`）；
  秩亏时用列主元 QR 选出可辨识子空间重新拟合，并对无法确定的参数方向逐一说明原因。
- 报告：参数估计、标准误差、残差向量、RSS/RMSE、自由度、雅可比秩与奇异值、
  所用 T_ref 与迭代次数。

## 数据集版本化与持久化

- 数据集版本只增不改：新增/剔除一釜都克隆当前版本并产生新版本（version 递增、
  记录 parent），旧版本及其上的标定永不覆盖。
- 每次标定记录绑定 `version_id`（并记录用作初值的 `parent_calibration_id`）。
- 全部状态存 PostgreSQL，应用重启后版本与结果都在。

## 字段级校验（错误响应 400，`details` 逐项列出字段）

- 浓度为负或不是有限数：`initial_concentrations.<组分>` /
  `samples[i].concentrations.<组分>` / `parameters[i].A`；
- 温度不为正：`temperature`（釜级为 `batches[i].temperature`）；
- 取样时刻不递增：`samples[i].time`（模拟接口为 `times[i]`）；
- 反应/初值/测量引用未定义组分：`reactions[i].stoichiometry.<组分>` 等；
- 网络层面：重复组分/反应名、非有限或为零的计量系数、对不可逆反应给 `reverse_orders` 等。

## 已核对的性质（见 `tests/`）

- A→B→C 一级串联，任意时刻 `A+B+C = 初始总量`，误差 < 1e-8；
- k1=0.2、k2=0.1 时 B 的峰值时刻 ≈ 6.9315 min（ln2/0.1）；
- 无噪声多温度合成数据，A、Ea 还原相对误差 < 1e-4；
- 单一温度数据集报告 Ea 不可辨识，k(T0) 仍准确；
- 所有 k 放大 c 倍，曲线在 t/c 时刻与原曲线重合（时间轴压缩 1/c）；
- 同一版本冷启动与热启动结果参数相对差 < 1e-6。
