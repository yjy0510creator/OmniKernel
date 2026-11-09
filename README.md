# OmniKernel — 算子自动生成与跨硬件适配框架

> 第三届"兴智杯"全国人工智能创新应用大赛 · 算子自动生成与跨硬件适配赛道 参赛作品

以**算子 Spec（中间描述）为单一事实源**，通过 **LLM Agent + 模板/DSL 混合生成 + 编译-测试反馈闭环**，自动生成大模型推理核心算子，并同步适配**昇腾 NPU（AscendC/CANN）**与 **NVIDIA GPU（Triton/CUDA）**两类后端；全流程对接 **AISHPerf 3.0 算子生成智能体评测基准**完成编译正确性、数值精度、性能三维验证。

---

## 1. 核心特性

| 特性 | 说明 |
|---|---|
| **一次描述，多端生成** | 每个算子一份 `operator_specs/*.yaml`（数学定义/IO/容差/workload 矩阵），同时驱动 NPU 与 GPU 后端 |
| **正确性保底** | 模板库产出经过静态闸门（py_compile / 括号平衡 / 占位符残留检查）的确定性代码；LLM Agent 只做增强，失败自动回滚到模板版本 |
| **反馈闭环** | `语法闸门 → 编译 → 数值测试` 三级 gate，报错自动回喂 LLM 迭代修复（`--agent --max-rounds 3`） |
| **跨硬件抽象** | 后端差异全部收敛在模板层；spec/harness/AISHPerf 配置对两后端完全共享 |
| **量化推理导向** | W8A8 量化 GEMM、FlashAttention、SwiGLU 融合等直接对齐 AISHPerf"真实量化推理部署可替代性"评测重心 |
| **无 GPU 可复现** | 模板生成、静态校验、参考实现交叉核验在纯 CPU 环境即可完成（本仓库即如此验证通过） |

## 2. 代码结构

```
OmniKernel/
├── operator_specs/          # ★ 算子描述层（单一事实源，6 个 YAML）
│   ├── rmsnorm.yaml         #   含: 数学定义 / io / dtype_support /
│   ├── rope.yaml            #   tolerance(fp16/bf16/fp32 阈值) /
│   ├── softmax.yaml         #   workloads(AISHPerf 形状矩阵) /
│   ├── silu_mul.yaml        #   codegen.{triton|ascendc} 模板与特化参数
│   ├── flash_attn.yaml      #
│   └── quant_gemm_w8a8.yaml #
├── templates/               # ★ 模板层
│   ├── triton/    (6)       #   GPU: Triton kernel（tl.dot/在线softmax/int8累加）
│   ├── ascendc/   (6)       #   NPU: AscendC kernel（Cube-Vector 流水/UB 分块）
│   └── cuda/      (1)       #   CUDA C++ 参照实现（warp 归约示例）
├── omnikernel/              # ★ 生成引擎
│   ├── spec.py              #   spec 解析 + 特化参数推导(auto_pow2 等)
│   ├── engine.py            #   模板渲染 + 语法闸门 + 落盘
│   ├── llm.py               #   OpenAI 兼容 LLM 客户端（可选）
│   ├── feedback.py          #   三级 gate 与报错回喂 prompt 构造
│   ├── pipeline.py          #   CLI: python -m omnikernel gen/list
│   └── agent 逻辑在 pipeline.py 内（回滚保底策略见 §5.2）
├── validation/              # ★ 验证框架
│   ├── reference.py         #   6 算子 fp32 eager 金标准 + 输入生成器
│   ├── harness.py           #   精度(atol/rtol) + 延迟(中位) + 峰值显存
│   └── run_all.py           #   CLI: 按 spec 矩阵自动遍历 op×后端×dtype×shape
├── aishperf/                # ★ AISHPerf 3.0 对接工作区（草案）
│   ├── suites/ entities/ rules/ metrics/ run/
├── tests/
│   ├── test_reference.py    #   参考实现 vs PyTorch官方/整数精确金标准（7项）
│   └── test_generated.py    #   生成产物静态校验（18项）
├── generated/               # 生成产物（每次 gen 覆盖；已随包附最新一次产物）
├── results/                 # 验证 JSON 报告
├── build.sh                 # 一键：生成双后端 + CPU 自检
└── requirements.txt
```

## 3. 环境要求

| 项 | GPU 后端（Triton） | NPU 后端（AscendC） | 纯 CPU 自检 |
|---|---|---|---|
| OS | Linux | Linux（昇腾服务器） | 任意 |
| Python | ≥3.9 | ≥3.9 | ≥3.9 |
| 框架 | PyTorch ≥2.1 + Triton ≥2.2 + CUDA 12.x | CANN ≥8.0（含 atvc）+ torch_npu ≥2.1 | 仅 PyTorch + pyyaml |
| 硬件 | NVIDIA GPU（sm80/sm90） | 昇腾 910B 等 | 无需 |

```bash
pip install -r requirements.txt
```

LLM Agent 为**可选增强**，不配置则自动退化为纯模板模式，结果确定性可复现：

```bash
export OMNIKERNEL_LLM_BASE_URL=https://<openai-compatible-endpoint>/v1
export OMNIKERNEL_LLM_API_KEY=***
export OMNIKERNEL_LLM_MODEL=<model-name>
```

## 4. 快速开始

```bash
# ① 生成（模板模式，确定性）：全部 6 算子 × 双后端 -> generated/
bash build.sh                       # 或: python -m omnikernel gen --op all --backend triton,ascendc

# ② 无 GPU 自检（参考实现 + 阈值框架链路）
python -m validation.run_all --op all --backend cpu_ref --dtype fp16 --device cpu

# ③ 质量测试（本仓库全部通过，见 §7 实测状态）
python tests/test_reference.py      # 参考实现交叉核验 7/7
python tests/test_generated.py      # 产物静态校验 18/18

# ④ Agent 增强生成（需 LLM 配置 + 真机才有数值 gate）
python -m omnikernel gen --op rmsnorm --backend triton --agent --max-rounds 3
```

在真机上的完整评测：

```bash
# NVIDIA GPU
python -m validation.run_all --op all --backend triton --dtype fp16,bf16 --device cuda:0
# 昇腾 NPU（ascendc 后端经 torch_npu 扩展加载，接口同为 run()）
bash aishperf/run/run_eval.sh ascendc npu:0
```

## 5. 生成方法论

### 5.1 三段式流程

```
operator_spec.yaml ──► 模板特化渲染 ──► 三级 gate 验证 ──► generated/<op>/<backend>/kernel.*
   (单一事实源)      (确定性, 保底)    (语法→编译→数值)         │
                                    ▲                        │失败: 报错回喂
                                    └── LLM Agent 优化改写 ◄──┘ (可选, ≤3轮, 失败回滚模板版)
```

### 5.2 关键设计决策

1. **模板保底、Agent 增强**：LLM 输出必须先过语法/编译 gate 才替换候选；任一轮失败回滚上一可用版本。生成质量下限由模板决定，上限由 Agent+真机反馈决定。
2. **容差进 spec**：`atol/rtol` 按 dtype 分层写在 spec 内，验证框架、AISHPerf 规则、代码生成共享同一阈值——避免"各测各的"。
3. **量化算子精度口径**：W8A8 GEMM 以"同量化输入 + fp32 反量化 matmul"为参考，隔离 kernel 误差与量化算法误差（工业界通行做法）。
4. **无 GPU 可复现**：确定性模板模式 + CPU 静态闸门 + 参考实现交叉核验，评审在任意机器均可复现生成链路。

## 6. AISHPerf 基准验证

`aishperf/` 按官方开放工作区（`aishperf-caict/aishperf_openness`，3.0）的
suites/entities/rules 结构组织评测配置草案：

- **套件**：`suites/omnikernel_operator_gen.yaml` — 编译→数值→性能三阶段 + 跨后端一致性；
- **实体**：`entities/ascend-910b.yaml`、`entities/nvidia-a100.yaml`；
- **规则**：`rules/operator_gen_rule.yaml` — 阈值与 spec 对齐，含各算子性能基线定义；
- **执行**：`run/run_eval.sh` 一键跑 生成→验证→汇总 `summary.json`。

> 接入提示：字段以官方仓库最新 schema 为准，合并前先用官方 `validate` 校验一次。

## 7. 实测状态与数据回填表

**已在纯 CPU 环境真实验证（本仓库可直接复现）：**

| 验证项 | 结果 | 复现命令 |
|---|---|---|
| 6 算子 × 双后端生成 + 语法闸门 | ✅ 12/12 | `python -m omnikernel gen --op all --backend triton,ascendc` |
| 生成产物静态校验 | ✅ 18/18 | `python tests/test_generated.py` |
| 参考实现 vs 官方/精确金标准 | ✅ 7/7 | `python tests/test_reference.py` |
| 验证框架全算子遍历 | ✅ 14/14 | `python -m validation.run_all --backend cpu_ref --device cpu` |

**待评测机回填（真机数据才有效，严禁凭空填写）：**

| 算子 | 后端 | 精度通过率 | 中位延迟 µs | 基线 µs | 加速比 | 峰值显存 MB |
|---|---|---|---|---|---|---|
| rmsnorm | triton | __ | __ | __ | __ | __ |
| rmsnorm | ascendc | __ | __ | __ | __ | __ |
| rope | triton | __ | __ | __ | __ | __ |
| rope | ascendc | __ | __ | __ | __ | __ |
| softmax | triton | __ | __ | __ | __ | __ |
| softmax | ascendc | __ | __ | __ | __ | __ |
| silu_mul | triton | __ | __ | __ | __ | __ |
| silu_mul | ascendc | __ | __ | __ | __ | __ |
| flash_attn | triton | __ | __ | __ | __ | __ |
| flash_attn | ascendc | __ | __ | __ | __ | __ |
| quant_gemm_w8a8 | triton | __ | __ | __ | __ | __ |
| quant_gemm_w8a8 | ascendc | __ | __ | __ | __ | __ |

回填方法：评测机执行 `bash aishperf/run/run_eval.sh <backend> <device>` 后，
把 `aishperf/run/summary.json` 数值誊入（或将该文件作为证据一并提交）。

## 8. 常见问题

- **Windows 运行**：Triton 不支持 Windows，仅可跑 `build.sh` 中的生成/静态校验/CPU 自检部分；GPU 验证请在 Linux 评测机进行。
- **`--agent` 无效果**：检查 `OMNIKERNEL_LLM_*` 三个环境变量；未配置时日志会提示自动退化为模板模式（这是预期行为，不是错误）。
- **ascendc 数值验证**：需实现一个薄封装将 `generated/<op>/ascendc/kernel.cpp` 经 `atvc` 编译注册为 torch_npu 自定义算子，导出与 Triton 后端同签名的 `run()`；注册示例可参照 CANN 官方 `custom_op` 文档。

## 9. 许可证与致谢

本作品代码供大赛评审使用。评测体系参照中国信通院 AISHPerf 3.0（无问芯穹、清华大学技术支持）。
