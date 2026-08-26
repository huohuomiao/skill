# v3 与 v3_1 结构变化对照

## 1. 总览

v3_1 从 v3 完整复制后独立开发，保留原有 P0 修复、三种优化模式、静态按需路由、全局预算、Schema 和 L1/L2/L3 验证入口。

v3_1 把该能力正式标记为“阶段 3”，并关闭一个预算契约缺口：已有 `optimization_plan.json` / `optimization_state.json` 时，普通 `plan` 不再覆盖状态；只有兼容的 `plan --resume` 可以继续。

文件哈希对比：v3 和 v3_1 均为 81 个文件；新增 1 个、删除 1 个、修改 7 个，其余 73 个文件内容完全一致。新增/删除来自说明书由 `STAGE2_...` 更名为 `STAGE3_...`。

## 2. 文件变化

| 路径 | v3 | v3_1 |
|---|---|---|
| `mlu-triton-optimize/scripts/optimization_control.py` | 普通 plan 会覆盖已有状态 | 已有状态默认拒绝；兼容时显式 `--resume` 且不改写 state |
| `mlu-triton-optimize/SKILL.md` | 只描述生成 plan/state | 区分首次初始化与预算恢复，禁止删除/重建状态 |
| `mlu-triton-optimize/references/optimization-control.md` | 无恢复机器契约 | 增加 plan/state 完整性和兼容性检查 |
| `mlu-triton-main/SKILL.md` | 文本要求重试不能重置预算 | 明确首次 plan、恢复 `--resume` 和不兼容停止规则 |
| `validation/validate.py` | 路由、预算上限、patience 测试 | 增加恢复保留 usage/patience/history 与拒绝覆盖/不兼容/残缺状态测试 |
| `HOW_TO_VALIDATE.md` | v3/阶段 2 验证说明 | 更新为 v3_1/阶段 3，并增加手工恢复验证 |
| `STRUCTURE_COMPARISON.md` | v2/v3 对照 | 重写为 v3/v3_1 对照 |
| `STAGE2_OPTIMIZATION_MANUAL.md` | 阶段 2 说明书 | 删除（由阶段 3 说明书替代） |
| `STAGE3_OPTIMIZATION_MANUAL.md` | 无 | 新增阶段 3 说明书，补充恢复与迁移契约 |

## 3. 目录结构

```text
v3_1/
├── mlu-triton-main/
│   └── SKILL.md
├── mlu-triton-code-gen/
├── mlu-triton-code-review/
├── mlu-triton-optimize/
│   ├── SKILL.md
│   ├── scripts/
│   │   └── optimization_control.py
│   └── references/
│       └── optimization-control.md
├── validation/
│   ├── validate.py
│   ├── run_mlu_integration.py
│   ├── contracts/
│   ├── cases/
│   └── fixtures/
├── STAGE3_OPTIMIZATION_MANUAL.md
├── HOW_TO_VALIDATE.md
└── STRUCTURE_COMPARISON.md
```

## 4. 行为变化

| 场景 | v3 | v3_1 |
|---|---|---|
| 首次优化 | `plan` 创建 plan/state | 保持不变，返回 `resumed=false` |
| 已有完整状态再次普通 plan | 覆盖并把 usage 清零 | 退出码 2，要求显式 `--resume` |
| 同输入/模式/预算恢复 | 没有专用入口 | `--resume` 成功，state 字节内容不被脚本改写 |
| 输入改变后恢复 | 可能重新建状态 | 退出码 2，要求新输出目录 |
| 模式或预算改变后恢复 | 可能重新建状态 | 退出码 2，要求新输出目录 |
| plan/state 只剩一个 | 没有显式处理 | 恢复失败，禁止猜测或补建 |
| usage / best / patience / history | plan 重跑可能丢失 | 恢复必须完整保留 |

## 5. 保持不变

- `correctness`、`balanced`、`max-performance` 的选择逻辑不变。
- OOB 与高级策略的静态路由条件不变。
- 默认预算数值和预算覆盖格式不变。
- 每次 Subagent/Worker 前的 check/record 规则不变。
- 精度失败候选不得成为最终代码。
- Worker 仍必须在当前 `JOB_ID` 中阻塞、串行执行。
- 真实精度与性能结论仍必须来自可用 MLU 环境。

## 6. 未引入的 v4 能力

v3_1 不包含 v4 的跨运行内容寻址缓存、Code Gen 内部检查点、运行清单和持续回归比较器。它只完成阶段 3 的按需优化、全局预算及预算安全恢复。
