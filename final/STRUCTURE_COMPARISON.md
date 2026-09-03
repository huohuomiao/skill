# MLU Triton 四 Skill 最终结构

## 1. 最终确定

运行时只包含四个可调用 Skill 和一个只读共享资源层：

```text
mlu-triton-main
  -> mlu-triton-code-gen
       -> mlu-triton-code-review
  -> mlu-triton-optimize（balanced / max-performance）

share（不可调用，只保存共享资料、运行探针与 JSON 契约）
validation（工程验证工具，不属于第五个 Skill）
```

`correctness` 模式跳过 Optimize；`balanced` 和 `max-performance` 执行 Optimize。Main 是唯一外层状态写入者，其他 Skill 只写各自业务产物。

## 2. 四个 Skill 的原始目标、问题与最终修改

| Skill | 原始目标 | 没有充分发挥效果的原因 / 缺陷 | final 修改 | 兼容性措施 |
|---|---|---|---|---|
| `mlu-triton-main` | 串联环境、需求、生成、优化和总结 | 职责边界过于绝对，无法可靠校验下游；阶段完成与缓存没有验证等级；恢复信任主要依赖文件存在 | 统一运行清单、阶段 DAG、内容指纹、失效传播、原子缓存和 L1/L2/L3 晋级门禁 | 保留 EnvConfig、Extractor、KernelGen、Optimizer 和最终三个产物路径 |
| `mlu-triton-code-gen` | 把需求逐步变成 kernel、wrapper 和测试 | 六个细粒度 Agent 重复读取相同资料，分发次数和上下文开销大；步骤间交接易丢失语义 | 普通路径合并为 `DesignKernel -> BuildKernel -> Code Review`，Triton 快速路径只执行 `BuildKernel -> Code Review`；增加分组检查点和调度度量 | 继续生成 Step 1-7 原文件名和最终 `triton_code_fix.py` / `triton_report.md` |
| `mlu-triton-code-review` | 静态审查、真实运行、必要时修复 | 审查与修复链过长；直接通过仍可能产生额外分发；缺少机器可读的 L3 证据 | 直接通过不启动修复 Agent；失败时只启动一次 `ReviewAndFix`；新增 `review_result.json` 区分通过、修复、失败和基础设施错误 | 保留 `step6_test_code_fix.py` 和 Markdown 报告 |
| `mlu-triton-optimize` | 根据 MLU 实测结果选择优化策略 | 多策略容易重复执行、重置预算或把不可比结果当成改进；中断恢复不够严格 | 保留按需路由和全局预算，增加 `plan --resume` 一致性检查；只有 L1/L2/L3 齐全才发布优化缓存 | 保留 plan/state、策略目录、优化代码和报告格式 |

## 3. share 的职责

`share` 不是 Skill，不包含流程路由。它只保存需要被多个阶段读取、且应参与源指纹的数据：

- `share/contracts/`：Step 产物、Review、运行清单、缓存 metadata、优化和回归 Schema。
- `share/mlu/references/`：MLU 平台规则、primitive 和 libdevice 资料。
- `share/mlu/runtime/`：设备与工具链探针；探测结果永不缓存。
- `share/mlu/perf-analyzer/`：性能证据采集工具。
- `share/manifest.json`：共享资源分组及使用边界。

原先散落在 `validation/contracts` 的运行契约已并入 `share/contracts`。`validation` 只保留测试程序、策略和 fixture，从而避免“测试目录反向成为生产依赖”。

## 4. L1 / L2 / L3 验证

| 等级 | 证明内容 | 典型检查 | 可以晋级的缓存 |
|---|---|---|---|
| L1 | 结构和语法成立 | frontmatter、文件存在、Python AST、JSON/Schema、路径边界 | 不能单独发布阶段缓存 |
| L2 | 流程行为和控制逻辑成立 | 路由、产物不变量、预算、恢复、缓存损坏拒绝、回归正负例 | Extractor 可在 L1+L2 后发布；Design/Build 检查点可复用 |
| L3 | 当前 MLU 上动态结论成立 | 真实编译/执行、精度、必要的性能测量、硬件与工具链绑定 | KernelGen、Review 检查点和 Optimizer 需 L1+L2+L3 |

阶段缓存规则是：

```text
env_config : 不缓存，每次重新探测
extractor  : L1 + L2
kernel_gen : L1 + L2 + L3
optimizer  : L1 + L2 + L3
finalize   : 不缓存，每次根据本次清单重新汇总
```

Code Gen 分组检查点规则：`design`、`build` 需要 L1+L2；`review` 需要 L1+L2+L3。旧的 Step 1-7 检查点名仍兼容，其中 Step 7 才允许携带 L3。

## 5. 分阶段缓存与恢复

缓存 key 由输入内容、相关 Skill/共享资料的源快照、上游指纹、模式、预算以及必要的 `run_context` 组成。metadata schema version 2 同时记录：

- `stage_config_version`、阶段和指纹；
- 输入、依赖和源文件哈希；
- 规范化硬件/工具链上下文；
- 模式和预算哈希；
- 已完成的验证等级；
- 每个产物的路径、大小和 SHA-256。

缓存先写临时目录、复算哈希，再用原子替换发布。验证等级不足时阶段可以完成本次运行，但 `cache_key` 必须为空；恢复时会重新校验 metadata、验证等级和全部产物哈希。

## 6. 兼容与迁移结论

- 不改四个 Skill 名称，不新增第五个可调用 Skill。
- 不改既有业务产物文件名；新增文件均为旁路证据：`review_result.json`、`dispatch_metrics.json`、`run_manifest.json`。
- 历史六个 Code Gen 角色文档保留用于审计和度量，但不参与 final 活跃路径，也不进入当前 KernelGen 源指纹。
- v3/v4 的旧产物可作为人工参考；旧清单和旧缓存 metadata 不会直接被当成 final 的可信完成状态。
- 优化 plan/state 契约保持兼容，恢复时禁止重建状态来绕过预算。

## 7. 已实施顺序

1. 以 v4 的缓存、恢复和回归控制面为安全基线。
2. 移植合并后的 Design/Build 与 ReviewAndFix 路由。
3. 把共享契约归入 `share`，建立共享资源清单。
4. 将阶段配置升级为 version 2，加入 L1/L2/L3 和分组检查点门禁。
5. 扩展离线验证、缓存恢复、负向用例和调度开销检查。

详细执行命令见 [HOW_TO_VALIDATE.md](HOW_TO_VALIDATE.md)。
