# LLM Inference Infrastructure 面试笔记

这份笔记面向有模型训练经验、准备 inference infrastructure / model serving 岗位的工程师。
从熟悉的计算、显存与通信分析出发，建立面向延迟、吞吐、成本和可靠性的推理系统设计方法。
资料核对日期：2026-10-25。
标为“假设算例”或“练习”的数字用于推导与设计讨论，不是实测性能；部署时应结合目标模型、硬件与软件版本验证。

## 1. 从训练视角切换到 serving

训练经验可以迁移到算子、显存和通信分析；答 serving 题时先明确用户体验与负载。
建议按“需求 → 工作负载 → 容量估算 → 调度 → 分布式 → 故障 → 验证”展开。
不要用一个峰值 tokens/s 概括系统能力；先确定计费、延迟、质量和成本口径。

自回归推理分为处理输入的 prefill，以及依赖已生成 token 的 decode。
Prefill 可并行处理 prompt 位置；普通 decode 每条序列每步推进一个 token。
两阶段计算形态不同，混合调度可能互相干扰。[SARATHI](https://arxiv.org/abs/2308.16369)

**设计练习：**分别描述以下三个流量池，不先指定硬件品牌。

- 交互聊天：短到中等 prompt，关注首 token 与流式停顿。
- 长文问答：长 prompt、较短输出，关注输入处理与 prefix reuse。
- 离线生成：输出较长，可容忍排队，关注单位成本和完成期限。

建议补问：到达率、突发度、输入/输出长度联合分布、并发、缓存共享率和租户隔离。
质量约束包括模型版本、采样策略、结构化输出以及量化后的任务表现。
把未知量写成假设，再说明取得 trace 后如何修正。

## 2. 指标：先定义时间边界

以下采用本文统一测量口径；工具定义不完全相同时必须显式映射。
令请求在客户端发出于 t0，收到首个输出 token 于 t1，收到第 i 个 token 于 ti。
令实际输出数量为 N，完成于 tN；流式 chunk 未必等于一个 token。

| 指标 | 本文定义 | 使用时的注意点 |
| --- | --- | --- |
| TTFT | t1 − t0 | 包括客户端所见网络与排队；注明是否计入 tokenization |
| ITL | ti − t(i−1)，i ≥ 2 | 是间隔分布；关注长停顿和尾部 |
| TPOT | (tN − t1)/(N−1)，N > 1 | 请求内部平均值；N=1 时记为不适用 |
| E2E latency | tN − t0 | 依赖输出长度，不能脱离长度比较 |
| Throughput | 完成请求数/s 或输出 tokens/s | 输入/输出 token 分开报告 |
| Goodput | 满足约定 SLO 的成功完成请求数/测量时长 | 明确是请求还是 token 口径 |

vLLM 提供 TTFT、请求阶段耗时等指标；客户端与 engine 的时钟边界应分别记录。
不能把服务端直方图直接当客户端端到端体验。[vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)

DistServe 以同时满足 prefill/decode 延迟要求的服务能力为目标。
本文 goodput 进一步明确成功条件及分母；不要把论文所有倍率迁移到别的负载。
[DistServe](https://arxiv.org/abs/2401.09670)

**推导练习：**首 token 0.3 s，随后 100 个间隔各 0.02 s。
输出共 101 tokens，TPOT=20 ms，E2E=2.3 s；TTFT 不是 2.3 s。
某个间隔 500 ms 可能被均值稀释，所以必须单独看 ITL 尾部。
建议分别报告请求 TPOT 的 p99 与所有 token ITL 的 p99，不能混称。

## 3. Roofline、batching 与容量估算

Arithmetic intensity I = FLOPs / HBM bytes。
Roofline 估算为 achieved FLOPs/s ≤ min(F_peak, BW_HBM × I)。
小算子还可能受 launch、并行度或同步限制；两项上界都不是实测承诺。
[NVIDIA GPU performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

**简化模型与自行推导：**dense 权重参数量 P，权重每参数 s bytes，batch B。
忽略 attention、激活和通信，单步矩阵乘约 2PB FLOPs，权重读取约 Ps bytes。
若权重跨 batch 复用，则 I≈2B/s；BF16 s=2 时约 B FLOPs/byte。
这解释增加 batch 为什么可能改善权重摊销，不能据此宣称端到端线性加速。
长上下文 decode 还要读大量 KV；MoE 应区分总参数、激活参数及实际访存。

**假设算例：**P=8×10^9，s=2，假设持续有效 HBM 带宽 2×10^12 bytes/s。
仅权重读取下界：16×10^9 / (2×10^12)=0.008 s=8 ms/step。
B=1 时对应理想化输出上界 125 tokens/s；B=16 时为 2000 tokens/s。
这是忽略其他成本的模型上界，不是任何硬件或模型的 benchmark。
若 B=16，则 I≈16 FLOPs/byte；仍需与该精度下的计算/带宽比比较。

建议用 profiler 判断时间花在 GEMM、attention、collective、CPU 还是空闲间隙。
Prefill 往往更容易形成大矩阵乘；decode 低 batch 利用率问题见原始研究。
不能把“prefill compute-bound、decode memory-bound”当适用于所有形状的定律。
[SARATHI](https://arxiv.org/abs/2308.16369)

## 4. KV cache：公式、GQA 和核对算例

标准 full-attention、各层同形状、无压缩且 K/V 同精度时：

```text
M_KV = 2 × L × H_kv × D_head × s_kv × Σ_j T_j  bytes
```

2 代表 K 与 V；L 为层数；Tj 是第 j 条序列实际存储的 token 数。
GQA 中多个 query heads 共享 KV heads，公式用 H_kv，不用 H_q。
MHA 的 H_kv=H_q；MQA 的 H_kv=1。[GQA paper](https://arxiv.org/abs/2305.13245)
公式是按 K/V 张量元素计数的本文推导；不含 allocator、padding 和元数据。

**已核对的假设算例，不对应某个命名模型：**

- L=32，H_q=32，H_kv=8，D_head=128，BF16 KV 即 s_kv=2 bytes。
- 每 token：2×32×8×128×2=131072 bytes=128 KiB。
- 每序列存 8192 tokens：131072×8192=1073741824 bytes=1 GiB。
- 16 条这样的独立序列：16 GiB KV，尚未加权重与运行时显存。
- 若只改成 MHA 的 H_kv=32，KV 为 4 倍：每序列 4 GiB。
- 若采用 1-byte KV，纯元素存储为 0.5 GiB/序列；另计 scales 与格式开销。

GB=10^9 bytes；GiB=2^30 bytes；不要混用容量单位。
T 是当前被缓存的长度，不是无条件用 max context，也不是只算 prompt。
页块取整时可用 Σ ceil(Tj/K)×K 替代 ΣTj，K 为 block tokens。
这只是容量近似；共享 block 需去重，不能对每条序列重复计费。

TP 后每卡 KV 不一定等于总 KV/TP：取决于 KV heads 的划分或复制。
建议逐层检查实际配置与布局；MLA、滑动窗口及混合层模型需重新计数。
DeepSeek-V3 的 MLA 是不同缓存结构的公开实例，不能直接套上述 GQA 公式。
[DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)

容量预算建议写成：权重 + 活跃 KV + workspace/activations + runtime + 安全余量。
**练习：**80 GiB 可用预算，假设权重16、runtime/workspace8、余量8 GiB。
则 KV 剩48 GiB，按上述每条1 GiB可容纳48条；这是静态容量上界。
实际 admission 还须考虑输出增长、不同长度、页块取整和 SLO。

## 5. Paged KV 与 prefix caching

PagedAttention 将逻辑 KV blocks 映射到非连续物理块，降低连续分配的浪费。
共享块与 copy-on-write 可支持复用或分支；页管理不会自动减少 attention 算术量。
块大小仍有尾块浪费与管理开销的取舍。[PagedAttention](https://arxiv.org/abs/2309.06180)

Prefix caching 复用匹配输入前缀的 KV，主要省重复 prefill。
它不免除后续 decode 对历史状态的 attention，也不保证输出生成更快。
[vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

缓存键必须对应完整计算语义；vLLM 的设计包含 parent hash、tokens 和额外标识。
文档也提供 cache salt 来限制不同信任组的共享。
[vLLM prefix cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/)

**设计建议：**模型 revision、adapter、位置配置及多模态输入都要纳入兼容性判断。
租户隔离应由服务端可信身份决定；不要让用户自行宣称可访问别人的缓存域。
分别记录 token hit rate、节省的 prefill 时间与缓存占用，避免只优化命中率。
低复用的长前缀可能挤掉高价值前缀；练习比较 LRU 与成本感知淘汰。

## 6. Continuous batching 与 chunked prefill

Iteration-level scheduling 可在模型迭代边界改变 batch，释放已完成请求并加入新工作。
其目标之一是避免整批等待最长请求结束；不是每个 kernel 都必须重新调度。
[Orca](https://www.usenix.org/conference/osdi22/presentation/yu)

Chunked prefill 将长输入处理拆成多个调度块，给 decode 留出执行机会。
块越小不一定越好：需权衡单步耗时、计算效率及 prompt 完成时间。
[SARATHI](https://arxiv.org/abs/2308.16369)

**建议的调度输入：**本轮 token budget、KV free blocks、请求 deadline 和公平性状态。
不要把 max sequences 与本轮 token budget 当成同一个控制量。
Decode priority 可以保护流式间隔，但需要 aging 或配额避免 prefill 饥饿。
输出长度未知时，采用保守额度或增长预测，并为误差准备拒绝/抢占策略。

**设计练习：**活跃 decode 有64条，另来一条32k prompt。
先定允许的 decode 间隔，再扫描 chunk size，观察 TTFT、ITL、goodput。
建议记录每轮 prefill/decode tokens 和轮次耗时，以定位长步阻塞。
不要声称“开启 chunking 必然同时降低所有延迟”。

## 7. Attention kernels 与执行开销

FlashAttention 用 tiling 和在线归约减少 HBM 读写，避免显式物化完整注意力矩阵。
它是 exact attention 的实现优化，不把 dense attention 的二次计算复杂度变成线性。
这里 exact 指算法语义，浮点执行不保证逐 bit 相同。[FlashAttention](https://arxiv.org/abs/2205.14135)

PagedAttention 主要解决 KV 布局和管理；FlashAttention 主要解决 attention 的 IO。
两者不是同义词，也不应简单当作互斥选项。
[PagedAttention](https://arxiv.org/abs/2309.06180)、[FlashAttention](https://arxiv.org/abs/2205.14135)

**排查建议：**分别 profile prefill 与 decode，不拿训练 attention 的速度替代服务结果。
检查 head size、dtype、mask、KV layout、长上下文与小 batch 的实际 kernel 路径。
若 CPU launch 占主导，再研究 fusion、CUDA Graph 和 shape bucketing。
评估 graph 时同时记录捕获/编译启动时间、额外显存、动态形状回退率。
这是建议的验证顺序，不预设所有 serving engine 的执行方式。

## 8. TP / PP / DP / EP：画出通信路径

TP 切分层内计算；Megatron 展示了线性层分片与通信的基本组合。
这篇论文研究训练，本文只借其并行分解方法，不把训练扩展率当推理数据。
[Megatron-LM](https://arxiv.org/abs/1909.08053)

| 方式 | 切分对象 | 面试应解释的成本 |
| --- | --- | --- |
| TP | 层内张量 | 层间同步、collective 延迟、较小局部 GEMM |
| PP | 模型层 | stage 传输、负载不均、pipeline bubbles |
| DP | 服务副本/请求 | 更多权重副本、路由和缓存局部性 |
| EP | MoE experts | token dispatch/combine、热点 expert、网络与负载均衡 |

vLLM 公开支持组合 TP/PP，以及 MoE 场景的 DP attention 与 expert 层并行。
具体支持矩阵与约束应以部署版本为准。[vLLM parallelism](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
EP 与跨节点 all-to-all 的现实例子可见公开报告；不推断其他公司的部署。
[DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437)

AllReduce 让每个 rank 得到归约结果；ReduceScatter 分发结果分片。
AllGather 汇集各 rank 分片；所有 rank 的 collective 调用参数必须匹配。
[NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

**简化通信推导：**假设 ring all-reduce，n ranks，每 rank 输入 m bytes。
理想每 rank 发送约 2(n−1)m/n bytes，接收量相同；该式不是发送与接收之和。
时间还需加分阶段 latency 和拥塞。
这是指定算法的模型，不代表 NCCL 永远选 ring。
面试建议写 t≈α×steps + bytes/BW_effective，再讨论 overlap 能隐藏多少。
单请求串行经过 PP stages，不应承诺 PP 按卡数等比例降低单 token 延迟。
练习优先比较“最小可装载 TP + 多副本”与“大 TP 单副本”的 SLO/成本曲线。

## 9. Quantization：区分权重、激活和 KV

AWQ 是 activation-aware 的低比特 weight-only quantization；校准利用激活统计。
权重4-bit不等于激活或KV也4-bit，格式、packing与kernel同样影响结果。
[AWQ](https://arxiv.org/abs/2306.00978)

SmoothQuant 用等价缩放迁移激活离群值的量化难度，以支持 W8A8。
不要把方法在论文模型上的质量结论推广到任意模型与业务。
[SmoothQuant](https://arxiv.org/abs/2211.10438)

**建议评估矩阵：**BF16 baseline、weight-only、weight+activation、KV quantization。
每种记录权重显存、KV容量、实际kernel、TTFT/ITL/goodput与任务质量。
校准覆盖长上下文、代码、数学及业务语言；单一 perplexity 不作为完整准入标准。
量化可能节省带宽，但 dequant、scale 与不匹配硬件也可能吞掉收益。
后一句是工程风险假设，应在目标设备测量后下结论。
发布建议固定模型与量化工件版本，保留质量阈值和回滚路径。

## 10. Speculative decoding：正确性先于速度

Draft q 提议 token，target p 并行验证；采样时按 min(1,p(x)/q(x)) 接受。
首次拒绝时从归一化的 max(p−q,0) 采样；全部接受再从 target 多采一个。
在算法假设下保持 target 分布，不保证相同随机种子得到相同序列。
[Speculative decoding, Algorithm 1](https://arxiv.org/html/2211.17192v2)

p/q 应对应实际采样变换后的分布；随意阈值接受不具有上述保证。
Greedy 验证与 stochastic rejection sampling 应分开解释。
**简化推导练习：**接受率独立同分布为 a，draft 长 k。
每轮期望推进量 Σ(i=0..k) a^i；a=0.8、k=4 得3.3616 tokens。
[同论文分析与假设](https://arxiv.org/html/2211.17192v2)

若自设 draft 总耗时4 ms、验证12 ms，普通 target 单步10 ms：
估计 speedup=3.3616×10/(4+12)=2.101；全部时间都是练习假设。
建议实测 accepted tokens/round、验证耗时、draft成本及额外KV。
高并发下争抢计算资源可能抵消收益；不能只报告接受率。

## 11. Disaggregated prefill/decode

DistServe 将两阶段放在不同 GPU，按各自延迟约束规划资源与并行度。
其收益来自减少干扰与独立配置；同时必须考虑 KV 传输带宽。
[DistServe](https://arxiv.org/abs/2401.09670)
Splitwise 也研究 phase splitting 与不同硬件资源的组合；属于公开方案实例。
[Splitwise](https://arxiv.org/abs/2311.18677)

**设计练习：**上面1 GiB KV，假设单流有效传输25 GB/s，则耗时下界约42.95 ms。
计算为 1073741824/(25×10^9) s；未计RPC、排队、重排、争用与重传。
若32个请求同用同一链路，总字节为32 GiB；不能每个都假设独占带宽。
建议明确 first token 由哪个池发出、何时转移所有权，以及失败是否重做 prefill。
缓存布局/精度、模型版本和 TP 划分必须兼容，或显式支付转换成本。
比较基线应包括 colocated + chunked prefill；拆分不是无条件升级。

## 12. 路由、admission、取消和可靠性

以下是设计建议；实现前需用容量模型与故障注入验证。
建议链路：API/auth → tokenization/limits → admission → router → scheduler → workers。
路由同时考虑排队工作量、活跃KV、预期长度和缓存收益，不只看请求数。
Cache affinity 应有拥塞惩罚，避免所有相同前缀流量压到一个热点副本。
Admission 同时约束排队时延、token预算、KV增长与租户份额。
超载时尽早拒绝并明确重试条件；盲目重试会增加系统负载。

Ray Serve 文档说明客户端断连/超时后的取消传播及异步取消处理。
取消 HTTP handler 不等于已经释放 GPU 侧状态，必须验证 engine 路径。
[Ray Serve HTTP/cancellation](https://docs.ray.io/en/latest/serve/http-guide.html)

建议用 request ID 把 disconnect 传到 scheduler，在安全迭代边界终止请求。
释放 KV 引用、队列额度和推测分支；检查异步任务与缓存传输是否残留。
流式已发部分结果后，不应把重新采样伪装成无缝续传；定义中断协议。
TP rank 故障可能影响整组；建议按并行组恢复，避免幸存 rank 永久等待。
发布先 warmup，再 readiness；drain 停止新流量并给已有请求有界退出时间。

建议指标：client TTFT/ITL/E2E、队列时长、拒绝/取消/失败率、goodput。
Engine 侧另看 KV blocks、抢占/重算、prefix hit、阶段 tokens 和每步时间。
vLLM 有队列、KV、缓存等观测面；名称与直方图桶依版本核对。
[vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/)
按模型/长度桶/服务等级分组；避免把 prompt、用户内容或高基数request ID当标签。
建议 traces 关联阶段时间，metrics 看总体趋势，logs 保留脱敏故障上下文。

## 13. Benchmark：可复现的实验而非单点数字

vLLM benchmark CLI 支持 request rate、并发限制和 TTFT/TPOT/E2E goodput 阈值。
使用前记录版本与实际命令，不能把当前文档选项当所有历史版本都支持。
[vLLM bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve/)

**建议实验协议：**

1. 固定模型 revision、tokenizer、模板、采样参数、dtype、硬件和拓扑。
2. 给出输入/输出长度联合分布、EOS处理、共享前缀和突发到达模式。
3. 区分冷启动、warmup、冷缓存和热缓存；运行长度覆盖稳态与尾部。
4. Open-loop 按计划到达率发请求；closed-loop 固定并发，两者都解释适用问题。
5. 防止客户端因服务变慢而少发请求，隐藏真实排队；核对实际发出时间。
6. 扫描负载到过载区，绘制 goodput、TTFT/ITL尾部、拒绝率和成本曲线。
7. 把失败、超时、拒绝放入总请求统计；不要只对成功样本报漂亮延迟。
8. 同时报告置信区间/重复运行变化；p99 样本不足时明确其不稳定性。
9. 报每百万成功且达标输出 tokens 的成本，并注明输入成本与资源空闲摊销。
10. 一次改变一个因素，随后验证组合；对量化另设质量准入。

## 14. 自拟面试练习与答题提纲

以下全部是原创练习，不声称来自任何公司的题库。
提纲引用前文定义与来源；回答应说明假设、失败情形和验证办法。

| 练习题 | 回答提纲 |
| --- | --- |
| 1. 为聊天 API 设计 serving 平台 | SLO/长度分布 → 容量 → admission → continuous batching → 故障与压测 |
| 2. 权重装得下为什么仍 OOM？ | §4 分拆 KV/工作区；输出增长、页取整、并发与实际分片 |
| 3. 吞吐上升但用户感觉更慢？ | §2 拆 TTFT/ITL；§6 查长prefill、batch与排队；比较goodput |
| 4. GQA 模型8k上下文要多少 KV？ | 写公式、代入128 KiB/token、1 GiB/序列，再讲TP和额外开销 |
| 5. 如何选 TP8 还是 TP2×DP4？ | §8 先保证容量；比较同步、batch、缓存局部性、SLO与容错 |
| 6. 长文请求拖慢短聊天怎么办？ | §6 长度分桶、chunking、公平调度；§11 验证是否值得拆池 |
| 7. Prefix hit 很高为何 TTFT 仍高？ | §5 token加权命中；排队、未命中suffix、负载热点与缓存迁移 |
| 8. Speculation 如何保证正确？ | §10 写p/q接受和残差采样；区分分布一致、greedy与bitwise一致 |
| 9. 量化后显存减半为何没加速？ | §9 区分权重/KV/激活；确认kernel、dequant、shape与瓶颈 |
| 10. MoE 每token激活少为何服务仍贵？ | §8 总权重常驻、EP通信、expert热点、batch和网络拓扑 |
| 11. 客户端取消但显存不降怎么办？ | §12 追request生命周期、引用计数、scheduler队列和异步任务 |
| 12. 如何证明新调度策略更好？ | §13 真实长度与到达分布、open-loop、SLO goodput、失败率与重复实验 |

## 15. 学习优先级与实践

P0：熟练手算 TTFT/TPOT、roofline 与 GQA KV；用单位检查发现错误。
P1：读 Orca/PagedAttention/SARATHI，画请求状态机和每轮调度图。
P2：做一个固定版本的 serving benchmark，解释负载拐点和一次尾延迟回归。
P3：结合 profiler 学 TP/PP/EP 与 kernel；用实测定位优化对象。
P4：再研究量化、speculation、disaggregation，给每项写适用边界和回滚条件。

**可交付练习：**一页容量表、一张负载/SLO曲线、一个故障注入记录、一份设计权衡表。
先在固定模型和软件版本下测量实际 KV 占用，核对布局、分片与容量估算的差异。
再扫描请求到达率与长度分布，记录 TTFT、ITL、goodput，并用 profiler 确认实际 kernel 和瓶颈。
若评估分离式 serving，测量 KV 传输的有效带宽与排队时间；若评估量化，同时测量任务质量变化。
最后执行一次取消或 worker 故障注入，核对资源释放、恢复时间及对用户请求的影响。
