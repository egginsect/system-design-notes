# ML Infrastructure and Inference: Staff Interview Study Guide

This guide covers production inference design, performance tradeoffs, and staff-level ownership across product, model, hardware, and reliability. Questions are practice prompts, not predictions of any company's interview format. Examples are estimates; validate against architecture, runtime, hardware, workload, and SLO.

## Table of contents

1. [Quick revision checklist](#quick-revision-checklist)
2. [A mental model for inference](#a-mental-model-for-inference)
3. [Memory and compute sizing](#memory-and-compute-sizing)
4. [Scheduling, attention, and parallelism](#scheduling-attention-and-parallelism)
5. [Serving interfaces and operational design](#serving-interfaces-and-operational-design)
6. [API correctness and model integration](#api-correctness-and-model-integration)
7. [Conventional ranking and embedding serving](#conventional-ranking-and-embedding-serving)
8. [Three worked designs](#three-worked-designs)
9. [Metrics, diagnosis, and validation](#metrics-diagnosis-and-validation)
10. [Staff-level experience prompts and study paths](#staff-level-experience-prompts-and-study-paths)
   - [Cerebras: inference API interview drill](#cerebras-inference-api-interview-drill)
11. [Fifteen practice questions with answer outlines](#fifteen-practice-questions-with-answer-outlines)
12. [Sources](#sources)

## Quick revision checklist

- [ ] Explain prefill versus decode and identify phase-specific bottlenecks.
- [ ] Calculate KV bytes per token, per sequence, and across concurrency; convert GB/GiB carefully.
- [ ] Budget weights plus KV, workspaces, runtime, fragmentation, and safety margin.
- [ ] Contrast dynamic/continuous batching, PagedAttention/FlashAttention, and prefix caching.
- [ ] Explain tenant isolation, cancellation, backpressure, fairness, retries, and version pinning.
- [ ] Compare quantization and parallelism choices with hardware and quality caveats.
- [ ] Derive TTFT, ITL, TPOT for `Nout > 1`; discuss chunked stream events.
- [ ] Define goodput and a realistic open-loop benchmark with workload and failure variation.
- [ ] Walk through all three worked designs, including assumptions and arithmetic.
- [ ] Bring staff-level stories for performance, reliability, rollout, and cross-team ownership.

## A mental model for inference

Autoregressive generation has two distinct phases. **Prefill** consumes the input prompt in parallel, computes attention and hidden states, and produces the first next-token distribution. **Decode** repeatedly selects a token and feeds it back, one or a small number at a time. Prefill often has high matrix parallelism and can be compute intensive; decode has less parallel work per sequence and repeatedly reads model weights and KV state, so it is often constrained by memory bandwidth, KV capacity, or scheduling overhead. Short prompts, batches, contexts, kernels, and accelerators can change the bottleneck.

A single aggregate “tokens per second” hides important behavior. A large prompt can delay first tokens if it monopolizes the accelerator; long generations can occupy decode slots and queue new requests. A request can have a good first-token delay and then emit slowly, or have a delayed first token followed by fast output. Track arrival, queue, prefill, decode, and stream delivery separately.

The serving objective balances latency, goodput, quality, availability, and cost. Closed-loop peak throughput may conceal queue growth under open-loop arrivals. Define useful work, fairness, request mix, and SLO before choosing batching or parallelism.

## Memory and compute sizing

For conventional full-attention grouped-query models with uniform sequence lengths, estimate KV cache bytes as:

`2 * layers * KV_heads * head_dim * bytes_per_element * cached_tokens * active_sequences`

The factor two stores keys and values. With variable lengths, sum cached tokens per active sequence rather than multiplying one uniform length by sequence count. For 32 layers, 8 KV heads, 128 dimensions, BF16 (2 bytes), one cached token uses `2*32*8*128*2 = 131,072` bytes, or **128 KiB/token**. At 8,192 cached tokens that is **1 GiB per sequence**. Ten such sequences need about 10 GiB for KV alone, before allocator fragmentation, temporary buffers, model weights, runtime state, and other requests. Sliding-window attention, hybrid attention, MLA, and other cache designs require an adapted formula. Layouts, block sizes, dtypes, and metadata vary, so this is a planning estimate.

Weight memory is another floor. Eight billion parameters at BF16 (2 bytes each) occupy 16,000,000,000 bytes, or about **14.9 GiB** in binary units. That is not the complete serving footprint: reserve room for KV, activations and workspaces, communication buffers, framework/runtime allocations, fragmentation, and safety headroom. Margin depends on concurrency and kernels. Quantized weights reduce weight storage, but do not automatically shrink KV, and dequantization or mixed precision can introduce compute and quality costs.

Estimate model and per-request memory, then sustainable prefill/decode throughput at target lengths and concurrency. Include utilization margin, failure reserve, and queueing. Distinguish installed fleet capacity from healthy, routable capacity during maintenance or failure. Avoid sizing solely from average request length: long-tail prompt and output lengths can dominate memory residency and service time.

Compute is also phase-dependent. Prefill work roughly grows with prompt tokens and model operations, while attention work grows with sequence length and architecture. Decode performs repeated model work per generated token; weights may be reread for each step, and memory bandwidth frequently limits the token rate. Measure achieved bandwidth, kernel utilization, batch composition, and queue time before attributing slow output to “the model.”

## Scheduling, attention, and parallelism

**Dynamic batching** collects requests that arrive near one another and runs them together. A batching window can improve device utilization but adds waiting time; fixed batches may leave slots idle or force short requests to wait. **Continuous batching** updates the active batch as sequences finish and new work arrives, avoiding the need for every sequence to share a fixed start and end boundary. It can improve utilization and reduce head-of-line blocking, while requiring careful admission, fairness, memory accounting, and variable-length scheduling. Triton's dynamic batching documentation describes queue delay and preferred batch behavior as explicit tuning controls; the same principle applies broadly even when an LLM runtime uses a different scheduler. [NVIDIA Triton dynamic batching guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

**PagedAttention** manages KV state in blocks, allowing noncontiguous allocation and sharing patterns that reduce waste from reserving large contiguous regions per request. Its paper presents the design and evaluation; production benefit depends on allocator, runtime, workload, and hardware. **FlashAttention** is an IO-aware exact attention algorithm that reduces high-bandwidth-memory traffic through tiling; it is a kernel/algorithm approach, not a KV cache paging scheme. The techniques address different pressure points and may be used together. Both still require cache capacity planning. [PagedAttention paper](https://arxiv.org/abs/2309.06180); [FlashAttention paper](https://arxiv.org/abs/2205.14135).

**Prefix caching** reuses computed KV for identical prefixes, which helps repeated system prompts or shared document prefixes. Cache keys must bind all computation-affecting inputs: model/version, tokenizer and template, relevant adapters, positions, and any other state that changes hidden values. Tenant isolation matters: don't reveal whether another tenant used a prefix through timing or cache hit signals, and don't share entries across security boundaries unless policy explicitly allows it. Consider per-tenant namespaces, authorization before lookup, bounded retention, and side-channel review. A cache is not authorization.

**Chunked prefill** splits a long prompt into smaller scheduling units so decode work can interleave and long prompts do not monopolize a device. This can improve tail latency for active generations while adding scheduling overhead and potentially changing prefill efficiency. Tune chunk size against prompt throughput, TTFT, inter-token latency, and workload mix rather than treating it as universally beneficial.

Quantization can target weights, activations, KV, or combinations. Lower-bit weights reduce storage and potentially bandwidth, but speedups depend on native hardware support, kernels, shapes, packing, and dequantization overhead. Calibration and quantization-aware methods can preserve quality better than naive conversion, but quality loss is model- and task-dependent. Evaluate representative prompts, languages, tool selection, structured output validity, long contexts, and sensitive quality slices. Validate numerical behavior and fallback paths on the actual accelerator; “4-bit” is not a complete performance specification.

Compilers and kernels add another workload dimension. Static shapes and length bucketing can improve compilation and kernel reuse; graph capture mechanisms such as CUDA Graphs can reduce launch overhead where supported, but constrain dynamic behavior and memory use. Compilation and warmup affect cold-start latency. Profile realistic shapes, then balance specialization against padding, compile time, and request flexibility.

Speculative decoding uses a draft model to propose tokens and a target model to verify them. For proposed token `x`, with draft probability `q(x)` and target probability `p(x)` after agreed logits transforms, accept with probability `min(1, p(x)/q(x))`; on rejection, sample from the normalized positive residual `max(p-q, 0)`. If all draft tokens are accepted, sample an extra token from the target distribution. Under the algorithm's assumptions, this preserves the target distribution; it does not promise identical seeded outputs across implementations. Mismatched transforms or approximate verification can change it. Drafting helps only when verification cost and acceptance rate improve end-to-end performance. [Speculative decoding paper](https://arxiv.org/abs/2211.17192) describes this construction, not a guarantee for every serving stack.

Parallelism choices include **replicas** (independent full model copies, simple request-level scaling), **tensor parallelism** (split tensor operations across devices, with communication in layers), **pipeline parallelism** (partition layers into stages, with bubbles and stage balance concerns), and **expert parallelism** (place routed experts across devices, with all-to-all traffic and load skew). Combine them when model size or throughput requires it, but account for topology, collective latency, fault domains, and how a partial group affects capacity. More devices per request can increase model capacity while reducing independent request slots.

With **prefill-decode disaggregation**, separate pools handle prompt processing and token generation, transferring KV state between them. This can let each pool use hardware and scheduling suited to its phase, and can isolate long prefills from decode latency. Transfer, serialization, compatibility, and orchestration are costs. An 8 GiB KV transfer over a 100 Gbit/s link takes `8*2^30*8/100,000,000,000`, or about **0.687 seconds ideal**. Real time is higher due to protocol overhead, contention, copies, and imperfect bandwidth. Poor locality or network capacity can erase the benefit.

## Serving interfaces and operational design

A streaming API should define request schema, authentication, quotas, model/version selection, output limits, stop conditions, error shape, and stream framing. Every request needs a deadline that covers queueing and execution. Propagate cancellation when a client disconnects or deadline expires, and reclaim scheduler slots and KV promptly. Bound queues and apply backpressure; unbounded buffering converts overload into memory exhaustion and extreme latency. Fairness can use per-tenant queues, weighted scheduling, concurrency caps, or token budgets, with explicit policy for premium classes and starvation prevention.

Retries are subtle for streams. Before any output, a retry may be safe if the operation is idempotent under the chosen request identifier and sampling semantics. After partial output, transparently restarting can duplicate content or produce a different continuation. Specify whether clients resume from a checkpoint, restart explicitly, or receive a terminal error with a request ID. Enforce auth and quotas on every admission and retry path; derive tenant identity from verified credentials, never a caller-provided tenant field alone. Bind tenant ownership checks to jobs, results, and cache namespaces. Model output cannot decide access rights. Pin model version for the lifetime of a request and include it in logs and responses where appropriate. This makes debugging and reproducibility possible while allowing a controlled rollout policy for new requests.

## API correctness and model integration

Treat the tokenizer, chat template, and model as one versioned interface. Record tokenizer and template revisions, BOS/EOS behavior, stop tokens, special-token escaping, truncation, and tool/message serialization. A change can alter prompts or termination even when the model weights are unchanged. Reject unsupported options explicitly rather than silently ignoring them; define conformance cases for every backend and model version.

Sampling behavior depends on the full logits pipeline: penalties, temperature, top-k/top-p, token masks, stop rules, and numeric precision. Define whether reproducibility is expected within a backend, across batch composition, or across devices; exact bitwise determinism may not be available. Establish tolerances and distributional checks where appropriate, and test numerical edge cases and seed semantics. Keep server-side generation timing distinct from client-observed timing, which includes gateway, network, buffering, and client processing. State the measurement start/end boundaries for TTFT and chunk latency in dashboards and SLOs.

Tool-call deltas may be incomplete or invalid JSON until the stream finishes. Clients should treat deltas as provisional; parse and validate the completed arguments against a schema, then authorize the requested action independently before invoking a tool. Valid JSON or schema conformance does not prove semantic correctness or permission. Structured-output constraints can improve syntax, but validate application-level meaning too. Multimodal paths need explicit preprocessing versions, supported formats, payload and duration limits, and accounting for image/audio/video expansion into model tokens or compute.

Plan model lifecycle at the gateway and serving layers: explicit aliases versus pinned versions, deprecation windows, compatibility notices, rollback behavior, and retirement of cached or persisted artifacts. These are engineering practices and design considerations, not claims about any particular provider's current behavior.

Deployments need warmup for compilation, allocation, cache population, and representative kernels. Route a canary slice, compare latency, errors, resource use, and quality, with actionable rollback criteria. Autoscaling must account for provisioning and model load delay; queues can breach SLO before replicas are ready. Keep burst headroom or admission controls. Plan for device/host failures, partitions, load failures, and degraded capacity; remove unhealthy workers and avoid retry storms. Include the largest relevant failure domain in capacity planning.

## Conventional ranking and embedding serving

Not all inference is autoregressive generation. Ranking often has strict latency, small feature operations, and a split between CPU-friendly sparse lookup/assembly and GPU-friendly dense computation. Tiny GPU batches may lose to transfer and launch overhead; large dense batches may benefit. Measure feature fetch, joins, preprocessing, model, postprocessing, and network. Cache stable features with freshness controls and test online/offline parity.

Embedding services can batch synchronously or asynchronously according to latency needs. Tokenization, truncation, pooling, normalization, and model version are part of the vector contract and must be recorded for migration. GPU batching can improve throughput; CPU can cost less at low volume. For retrieval, monitor dimension/version compatibility, freshness, and downstream recall.

## Three worked designs

### 1. Multi-tenant streaming LLM

**Assumptions and sizing.** Assume a 70B parameter BF16 decoder, 80 layers, 8 KV heads, head dimension 128, and 2 bytes per KV element. One sequence at 8,192 tokens needs `2*80*8*128*2*8192 = 2,684,354,560` bytes, about 2.5 GiB. A 70B BF16 weight set is `70,000,000,000*2 = 140,000,000,000` bytes = **140 GB decimal = 130.4 GiB**. Tensor parallelism across 4 devices and pipeline parallelism across 2 groups gives 8 devices per replica; ideal weight share is `130.4/8 = 16.3 GiB/device`, before overhead. If each device has 80 GiB, ideal KV share per sequence is `2.5/8 = 0.3125 GiB/device` only if layers and KV heads shard evenly; runtimes may replicate KV. Per-token memory alone does not determine safe concurrency: include workspaces, uneven partitions, fragmentation, runtime allocations, and workload mix. As an illustrative sizing assumption, suppose open-loop benchmarking under target SLOs (**TTFT p95 <1 s, TPOT p95 <40 ms**) finds 25 requests/s goodput per 8-device replica. For 100 requests/s, targeting 70% of measured goodput and adding one full replica for N+1 reserve gives `ceil(100/(25*0.7))+1 = 7 replicas`, or 56 devices. This is assumed benchmark data, not actual hardware performance; remeasure with representative prompt/output lengths and failures.

**API shape.** `POST /v1/chat/completions` accepts an authenticated tenant, pinned model, messages, `stream`, output token limit, sampling parameters, and deadline. Stream events carry request ID, model version, incremental content or tool-call deltas, and a terminal usage/finish event. A client disconnect cancels work; a partial-stream failure returns a defined terminal error and does not silently regenerate the response.

**Data flow.** Authenticate and quota-check; validate schema and version; tokenize using the pinned template; enqueue in a tenant-aware scheduler; optionally match an authorized prefix cache; prefill in chunks; admit decode under KV budget; stream chunks with backpressure; record usage and terminal status; release KV. A bounded queue and per-tenant concurrency prevent one customer from consuming all slots.

**Tradeoffs and failures.** Continuous batching improves occupancy but complicates fairness. Prefix caching saves repeated prompt work but risks cross-tenant information leakage if keys or timing are shared improperly. A failed tensor-parallel member can invalidate the whole replica; route away and preserve enough spare capacity. Device OOM calls for admission reduction and cache cleanup, not unbounded retries. Slow clients require bounded output buffering. Roll out model changes with warmup, canary traffic, quality checks, and version pinning.

**Validation.** Replay an open-loop distribution of prompt/output lengths, tenant rates, cancellation, tool calls, and burst arrivals. Measure TTFT, TPOT/ITL, queue time, goodput, KV occupancy, fairness, errors, and cost. Inject worker loss, slow clients, and cancellation. Test that cache namespace rules hold and partial streams are never duplicated by automatic retry.

### 2. Low-latency ranking

**Assumptions and sizing.** Assume 20,000 ranking requests/s at peak, each scoring 100 candidates against a 256-feature dense representation plus sparse categorical features, with a 20 ms p99 end-to-end target. Reserve 5 ms for feature assembly and network, leaving 15 ms for queue and inference. Suppose a measured CPU worker handles 500 full 100-candidate requests/s at the target tail latency; 40 workers meet peak arithmetically, but add 25% headroom and N+1 capacity: plan at least 51 equivalent workers, then benchmark. If dense model kernels are GPU efficient only above a batch of 64 requests (6,400 candidates), use a microbatch window bounded to 2 ms and route sparse preprocessing to CPU. The assumptions are placeholders; benchmark representative feature cardinality and skew.

**API shape.** `POST /rank` takes request ID, user/context features or authorized feature references, candidate IDs, and model version. It returns ordered candidates, scores or calibrated ranking signals as contractually allowed, model version, and a trace ID. Enforce caller auth, candidate count limits, deadline, and feature access controls.

**Data flow.** Validate and authorize; fetch online features with deadlines; assemble and normalize; route to CPU/GPU backend based on batch shape; score; apply business constraints and deterministic tie-breaking; return ranked results. Keep feature schema and model artifact version coupled. Cache only features with acceptable freshness and tenant semantics.

**Tradeoffs and failures.** CPU avoids transfer/launch overhead for small work; GPU can improve dense throughput for batched work. Microbatching raises utilization but spends latency budget. Feature-store tail latency can dominate, so model optimization alone may do little. On GPU loss, fallback to CPU only if a validated smaller model or reduced candidate policy meets product needs; otherwise shed controlled low-priority load. Stale or missing features need explicit defaults and quality monitoring.

**Validation.** Replay real feature distributions with privacy-safe fixtures, peak open-loop load, and high-cardinality/skew cases. Track p50/p95/p99 per stage, ranking quality metrics, fallback frequency, stale features, and cost. Canary model and feature changes; compare both online latency and delayed outcome metrics. Verify deterministic version attribution and failure behavior.

### 3. Asynchronous batch embeddings

**Assumptions and sizing.** Assume 50 million documents/day, average 300 tokens, with a 24-hour completion objective. Average input rate is about `50M/86,400 = 579` documents/s. Assume a benchmarked worker sustains 1,000 documents/s at the chosen batch size and sequence mix; with 2x peak factor and 30% capacity reserve, provision roughly `579*2/1000/0.7 = 1.66`, so at least 2 workers by average arithmetic. This estimate is fragile to large documents and outages; use token-based queues and validate burst and p99 document length. Store vectors with model, tokenizer, pooling, normalization, and dimension metadata.

**API shape.** `POST /embedding-jobs` accepts dataset reference, tenant, model version, truncation policy, and idempotency key; returns job ID. `GET /embedding-jobs/{id}` reports state and counts. Results are written to a tenant-authorized object location with a manifest, checksums, schema, and per-record errors. A separate synchronous endpoint can serve small interactive requests under a tighter quota.

**Data flow.** Authenticate; validate source access; create idempotent job; enqueue bounded work units; workers read, tokenize, batch by compatible model and length, embed, write partitioned results, and checkpoint progress. Retry transient partitions with idempotent output keys; dead-letter invalid inputs. Publish completion only after manifest and writes are durable. Apply retention and deletion policy to source text and outputs.

**Tradeoffs and failures.** Larger batches improve device utilization but increase latency and memory. Length bucketing reduces padding but can reorder work and delay rare long inputs; use a maximum wait. Tokenizer or model changes require a new versioned output namespace. Worker loss replays a partition; idempotency prevents duplicate records. Poison documents are isolated rather than blocking the whole job. Queue age and token backlog are better autoscaling signals than CPU alone.

**Validation.** Test idempotent retries, partial writes, corrupt inputs, deletion, tenant isolation, and model-version changes. Benchmark throughput by token-length bucket and batch size. Reconcile submitted, successful, failed, and written record counts. Validate vector dimensions and downstream retrieval quality on a sample before switching consumers.

## Metrics, diagnosis, and validation

For generated output, distinguish **TTFT** (time to first token), **ITL** (time between output events/tokens), and **TPOT**. A common request-level estimate is `TPOT = (end_to_end - TTFT) / (Nout - 1)` for `Nout > 1`. Runtime metrics may report per-output-event ITL; bundled stream chunks can contain multiple tokens, so chunk cadence and request TPOT are not interchangeable. vLLM documents this distinction in its metrics reference. Also measure queue time, prefill time, decode time, cancellation, output tokens, active sequences, KV usage, and error class. Clearly separate server timestamps from client-observed TTFT/chunk latency, and specify whether measurement begins at request creation, send, gateway receipt, or server admission. [vLLM metrics reference](https://docs.vllm.ai/en/latest/design/metrics/).

Use **goodput**: requests or tokens that meet the agreed SLO and quality conditions, rather than raw throughput alone. Track cost per successful request or useful token, accelerator utilization, and quality regressions. Benchmark open-loop arrival schedules with realistic prompt/output distributions, burstiness, concurrency, tenant mix, and failure injection. Closed-loop clients slow their arrival as latency rises and can conceal queue collapse. Report warm and cold behavior, percentiles, confidence across repeats, and hardware/software versions.

| Symptom | Likely diagnosis | First mitigations to evaluate |
|---|---|---|
| High TTFT, normal ITL | Queueing, long prefill, cold worker, tokenizer or feature fetch | Inspect stage timings; warm capacity; chunk prefill; cap queue; improve preprocessing |
| Normal TTFT, slow ITL | Decode bandwidth, oversized active batch, KV pressure, slow stream consumer | Check per-step throughput and KV; tune concurrency; bound backpressure; evaluate quantization/hardware |
| Both degrade as load rises | Saturation and queue growth; open-loop arrival exceeds goodput | Admission control; add ready capacity; improve batching; shed according to policy |
| OOM at modest request count | Long contexts, fragmentation, underestimated KV/workspace, leaked cancellation | Reconcile memory accounting; bound token admission; inspect allocator and cleanup |
| High GPU utilization, poor goodput | Inefficient shapes, padding, CPU bottleneck, communication, excess retries | Profile full path; bucket lengths; inspect collectives and CPU stages; stop retry amplification |
| Good latency, worse ranking/retrieval quality | Feature skew, model/template mismatch, quantization or preprocessing change | Version and slice quality checks; compare inputs and artifacts; rollback suspect canary |
| Uneven tenant latency | Unfair queueing, noisy neighbor, skewed request lengths | Per-tenant budgets; weighted fair scheduling; isolate cache and observe tail by tenant |
| Scale-out does not reduce queue | Startup/load delay, shared bottleneck, unhealthy targets, bad autoscaling signal | Scale on queue age and token work; verify readiness; identify shared network/store limit |

## Staff-level experience prompts and study paths

Prepare stories with context, constraints, alternatives, decision, measurable result, and learning. Explain alignment across model, runtime, infrastructure, product, and security teams; where data changed your view; how rollout risk was managed; and how the system failed. Staff scope includes cross-team interfaces and operational ownership as well as technical depth. Use these concrete prompts: **Performance:** describe a bottleneck you found through profiling, the counterfactual you tested, and measured latency/cost gains at realistic load. **Incident:** explain a production inference degradation, how you separated symptoms from cause, contained impact, communicated, and prevented recurrence. **Cross-team launch:** describe a model/API launch spanning research, platform, product, and safety; how you resolved incompatible requirements, staged rollout, and measured adoption and quality.

**Platform path:** focus on fleet capacity, model lifecycle, routing, quotas, multi-tenancy, SLOs, deployment/rollback, failure domains, observability, and cost allocation. Be ready to explain how platform abstractions expose model-specific tuning without hiding safety-critical details.

**Engine and performance path:** focus on prefill/decode scheduling, KV layouts and allocation, kernels, numerical precision, batching, parallelism, profiling, and reproducible workload replay. Present a bottleneck hypothesis, the measurement that distinguishes it, and a result under representative load.

**Cerebras-oriented path:** the public Staff SWE, Inference API role emphasizes a hybrid design with GPU prefill and Cerebras decode. Prepare tokenizer and chat-template correctness, sampling and logits processing, numerical behavior, tool calls, structured outputs, multimodal request semantics, API conformance, and workload replay. Think through consistency at the boundary between GPU-produced prompt state and Cerebras decode, including transfer and failure behavior. Public Cerebras material describes WSE-3T with 44 GB on-chip memory and large models distributed across wafer-scale pipelines, with weights mapped locally and activations transferred; do not assume every full model fits on one wafer. The public API documents `POST /v1/chat/completions`, streaming, tool use, and model-specific parameters. These are preparation topics, not a known interview format. (Sources: [role listing](https://jobs.ashbyhq.com/cerebras/78f1098a-6022-40ba-b7a0-948711921de2/), [deployment article](https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras), [API reference](https://inference-docs.cerebras.ai/api-reference/chat-completions).)

### Cerebras: inference API interview drill

As a proposed practice exercise, design the lifecycle from GPU prefill through KV transfer to Cerebras decode. Specify compatibility for model, tokenizer, positions, cache dtype, and layout; propagate deadlines and cancellation across stages; define failure behavior before and after the first client-visible output. Build a conformance matrix for stop conditions, streamed tool-call deltas, structured outputs, sampling behavior, and numerical tolerances. This is an original design drill based on public role themes, not a claim about internal implementation or interview format.

## Fifteen practice questions with answer outlines

1. **How do prefill and decode differ?** Explain parallel prompt processing versus sequential token generation; connect compute intensity, weight/KV bandwidth, queueing, and TTFT/ITL. Name workload conditions that can change the bottleneck.
2. **Estimate KV capacity for a model.** State the formula and units; identify KV heads, layers, head dimension, dtype, context, and active sequences. Add allocator/runtime margin and discuss paging or KV quantization separately.
3. **Why can an 8B BF16 model fail to fit in 16 GiB?** Compute 16 GB decimal as about 14.9 GiB, then add KV, workspaces, activations, runtime, fragmentation, and headroom. Explain concurrent-context impact.
4. **Dynamic versus continuous batching?** Compare waiting windows and fixed groups with active-set updates. Discuss utilization, tail latency, variable lengths, fairness, and admission control.
5. **PagedAttention versus FlashAttention?** Describe KV block allocation/sharing versus IO-aware attention computation. Explain distinct problems, possible combination, and workload-specific measurement.
6. **How would you safely use prefix caching?** Bind cache keys to all computation inputs and model versions; authorize before lookup; isolate tenants; bound retention; assess timing leakage and invalidation.
7. **When is chunked prefill useful?** Explain decode interference and long-prompt monopolization; evaluate chunk size against TTFT, ITL, prefill efficiency, and mixed-load behavior.
8. **How do you evaluate quantization?** Separate weights, activations, and KV; inspect hardware-native kernels and dequantization; test quality slices, structured output, tools, long context, and numerical regressions.
9. **When does speculative decoding preserve exact output distribution?** State that verifier acceptance/rejection must implement the target distribution under the algorithm's assumptions and identical sampling transforms; distinguish exactness from speed benefit and test acceptance/cost.
10. **Choose a parallelism strategy for a model that does not fit.** Compare replicas, tensor, pipeline, and expert parallelism; calculate per-device memory; then cover communication topology, bubbles/skew, fault domains, and serving throughput.
11. **When would you disaggregate prefill and decode?** Compare specialized capacity and isolation against KV transfer, serialization, network contention, and orchestration. Estimate transfer lower bound and benchmark end-to-end benefit.
12. **Design retry and cancellation semantics for streaming.** Cover deadlines, client disconnect cleanup, bounded buffering, idempotency before output, partial-output ambiguity, resume/restart contract, and quota enforcement.
13. **How would you autoscale an inference fleet?** Use queue age, token work, phase-specific utilization, and ready capacity; include model load delay, warmup, canary/rollback, burst headroom, and node failure reserve.
14. **A latency dashboard regresses; how do you debug?** Split queue, prefill, decode, network/client, and feature stages; compare workload mix and versions; distinguish TTFT from ITL/TPOT; reproduce with open-loop replay before changing capacity.
15. **What makes a staff-level inference change successful?** Tie a technical result to SLO goodput, quality, cost, and safe rollout; describe cross-team interfaces, ownership, failure handling, adoption, and a lesson that changed later decisions.

## Sources

- [Cerebras public Staff SWE Inference API role](https://jobs.ashbyhq.com/cerebras/78f1098a-6022-40ba-b7a0-948711921de2/)
- [Cerebras, “How We Deploy Trillion-Parameter Frontier Models on Cerebras”](https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras) (2026-10-22)
- [Cerebras Inference API chat completions](https://inference-docs.cerebras.ai/api-reference/chat-completions)
- [vLLM metrics and definitions](https://docs.vllm.ai/en/latest/design/metrics/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192)
- [NVIDIA Triton dynamic batching guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)

Sources checked 2026-10-25.
