# TurboQuant v2.0

LLM 推理中的 KV Cache 压缩系统。通过随机旋转 + Lloyd-Max 量化 + QJL 投影，将 KV Cache 压缩至原始 fp16 的数分之一，同时保持注意力精度。

## 整体架构

```
                        ┌──────────────────────────────────────────────┐
                        │          start_with_turboquant.py            │
                        │  (入口: 加载 TQ patch → 启动 vLLM 服务)       │
                        └──────────────┬───────────────────────────────┘
                                       │ enable_no_alloc()
                        ┌──────────────▼───────────────────────────────┐
                        │             adapter (适配层)                   │
                        │  ┌────────────┐  ┌──────────────┐            │
                        │  │ base.py    │  │vllm_adapter  │ ascend (WIP)│
                        │  │ ABC 接口    │  │ monkey-patch │            │
                        │  └────────────┘  └──────┬───────┘            │
                        └─────────────────────────┼────────────────────┘
                                                   │ install_hooks()
                        ┌──────────────────────────▼────────────────────┐
                        │              runtime (运行时)                   │
                        │  ┌─────────┐ ┌──────────┐ ┌───────────────┐  │
                        │  │context  │ │kv_store  │ │ ring_buffer   │  │
                        │  │请求槽管理│ │压缩KV存储 │ │ 环形缓冲(精确) │  │
                        │  └────┬────┘ └────┬─────┘ └──────┬────────┘  │
                        │       │    capture.py   │               │      │
                        │       │  (KV捕获引擎)    │               │      │
                        │       └────────┬────────┘               │      │
                        │         attention.py                     │      │
                        │         (混合注意力引擎) ◄────────────────┘      │
                        └──────────────┬──────────────────────────────────┘
                                       │ 调用量化 / Triton 内核
                        ┌──────────────▼──────────────────────────────────┐
                        │              core (核心算法)                      │
                        │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
                        │  │codebook  │ │rotation  │ │ triton_kernels   │ │
                        │  │Lloyd-Max │ │随机旋转矩阵│ │ 5个融合GPU内核    │ │
                        │  └──────────┘ └──────────┘ └──────────────────┘ │
                        │              quantizer.py                        │
                        │        (MSE量化 + 内积量化)                       │
                        └─────────────────────────────────────────────────┘
```

## 核心数据流

推理分为两个阶段：

### Prefill 阶段（首次填充）

```
输入 Token KV
     │
     ▼
KVCaptureEngine.ingest_prefill()
     │
     ├── seq_len > ring_capacity?
     │       │
     │       ├── YES: 前段 → CompressedKVStore.append_chunk()
     │       │        (TurboQuantProd 量化 key + Group 量化 value)
     │       │
     │       │        后段 → RingBuffer.write()
     │       │        (精确存储最近 ring_capacity 个 token)
     │       │
     │       └── NO:  全部 → RingBuffer.write()
     │
     ▼
  使用 SDPA 计算 prefill attention (不走 paged cache)
  注: paged KV cache 已在初始化时通过 KV Sharing 机制仅保留 1 层
```

### Decode 阶段（逐 token 生成）

```
新 Token KV
     │
     ▼
KVCaptureEngine.ingest_decode()
     │
     ├── RingBuffer.write() → 如果溢出:
     │       │
     │       └── 溢出的 token → CompressedKVStore.append_chunk()
     │
     ▼
AttentionEngine.compute_decode_attention()
     │
     ├── 仅有压缩 KV → tq_fused_decode() (Triton 融合内核)
     │
     ├── 仅有环形缓冲 → tq_recent_buffer_decode() (Triton 内核)
     │
     └── 两者都有 → _hybrid_attention()
             │
             ├── Pass1: tq_fused_decode() → (acc_c, m_c, l_c)
             ├── Pass2: tq_recent_buffer_decode() → (acc_r, m_r, l_r)
             └── Merge: tq_hybrid_merge() → 合并两段 online softmax
```

## 模块详解

### `turboquant/core/` — 核心量化算法

#### `codebook.py` — Lloyd-Max 码本计算

| 函数 | 功能 |
|------|------|
| `beta_pdf(x, d)` | 计算 d 维单位球面上均匀分布单坐标的 PDF (Beta 分布) |
| `_conditional_mean(lo, hi, d)` | 计算条件期望 E[X \| lo < X < hi] |
| `_mse_cost(centroids, d)` | 计算给定质心集的 MSE 代价 |
| `compute_lloyd_max_codebook(d, bits)` | 迭代求解最优 Lloyd-Max 码本（返回 centroids + boundaries） |
| `get_codebook(d, bits)` | 获取码本（内存缓存 → 磁盘 JSON → 实时计算，三级缓存） |
| `get_codebook_tensors(d, bits, device)` | 返回 GPU Tensor 格式的码本（centroids + boundaries） |

预计算码本存储在 `core/codebooks/codebook_d{D}_b{bits}.json`，覆盖 d=64/128/576, bits=1/2/3/4。

#### `rotation.py` — 随机旋转与 QJL 投影

| 函数 | 功能 |
|------|------|
| `generate_rotation_matrix(d, device, seed)` | 通过 QR 分解生成正交矩阵 Pi ∈ R^{d×d} (Algorithm 1) |
| `generate_qjl_matrix(d, device, seed)` | 生成 QJL 随机投影矩阵 S (i.i.d. N(0,1)) |
| `rotate_forward(x, Pi)` | 正向旋转: y = x @ Pi^T |
| `rotate_backward(y, Pi)` | 逆向旋转: x = y @ Pi |

#### `quantizer.py` — 量化器

**数据结构:**
- `MSEQuantized(indices, norms, bits)` — MSE 量化结果（bit-packed 索引 + L2 范数）
- `ProdQuantized(mse_indices, qjl_signs, residual_norms, norms, mse_bits)` — 内积量化结果

**辅助函数:**
- `_pack_indices(indices, bits)` — 将量化索引 bit-pack 成 uint8
- `_unpack_indices(packed, bits, d)` — 解包 bit-packed 索引

**量化器类:**

| 类 | 算法 | 量化流程 | 反量化流程 |
|----|------|---------|-----------|
| `TurboQuantMSE` | Algorithm 1 (最小均方误差) | 归一化 → 旋转 → 最近质心查找 → bit-pack | 查表 → 逆旋转 → 恢复范数 |
| `TurboQuantProd` | Algorithm 2 (优化内积) | 先做 MSE(b-1 bits) → 残差 → QJL sign(S·r) → 1 bit/坐标 | MSE 反量化 + QJL 反量化(无偏估计) |

`TurboQuantProd.attention_score(query, quantized_key)` 直接计算 query 与量化 key 的注意力分数，无需完整反量化。

#### `triton_kernels.py` — 融合 Triton GPU 内核

5 个 Triton JIT 内核及其 Python 封装：

| 内核 | 封装函数 | 功能 |
|------|---------|------|
| `_tq_fused_score_kernel` | `tq_fused_score()` | 融合 MSE+QJL 注意力分数计算 |
| `_tq_fused_decode_kernel` | `tq_fused_decode()` | 全融合：分数 + online softmax + value 反量化 + 加权求和 |
| `_tq_fused_decode_graph_kernel` | `tq_fused_decode_graph()` | CUDA-Graph 兼容版本（动态 N，while 循环读取设备端计数器） |
| `_tq_recent_buffer_kernel` | `tq_recent_buffer_decode()` | 环形缓冲精确 KV 的 fused attention |
| `_tq_hybrid_merge_kernel` | `tq_hybrid_merge()` | 合并两段 online softmax 状态 |

所有内核原生支持 GQA (Grouped Query Attention)。

---

### `turboquant/runtime/` — 运行时 KV Cache 管理

#### `ring_buffer.py` — RingBuffer

固定大小的环形缓冲区，存储最近的精确 KV token。

| 方法 | 功能 |
|------|------|
| `write(key, value)` | 写入 token，返回溢出的旧 token |
| `write_single_graph(key, value)` | CUDA-Graph 兼容的单 token 写入 |
| `read_all()` | 读取所有有效 token |
| `drain()` | 读取并清空 |
| `reset()` | 重置缓冲区 |

预分配张量: `keys(N_cap, H_kv, D)`, `values(N_cap, H_kv, D)`，设备端计数器 `_count_tensor`。

#### `kv_store.py` — CompressedKVStore

每个请求独立的压缩 KV 存储，预分配固定地址缓冲区。

| 方法 | 功能 |
|------|------|
| `append_chunk(key, value)` | 量化并存储一批 KV token |
| `get_flat_cache()` | 返回连续缓冲区视图（CUDA-Graph 用） |
| `get_quantized_view(n)` | 返回 ProdQuantized + ValueQuantized |
| `reset()` | 清空存储 |

**存储布局（全部预分配）:**

| 缓冲区 | 形状 | 类型 | 用途 |
|--------|------|------|------|
| `mse_indices_buf` | (H, N, packed_d_mse) | uint8 | Key MSE 量化索引 |
| `qjl_signs_buf` | (H, N, packed_d_signs) | uint8 | Key QJL 符号位 |
| `key_norms_buf` | (H, N) | float16 | Key L2 范数 |
| `res_norms_buf` | (H, N) | float16 | 残差 L2 范数 |
| `value_data_buf` | (H, N, D) | uint8 | Value 量化数据 |
| `value_scales_buf` | (H, N, n_groups) | float16 | Value 组量化 scale |
| `value_zeros_buf` | (H, N, n_groups) | float16 | Value 组量化 zero-point |
| `_n_tensor` | (1,) | int32 | 设备端 token 计数器 |

辅助函数:
- `quantize_values(v, bits, group_size)` — 非对称组量化
- `dequantize_values(vq, group_size)` — 反量化
- `unpack_values(packed, bits)` — 解包 bit-packed value

#### `capture.py` — KVCaptureEngine

编排 KV 捕获管道: `incoming KV → ring buffer → compressed store`

| 方法 | 功能 |
|------|------|
| `ingest_prefill(key, value)` | Prefill 批量捕获：前段压缩，后段入环形缓冲 |
| `ingest_prefill_from_paged(key_cache, value_cache, slot_mapping, seq_len)` | 从 paged cache 提取后捕获 |
| `ingest_decode(key, value)` | Decode 单 token 写入，自动处理溢出 |
| `ingest_decode_graph(key, value)` | CUDA-Graph 兼容的 decode 写入 |
| `flush()` | 强制将环形缓冲排空到压缩存储 |
| `reset()` | 重置所有状态 |

#### `context.py` — 请求上下文管理

| 类 | 功能 |
|----|------|
| `RequestKVState` | 单请求单层的 KV 状态：CompressedKVStore + RingBuffer + KVCaptureEngine |
| `RequestSlotManager` | 线程安全的请求状态池（每层一个），管理 allocate/get/release 生命周期 |

`RequestSlotManager` 使用 `threading.Lock` 保护并发访问，支持多租户推理。

#### `attention.py` — AttentionEngine

混合注意力计算引擎，调度最优内核路径：

| 方法 | 功能 |
|------|------|
| `compute_decode_attention(query, state)` | 主入口：根据数据分布调度到压缩/精确/混合路径 |
| `_compressed_attention(query, state)` | 压缩 KV 上的 fused decode |
| `_ring_buffer_attention(query, state)` | 精确 KV 上的 fused decode |
| `_hybrid_attention(query, state)` | 混合两段 online softmax |
| `compute_decode_attention_graph(query, state, layer_buffers)` | CUDA-Graph 兼容路径 |
| `preallocate_layer_buffers(QH, D, device)` | 预分配 CUDA Graph 所需的 7 个缓冲区 |

---

### `turboquant/adapter/` — 框架适配层

#### `base.py` — FrameworkAdapter (ABC)

定义推理框架适配器接口:

| 抽象方法 | 功能 |
|---------|------|
| `discover_layers(model)` | 发现模型中的所有 attention 层 |
| `get_sequence_info(attn_metadata)` | 从框架元数据提取每个请求的序列信息 |
| `get_request_id(seq_info)` | 获取请求唯一标识 |
| `extract_kv_tensors(key, value, layer_info)` | 将框架格式的 KV 转换为 TQ 格式 (H_kv, seq_len, D) |
| `write_output(output, target, layer_info)` | 将 TQ 输出写回框架格式 |
| `install_hooks(model, config)` | 安装 TQ hooks 到模型（主入口） |
| `free_kv_cache(model)` | 释放框架的 paged KV cache |

数据类: `AttentionLayerInfo`（层信息）, `SequenceInfo`（请求序列信息）。

#### `vllm_adapter.py` — vLLM 适配器

通过 monkey-patch 拦截 vLLM attention forward，路由到 TQ 管道。

**KV Sharing 机制 (关键优化):**

在 `enable_no_alloc()` 中，安装 hooks 后会设置 `kv_sharing_target_layer_name` 属性，使所有 TQ 管理的 attention 层共享第 1 层的 paged KV cache。随后从 `get_kv_cache_specs` 返回值中移除其余层的规格，使 vLLM 仅为 1 层分配 paged cache 而非 N 层。由于 TQ 在 patched forward 中自行捕获 KV 并使用 SDPA/Triton 内核计算注意力，paged cache 实际不被使用，但保留 1 层以满足 vLLM 框架要求。这将 paged cache 内存开销从 O(N) 降至 O(1)。

**关键类/函数:**

| 类/函数 | 功能 |
|---------|------|
| `TQConfig` | TQ 配置数据类 (key_bits, value_bits, ring_capacity, ...) |
| `VllmLayerState` | 每层状态: slot_manager + attention_engine + buffers |
| `VllmAdapter` | vLLM 适配器实现，管理所有层的 hook |
| `enable_no_alloc(...)` | **预导入补丁**，在 vLLM engine 初始化前调用，patch `Executor.get_kv_cache_specs` 自动安装 hooks |
| `install_hooks(...)` | 便捷函数：创建适配器 → 发现层 → 安装 hooks（此时 KV cache 尚未分配，不释放） |
| `_make_patched_forward(orig_fn, state, no_alloc)` | 创建 patched forward（闭包） |
| `_handle_prefill(...)` | Prefill: 通过 capture engine 捕获 KV，用 SDPA 计算注意力 |
| `_handle_decode_capture(...)` | Decode: 通过 capture engine 捕获新 token |
| `_single_seq_decode(...)` | 单请求 TQ decode attention |
| `_multi_seq_decode(...)` | 多请求 TQ decode attention |

#### `ascend_adapter.py` — Ascend NPU 适配器 (占位)

为华为 Ascend NPU 预留的适配器骨架，所有方法抛出 `NotImplementedError`。

---

### `turboquant/monitor/` — GPU 监控

#### `gpu_monitor.py`

| 类 | 功能 |
|----|------|
| `GPUStats` | GPU 状态快照数据类（利用率、显存、TQ 统计） |
| `GPUMonitor` | 单节点 GPU 监控器，后台线程周期采样 |
| `MultiNodeMonitor` | 多节点聚合监控（占位） |

使用 `pynvml` 获取 GPU 利用率，`torch.cuda` 获取显存信息，可传入 `tq_stats_provider` 回调获取 TQ 状态。

---

### `turboquant/utils/` — 工具

#### `memory.py`

| 函数 | 功能 |
|------|------|
| `estimate_memory(...)` | 估算 TQ 配置的显存占用，计算最大并发请求数 |
| `estimate_compression_ratio(head_dim, key_bits, value_bits)` | 估算压缩比 vs fp16 |

---

### 入口文件

#### `start_with_turboquant.py`

服务启动入口。流程:
1. 调用 `tq.enable_no_alloc()` 安装预导入补丁
2. 引入 vLLM CLI 参数解析
3. 强制 `enforce_eager=True`（避免 CUDA Graph 与 TQ no_alloc hooks 冲突）
4. 启动 vLLM OpenAI 兼容 API 服务

#### `Dockerfile`

基于 `vllm/vllm-openai:v0.18.0`，安装 TQ 包后以 `start_with_turboquant.py` 为入口。

## 调用链关系

### 初始化链路

```
start_with_turboquant.py
  │
  ├── tq.enable_no_alloc(key_bits=4, value_bits=3, ...)
  │     │
  │     └── patches Executor.get_kv_cache_specs
  │           │
  │           └── [vLLM engine init 时自动触发]
  │                 └── patched_get_kv_cache_specs()
  │                       │
  │                       ├── _worker_install_tq(worker)
  │                       │     │
  │                       │     ├── VllmAdapter.discover_layers(model)
  │                       │     │     └── 遍历 model.static_forward_context
  │                       │     │           └── AttentionLayerInfo(...)
  │                       │     │
  │                       │     ├── 每层创建:
  │                       │     │     ├── RequestSlotManager (线程安全请求池)
  │                       │     │     ├── AttentionEngine (注意力引擎)
  │                       │     │     ├── AttentionEngine.preallocate_layer_buffers() (7个缓冲区)
  │                       │     │     └── VllmLayerState (组合以上)
  │                       │     │
  │                       │     ├── _make_patched_forward() → monkey-patch impl.forward
  │                       │     │
  │                       │     └── KV Sharing: 所有 TQ 层共享第 1 层的 paged cache
  │                       │           ├── 第 1 层: kv_sharing_target_layer_name = None
  │                       │           └── 其余层: kv_sharing_target_layer_name = 第 1 层名
  │                       │
  │                       ├── orig_get_kv_cache_specs()
  │                       │     └── 返回原始 KV cache 规格
  │                       │
  │                       └── 从规格中移除 shared layers → vLLM 只为 1 层分配 paged cache
  │
  └── vllm run_server(args)
```

### Prefill 推理链路

```
vLLM forward (prefill)
  │
  └── patched_forward()
        │
        ├── _handle_prefill(vllm_state, key, value, attn_metadata)
        │     │
        │     ├── _reshape_kv() → (H_kv, seq_len, D)
        │     │
        │     └── state.capture_engine.ingest_prefill(k, v)
        │           │
        │           ├── seq_len <= capacity → RingBuffer.write()
        │           │
        │           └── seq_len > capacity:
        │                 ├── CompressedKVStore.append_chunk(前段)
        │                 │     ├── TurboQuantProd.quantize(key)
        │                 │     │     ├── TurboQuantMSE.quantize() → MSEQuantized
        │                 │     │     └── QJL sign projection → qjl_signs
        │                 │     └── quantize_values(value) → ValueQuantized
        │                 │
        │                 └── RingBuffer.write(后段)
        │
        ├── no_alloc? → _prefill_attention_sdpa()
        │                 └── F.scaled_dot_product_attention(q, k, v)
        │
        └── _write_result(result, output)
```

### Decode 推理链路

```
vLLM forward (decode)
  │
  └── patched_forward()
        │
        ├── _handle_decode_capture(vllm_state, key, value, attn_metadata)
        │     │
        │     └── capture_engine.ingest_decode(k, v)
        │           │
        │           ├── RingBuffer.write(k, v) → overflow?
        │           │     └── YES → CompressedKVStore.append_chunk(overflow)
        │           │
        │           └── (ring buffer 未满则直接存储)
        │
        └── _single_seq_decode / _multi_seq_decode
              │
              └── AttentionEngine.compute_decode_attention(query, state)
                    │
                    ├── has_compressed AND has_ring → _hybrid_attention()
                    │     │
                    │     ├── tq_fused_decode()          [Triton Kernel 2]
                    │     │     └── 融合: MSE+QJL分数 + online softmax + value反量化
                    │     │
                    │     ├── tq_recent_buffer_decode()  [Triton Kernel 4]
                    │     │     └── 精确 KV attention (online softmax)
                    │     │
                    │     └── tq_hybrid_merge()           [Triton Kernel 5]
                    │           └── 合并两段 online softmax 状态
                    │
                    ├── has_compressed only → tq_fused_decode() [Kernel 2]
                    │
                    └── has_ring only → tq_recent_buffer_decode() [Kernel 4]
```

## 量化原理

### Algorithm 1: TurboQuantMSE (最小均方误差)

1. 归一化: x̂ = x / ||x||
2. 随机旋转: y = Π · x̂ (使各坐标分布趋近 N(0, 1/d))
3. 标量量化: 每个坐标找最近 Lloyd-Max 质心
4. 存储: bit-packed 索引 + L2 范数

### Algorithm 2: TurboQuantProd (优化内积)

在 Algorithm 1 基础上增加 QJL 残差补偿:

1. 用 (b-1) bits 做 MSE 量化 → x̃
2. 计算残差: r = x - x̃
3. QJL 投影: sign(S · r) → 1 bit/坐标
4. 内积无偏估计: `<y, x̃>` + ||r|| · √(π/2)/d · `<S^T·signs, y>`

### Value 量化

非对称组量化: 将 D 维向量按 group_size 分组，每组独立计算 min/max → scale/zero-point，量化为 uint8。

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `key_bits` | 3-4 | Key 每坐标量化比特数 |
| `value_bits` | 2-3 | Value 每元素量化比特数 |
| `value_group_size` | 32 | Value 组量化组大小 |
| `ring_capacity` | 128 | 环形缓冲容量（精确存储最近 N 个 token） |
| `max_tokens_per_request` | 32768 | 每请求最大压缩 token 数 |
| `max_num_seqs` | 256 | 最大并发请求数 |
| `no_alloc` | True | Prefill 后是否释放 paged KV cache |

## 依赖

- Python >= 3.10
- PyTorch >= 2.1
- NumPy, SciPy
- Triton >= 3.0 (vLLM 模式)
- vLLM >= 0.18.0 (vLLM 模式)
- pynvml (GPU 监控，可选)
