---
title: 啃完 llama2.c：从 ONNX 部署到自己写一个 LLM 推理引擎
date: 2026-05-09
tags: [LLM, 推理, C, Transformer, 学习笔记]
description: 700 行 C 跑通 Llama 2 推理。给从 CV/ONNX 转过来、第一次看 Transformer 推理代码的人。
---

# 啃完 llama2.c：从 ONNX 部署到自己写一个 LLM 推理引擎

## 写在前面：这篇文章给谁看

如果你和我一样，过去一直在做 CV 小模型的部署——拿 ONNX、上 TensorRT、写写 plugin、调调 INT8——但是从来没有真正打开过一个 LLM 推理引擎的代码，那这篇文章就是给你写的。

[karpathy/llama2.c](https://github.com/karpathy/llama2.c) 是 Andrej Karpathy 用大约 700 行 C 写的 Llama 2 推理引擎。它的好处是：

- **零依赖**：只靠 `libc` 和 `libm`，一个 `gcc -O3 -o run run.c -lm` 就能跑。
- **完整可读**：核心逻辑全在一个 `run.c` 里，没有任何抽象层、模板、宏地狱。
- **架构是真的**：不是一个玩具，跑出来的 token 和 PyTorch 完全一致（仓库里的 `test_all.py` 就是用 200 步逐 token 对齐测的）。

读懂它之后，你会建立起对 LLM 推理的一个非常清晰的"机械"理解，再去看 `llama.cpp`、`vLLM`、`TensorRT-LLM` 就有了一个最朴素的参照系。

我会假设你：

- 写过 C/C++、知道 `mmap`、知道指针运算。
- 部署过 ONNX，懂"算子"、"权重"、"推理图"这些概念。
- **不熟 Transformer 内部结构**，没看过 attention 公式细节也没关系——这篇会从零讲起。

---

## 1. 先建立大局观

我们先不看代码，只看流程。一次推理在做什么：

```
用户输入文本: "Once upon a time"
        │
        ▼
   [ Tokenizer ]   把字符串切成整数 ID 列表，比如 [1, 9038, 2501, 263, 931]
        │
        ▼
   [ Transformer ]  对每个位置算出"下一个最可能的 token 是谁"
        │           ── 这一步要做 N 层 (Llama 7B 有 32 层)
        │              每层都包含：Attention + FFN
        ▼
   [ Sampler ]     从概率分布里抽一个 token 出来 (greedy / top-p / temperature)
        │
        ▼
   把这个新 token 拼回输入末尾，回到第二步，继续算下一个。
```

整个 LLM 推理就是一个**循环**：每一轮吃一个 token、吐一个 token，直到达到 max_steps 或者吐出 EOS。

对应到 llama2.c 的源码，这三块就是：

| 模块 | 文件位置 | 对应 ONNX 部署里的什么 |
|---|---|---|
| Transformer 前向 | `run.c` 中的 `forward()` | TensorRT engine 的 `enqueue()` |
| Tokenizer | `run.c` 中的 `encode()` / `decode()` | 通常是预处理，CV 里没有对应物 |
| Sampler | `run.c` 中的 `sample()` | 后处理（NMS 之类的对应物） |
| 权重格式 | `model.bin` | `.onnx` / `.engine` |

---

## 2. 权重格式：`.bin` 文件长什么样

ONNX 文件里，权重是按算子名挂在 graph node 上的；llama2.c 走了一条**极端朴素**的路：把所有权重按一个固定顺序、紧挨着、以 fp32 写到一个二进制文件里。没有 metadata、没有 tag、没有 index——完全靠"读的人和写的人约定好顺序"。

文件头是一个 28 字节的 `Config`：

```c
typedef struct {
    int dim;         // 隐藏维度，比如 4096 (7B 模型)
    int hidden_dim;  // FFN 内部的中间维度
    int n_layers;    // Transformer 层数 (7B = 32)
    int n_heads;     // 注意力头数
    int n_kv_heads;  // K/V 头数 (可以 < n_heads, 这就是 GQA)
    int vocab_size;  // 词表大小，Llama 2 是 32000
    int seq_len;     // 最大序列长度
} Config;
```

> **彩蛋小坑**：`vocab_size` 居然用正负号偷偷编码了一个布尔位——正数表示输出层和 embedding 层共享权重（weight tying），负数表示不共享。这种"hacky"的做法在 `read_checkpoint()` 里能看到 `abs()` 操作。原作者自己也吐槽 `bit yikes`。

文件头之后，就是一连串的 fp32 数组，按这个顺序排列：

```
[ Config 头 28 字节 ]
[ token_embedding_table:   vocab_size × dim ]
[ rms_att_weight:          n_layers × dim    ]   每层 attention 前的 RMSNorm 缩放因子
[ wq:  n_layers × dim × (n_heads × head_size)    ]   Q 投影
[ wk:  n_layers × dim × (n_kv_heads × head_size) ]   K 投影 (注意是 kv_heads)
[ wv:  n_layers × dim × (n_kv_heads × head_size) ]   V 投影
[ wo:  n_layers × (n_heads × head_size) × dim    ]   输出投影
[ rms_ffn_weight:          n_layers × dim    ]   FFN 前的 RMSNorm
[ w1, w2, w3:              n_layers × dim × hidden_dim 三块 ]   FFN 三个矩阵
[ rms_final_weight:        dim ]
[ (跳过 seq_len × head_size 个 fp32 — 旧版本残留的 RoPE 频率表) ]
[ wcls (可选): vocab_size × dim ]   如果不共享 embedding 才有
```

这就是为什么在 `run.c` 的 `memory_map_weights()` 里你会看到一长串 `ptr += ...; ptr += ...;`——它是按上面这个顺序，把一个大数组切成几十个子数组的指针：

```c
w->token_embedding_table = ptr;
ptr += p->vocab_size * p->dim;
w->rms_att_weight = ptr;
ptr += n_layers * p->dim;
w->wq = ptr;
ptr += n_layers * p->dim * (p->n_heads * head_size);
// ... 一直这样切下去
```

类比 ONNX：你可以把这个 `.bin` 想成一个**没有 graph、只有 raw weights initializer 段、并且写死了 layout 顺序的极简 ONNX**。

---

## 3. 怎么把模型"加载"进内存：mmap

这一步是我个人觉得 llama2.c 最巧妙的地方之一。

普通做法：`fopen → fread → 把整个文件读到 malloc 出来的 buffer 里`。7B 模型 26GB，这意味着你要 malloc 26GB、再 read 26GB。

llama2.c 用的是 `mmap`：

```c
*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
```

`mmap` 把文件**直接映射成一段虚拟内存**。OS 负责按页（4KB）lazy load——你访问哪一页，它才从磁盘读哪一页。好处：

1. **零拷贝**：内核 page cache 里的数据直接被映射到你的进程地址空间。
2. **多进程共享**：你启 4 个 `./run` 进程，OS 只在物理内存里放一份权重。
3. **启动飞快**：`mmap` 调用本身只是建个映射表，毫秒级返回。

代价是：第一次访问每一页时会有缺页中断的延迟。在 LLM 这种"每个权重都要读一遍"的场景下，这个延迟均摊到第一次 forward 上。

> 对照：TensorRT engine 文件你也是 `IRuntime->deserializeCudaEngine()` 一次性加载到 GPU 显存。`mmap` 是 CPU 推理才用得上的技巧——GPU 推理你逃不掉显存拷贝。

加载完之后，`TransformerWeights` 结构体里的每一个指针，都指向 `mmap` 出来的同一块大内存的不同偏移：

```
mmap 出来的虚拟内存:
┌─────────────┬──────────────┬─────────┬─────────┬────────┐
│ token_embed │ rms_att      │ wq      │ wk      │ ...    │
└─────────────┴──────────────┴─────────┴─────────┴────────┘
       ▲             ▲              ▲          ▲
       │             │              │          │
   token_embedding   rms_att_weight wq         wk
```

---

## 4. Transformer 是什么：5 分钟速通

如果你没看过 Transformer，下面这张图是核心。一个 Transformer 由 N 个**完全一样**的 Block 串起来：

```
                    输入: 一串 token id
                          │
                          ▼
              ┌─────────────────────┐
              │ Token Embedding     │  把每个 id 查表变成一个 dim 维向量
              └─────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │  Block × N (Llama 7B 是 N=32)    │
        │                                   │
        │   x ─┬─ RMSNorm ─ Attention ─┐   │
        │     │                        +    │  ← 残差连接
        │     └────────────────────────┘   │
        │                │                  │
        │     ┌──────────┴────────────┐    │
        │     │                       │    │
        │     ├─ RMSNorm ─ FFN ───────┤    │
        │     +                       │    │
        │     └───────────────────────┘    │
        │                │                  │
        └────────────────┼──────────────────┘
                          ▼
                       RMSNorm
                          │
                          ▼
                      Linear (classifier)
                          │
                          ▼
                logits: vocab_size 维向量
                每个值 = "这个位置是某 token 的得分"
```

每个 Block 做两件事：

- **Attention**：让序列里每个位置去"看"前面所有位置，把它们的信息按相关度聚合过来。
- **FFN (Feed-Forward Network)**：一个两层 MLP，对每个位置独立做一次非线性变换。

围绕这两块，Llama 用的"小组件"是：

| 组件 | 作用 | 你之前可能熟悉的对应物 |
|---|---|---|
| RMSNorm | 归一化 | LayerNorm 的简化版 |
| RoPE | 给 Q/K 注入位置信息 | 老 Transformer 里的 sin/cos position encoding |
| GQA | K/V 头数 < Q 头数，省 KV cache | 普通 multi-head attention |
| SwiGLU | FFN 的非线性 | ReLU/GELU |

下一节我们逐个讲这些是什么，并且对着 `forward()` 的源码走一遍。

---

## 5. 跟着 `forward()` 走一遍

`run.c` 的 `forward(transformer, token, pos)` 是整个引擎的心脏。它的输入是：

- `token`：当前要喂进去的那一个 token ID（**只有一个！** 后面会解释为什么）。
- `pos`：当前 token 在整个序列里的位置（0, 1, 2, ...）。

它的输出是：`logits[vocab_size]`，告诉你下一个位置每个 token 的得分。

这里的关键观察是：**LLM 推理是一个 token 一个 token 算的，不是一次把整个序列丢进去**。这个跟训练完全不同，原因下面讲 KV cache 时会说。

### 5.1 第一步：查 embedding

```c
float* content_row = w->token_embedding_table + token * dim;
memcpy(x, content_row, dim*sizeof(*x));
```

`token_embedding_table` 是个 `[vocab_size, dim]` 的大矩阵。第 `token` 行就是这个 token 对应的初始向量。直接 memcpy 到工作 buffer `x` 里，结束。

类比 ONNX：这就是一个 `Gather` 算子。

### 5.2 第二步：进入每一层 Block 的循环

```c
for (unsigned long long l = 0; l < p->n_layers; l++) {
    // ... attention + ffn ...
}
```

下面所有内容都在这个循环里，对每一层 `l` 都做一次。

### 5.3 RMSNorm：比 LayerNorm 更简单的归一化

```c
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}
```

公式：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

和 LayerNorm 的差别：LayerNorm 要先减均值，再除以标准差，还有 bias；RMSNorm **不减均值、没有 bias**，只用 RMS（root mean square）做缩放。少几次运算，效果差不多——所以 Llama 选了它。

### 5.4 QKV 投影：把 x 打成查询/键/值

```c
matmul(s->q, s->xb, w->wq + l*dim*dim,    dim, dim);
matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
```

注意 `s->k` 和 `s->v` 这两行其实**直接写进了 KV cache**，不是写进临时 buffer：

```c
int loff = l * p->seq_len * kv_dim;          // 第 l 层的起始偏移
s->k = s->key_cache   + loff + pos * kv_dim; // 当前位置的 K
s->v = s->value_cache + loff + pos * kv_dim; // 当前位置的 V
```

`key_cache` 和 `value_cache` 是两个超大的 buffer，提前在 `malloc_run_state()` 里 calloc 好：

```
key_cache shape: [n_layers, seq_len, kv_dim]
```

也就是说，**每一层、每一个位置的 K 和 V 都被永久存着**。这就是 KV cache。

#### 为什么需要 KV cache？

回想一下 attention 公式：第 `pos` 个位置的 token 要和位置 `0..pos` 所有 token 的 K/V 做交互。

如果没有 cache，每生成一个新 token，你都得把前面所有 token 重新过一遍 QKV 投影。生成第 100 个 token 时，前 99 个的 K/V 你就重算了 99 次。

KV cache 的洞察非常简单：**前面那些 token 的 K/V 算完之后再也不会变，存起来就行**。

所以 LLM 推理实际上只对**当前位置**做 QKV 投影，K 和 V 顺手存进 cache，而 Q 拿来跟 cache 里所有历史 K 做注意力。这就是为什么 `forward()` 一次只接收一个 token 的原因。

> 这是"prefill"和"decode"两个阶段的差别——prefill 是处理 prompt（一次性大批量算），decode 是逐 token 生成（每步只算一个）。llama2.c 的 `forward()` 把它们统一成了"一次一个 token"，所以 prefill 的吞吐其实没优化到极致；vLLM 之类生产引擎会专门加速 prefill。

#### GQA：为什么 K 和 V 是 `kv_dim`，Q 是 `dim`？

```c
int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
int kv_mul = p->n_heads / p->n_kv_heads;  // KV 共享因子
```

普通 multi-head attention 里 `n_kv_heads == n_heads`，每个查询头都有自己专属的 K/V 头。但 Llama 2 里 K/V 头数可以**少于** Q 头数——多个 Q 头共享同一组 K/V，这就是 GQA (Grouped Query Attention)。

好处：KV cache 缩小一倍/几倍。Llama 2 70B 用 GQA 把 KV cache 从原本不可承受变成了可以接受的大小。

`kv_mul` 就是"每个 KV 头被几个 Q 头共享"。后面 attention 循环里，Q 头 `h` 对应的 KV 头是 `h / kv_mul`。

### 5.5 RoPE：旋转位置编码

```c
for (int i = 0; i < dim; i+=2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1;
    for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? s->q : s->k;
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
    }
}
```

这一段如果第一次看，会有点懵。我来翻译一下它在干嘛。

**问题**：attention 这个操作本身是"位置无关"的——你把序列打乱顺序，结果不变。所以必须想办法注入"我是第几个位置"的信息。

**老办法**：在 embedding 上加一个 `sin(pos)`、`cos(pos)` 之类的"位置编码"向量。

**RoPE 的办法**：把 Q 和 K 看成一堆 2D 复数（每相邻两个 float 当一个复数），然后**根据位置 pos 把每个复数旋转一个角度**。旋转角度跟 `pos × freq` 成正比，频率随维度递减。

数学上的好处是：两个向量的点积只取决于它们的**相对位置差**。这就是"相对位置编码"——它对长度外推有更好的特性。

代码角度看：
- 每两个相邻浮点 `(v0, v1)` 看成一个 2D 向量。
- `(fcr, fci)` 是 `(cos θ, sin θ)`，角度由 `pos` 和频率决定。
- 经典 2D 旋转矩阵套上去：
  ```
  [v0']   [cos θ  -sin θ] [v0]
  [v1'] = [sin θ   cos θ] [v1]
  ```

`rotn = i < kv_dim ? 2 : 1` 这个判断处理 GQA：在 K 的维度范围内同时旋转 Q 和 K，超出之后只旋转 Q（因为 K 已经没那么多维度了）。

### 5.6 多头注意力：核心循环

```c
#pragma omp parallel for private(h)
for (h = 0; h < p->n_heads; h++) {
    float* q = s->q + h * head_size;
    float* att = s->att + h * p->seq_len;

    // 算 attention scores: q · k_t for t = 0..pos
    for (int t = 0; t <= pos; t++) {
        float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += q[i] * k[i];
        score /= sqrtf(head_size);
        att[t] = score;
    }

    // softmax
    softmax(att, pos + 1);

    // weighted sum of values
    float* xb = s->xb + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
    }
}
```

这就是 attention 的全部。用人话翻译：

对每个头 `h`：
1. **算分数**：当前位置的 Q 跟 cache 里位置 `0..pos` 所有的 K 做点积，得到一组分数。除以 `sqrt(head_size)` 是为了数值稳定。
2. **softmax**：把分数变成"权重"，加起来等于 1。
3. **加权求和**：用这组权重把对应的 V 加起来，就是这个头的输出。

注意 `(h / kv_mul) * head_size` 这个偏移——这就是 GQA 的体现：Q 头 `h` 共享 KV 头 `h / kv_mul`。

`#pragma omp parallel for` 是 OpenMP 提示——如果用 `make runomp` 编译，多头之间就会被自动并行化。这就是 llama2.c 的"魔法"：写得跟单线程一样朴素，靠一行 pragma 就能多线程。

> 这里**没有** mask！为什么？因为我们是逐 token 生成的，cache 里只有 `0..pos` 的 K/V，根本没有"未来"的 token 可以看。Causal mask 在训练时是必要的，但在 autoregressive decoding 里隐式地由"cache 里只有过去"实现了。

### 5.7 输出投影 + 残差

```c
matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
```

把多头拼起来过 `wo`，然后加回到主流 `x` 上——这就是残差连接。

### 5.8 FFN：SwiGLU

```c
rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
matmul(s->hb,  s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb[i];
    val *= (1.0f / (1.0f + expf(-val)));   // silu(x) = x * sigmoid(x)
    val *= s->hb2[i];                       // 乘 w3(x)
    s->hb[i] = val;
}
matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

for (int i = 0; i < dim; i++) x[i] += s->xb[i];
```

公式：

```
FFN(x) = w2( silu(w1(x)) ⊙ w3(x) )
其中 silu(x) = x * sigmoid(x)
```

跟传统 FFN（`w2(relu(w1(x)))`）的区别：
- 多了一个 `w3` 分支，做"门控"——`w3(x)` 决定每个维度让多少 `silu(w1(x))` 通过。
- 用 `silu` 而不是 `relu`，更平滑。

这就是 GLU (Gated Linear Unit) 家族的 SwiGLU 变体。Llama 用了它，PaLM 也用了。代价是参数量比传统 FFN 多 1.5 倍（多了一个矩阵 w3），但效果好。

最后再加一次残差，结束这一层。

### 5.9 出 Block 后：最后归一化 + 分类头

```c
rmsnorm(x, x, w->rms_final_weight, dim);
matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
return s->logits;
```

走完 N 层之后，做一次最终的 RMSNorm，然后乘上分类矩阵 `wcls`（如果开了 weight tying，它就是 `token_embedding_table` 本身），得到 `vocab_size` 维的 logits。

至此 forward 结束。下一节讲怎么从 logits 抽 token。

---

## 6. Sampler：怎么从 logits 选下一个 token

`forward()` 出来的是 `[vocab_size]` 维 logits，你需要从里面挑一个 token。`run.c` 提供了三种策略：

### 6.1 Greedy（temperature == 0）

```c
int sample_argmax(float* probabilities, int n) {
    // 直接挑得分最高的
}
```

简单暴力，输出**完全确定**——同样的 prompt 出来的永远是同样的文本。适合调试和测试（`test_all.py` 就是用 `-t 0` 来保证可复现）。

### 6.2 Temperature sampling

```c
for (int q = 0; q < vocab_size; q++) logits[q] /= temperature;
softmax(logits, vocab_size);
// 然后按概率分布抽
```

把 logits 除以 temperature，再 softmax 成概率，再按概率抽样。
- temperature < 1：分布变尖，更像 greedy。
- temperature > 1：分布变平，更随机。
- temperature == 1：原始分布。

### 6.3 Top-p（nucleus）sampling

```c
int sample_topp(float* probabilities, int n, float topp, ...) {
    // 1. 按概率从大到小排序
    // 2. 累加，直到累计概率超过 topp（比如 0.9）
    // 3. 只在这个"核"里采样
}
```

只在"概率累计前 90%"的 token 里抽样，过滤掉长尾的低概率 token。这能避免模型偶尔抽到一些意外的、让生成"跑偏"的 token。

实际中常用 `-t 1.0 -p 0.9`，作者在 README 里也是这么推荐的。

> 一个工程小技巧：`sample_topp` 里有个 `cutoff = (1 - topp) / (n - 1)`，先粗筛掉所有小于这个阈值的 token，再排序。这能把 `vocab_size = 32000` 的排序压力从全量降到几百个，是个非常实用的优化。

### 6.4 RNG

```c
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
```

xorshift\* 算法，自带 seed，可复现。不依赖 `rand()` 是因为不同平台的 `rand()` 行为不一致。

---

## 7. Tokenizer：BPE 是什么

LLM 不是按字符也不是按词工作，而是按 **subword token**。比如 `"Once"` 可能就是一个 token，但 `"Onnomatopoeia"` 这种生僻词会被切成 `["On", "nomato", "poeia"]` 之类。

Llama 用的是 SentencePiece BPE。`tokenizer.bin` 文件里存的是：

```
[max_token_length: int32]
对每个 token i (0 <= i < vocab_size):
  [score: float32]
  [length: int32]
  [bytes: length 个字节]
```

`run.c` 读取这个文件，把每个 token id 对应的字符串和分数存起来。

### 编码：`encode()` 在干嘛

BPE 的编码算法在 `run.c` 里大致是这样：

1. **逐字符切**：把输入字符串按 UTF-8 codepoint 切开，每个 codepoint 查表得到一个 token id。如果查不到（生僻字符），fallback 成 byte token。
2. **贪心合并**：循环扫一遍当前 token 序列，找出能合并成"分数最高的"那个 token 的相邻 pair，合并它。一直合并直到没有能合并的 pair 为止。

```c
while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;
    for (int i = 0; i < (*n_tokens - 1); i++) {
        sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
            best_score = t->vocab_scores[id];
            best_id = id;
            best_idx = i;
        }
    }
    if (best_idx == -1) break;
    tokens[best_idx] = best_id;
    // 删除 best_idx+1，整个序列前移一位
    ...
}
```

时间复杂度其实不太好（每轮 O(n × vocab lookup)），但对大模型推理来说 prompt 长度通常几百 token，编码时间相比 forward 可以忽略。

### 解码：`decode()` 的小细节

```c
char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }   // BOS 后剥前导空格
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {     // 处理 byte fallback token
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}
```

两个细节：
- SentencePiece 习惯在 BOS 后剥掉前导空格。
- 一些 token 是 `<0x41>` 这种字面 byte，要解析成实际 byte。

---

## 8. 主循环：把这些拼起来

```c
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
              char *prompt, int steps) {
    int* prompt_tokens = ...;
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];   // prompt 阶段，下一个 token 用真实的
        } else {
            next = sample(sampler, logits);  // generate 阶段，从 logits 采样
        }
        pos++;

        if (next == 1) break;   // BOS 当作终止信号

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);

        token = next;
    }
}
```

这里有个非常重要的概念：**prefill vs decode**。

- 当 `pos < num_prompt_tokens - 1` 时：我们在"消化 prompt"。每一步还是调 `forward()`，但**不用采样**——直接拿下一个 prompt token 当 `next`。这样 KV cache 被填充起来，但输出被忽略。
- 当 `pos >= num_prompt_tokens - 1` 时：prompt 消化完了，开始真正生成。每步采样得到 `next`，下次再喂回去。

llama2.c 的 prefill 和 decode 用的是**同一段代码**——一次只算一个 token。这在工程上很简洁，但效率不是最优。生产引擎（vLLM、TensorRT-LLM）会让 prefill 一次性把所有 prompt token 拼成一个 batch 喂进去，能更好地利用 GPU。

---

## 9. Chat 模式：仅仅是套了个 prompt 模板

`chat()` 函数干的事其实非常朴素：

```c
char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
char user_template[]   = "[INST] %s [/INST]";
```

把用户输入按 Llama 2 Chat 的格式套一下，喂给同一个 `forward()`。模型的"对话能力"完全来自训练时见过的数据格式。

EOS（token id = 2）作为 assistant 回合的结束信号；遇到它就停止生成、把 `user_turn` 设回 1，等下一轮用户输入。

这个例子让我意识到：**对话格式不是模型的属性，是 prompt 的约定**。任何"chat tuned"模型都只是在某种特定格式上 fine-tune 过。

---

## 10. 量化：runq.c 简单聊

`runq.c` 跟 `run.c` 几乎一样，只在两点不同：

1. **权重存储**：所有矩阵权重都被量化成 int8，按 group（默认 64 个元素一组）共享一个 fp32 scale。也就是 Q8\_0 格式（llama.cpp 的术语）。
2. **matmul 的输入也量化**：每次 matmul 前，把 fp32 激活动态量化成 int8，然后做 int8 × int8 累加，最后乘回 scale。

收益：
- **模型变小 4 倍**：fp32 weight 4 字节，int8 weight 1 字节（外加少量 scale 开销）。
- **matmul 变快**：int8 × int8 的整数乘加，CPU 上跑得比 fp32 快几倍。

代价：略微的精度损失。Q8\_0 是个非常保守的量化方案——只在矩阵权重上做，scale 还是 fp32，RMSNorm 这类敏感算子完全不动。所以质量损失非常小。

> ONNX 部署里你做过 INT8 量化的话这里不会陌生：本质上就是 weight + activation per-group symmetric quantization。区别是这里没有 calibration——它是 PTQ 但用的是 `min(|w|)/127` 这种最朴素的 scale 算法，不做误差最小化。

---

## 11. 训练侧（model.py）一瞥

`model.py` 是 PyTorch 写的训练代码。和 `run.c` 的差异：

| 维度 | model.py | run.c |
|---|---|---|
| 一次处理 | 整个 batch × 整个序列 | 单个 token |
| KV cache | 不需要 | 必须有 |
| Causal mask | 显式构造 | 隐式（cache 里没未来） |
| RoPE | 预计算 freqs，前向时查表 | 每步重算 sincos |
| Attention | `F.scaled_dot_product_attention` (Flash) | 手写 for 循环 |
| 数据类型 | fp32 / bf16 / fp16 | fp32（runq.c 是 int8） |

从 PyTorch 训练完之后，`export.py` 把模型按上面"Section 2"讲过的 `.bin` 格式 dump 出来：

```bash
python export.py out/model.bin --version 0   # fp32 v0 给 run.c
python export.py out/model.bin --version 2   # int8 v2 给 runq.c
```

序列化的代码全在 `serialize_fp32()` 和 `serialize_int8()` 里，几行就写完。

---

## 12. 怎么自己跑一遍

```bash
# 拉代码
git clone https://github.com/karpathy/llama2.c.git
cd llama2.c

# 拿一个 15M 的玩具模型
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# 编译
make runfast      # 或者 make run / make runomp

# 跑！
./run stories15M.bin -t 0.8 -n 256 -i "Once upon a time, Lily met a Shoggoth"
```

我自己 M1 MacBook 上跑 `stories15M`，大概 100+ tok/s，看着 token 一个一个流出来非常爽。

如果想跑测试对齐 PyTorch 输出：

```bash
make run        # 必须先构建 ./run
pytest          # 会自动下载 stories260K，跑 200 步对齐
```

---

## 13. 学完这些之后，可以做什么

我自己列的一个学习路径：

1. **改 `run.c` 的 sampler**：加个 top-k、加个 repetition penalty、加个 mirostat。
2. **加一个 batch 维度**：让 `forward()` 接受多个 token 同时算，看看 prefill 加速能到什么程度。
3. **写一个 CUDA 版本**：把 matmul、attention、softmax 换成 CUDA kernel。仓库 todo 里就有 `run.cu`。
4. **去看 `llama.cpp`**：现在再去看 `ggml`、`ggml-alloc`、`ggml-cuda`，每个数据结构都有了一个 llama2.c 的对应物可以参照。
5. **去看 vLLM 的 PagedAttention**：理解为什么"KV cache 是块连续 buffer"在 batch serving 场景下不够用。

llama2.c 给我最大的启发是：**LLM 推理引擎本质上不复杂**。复杂的是：

- 怎么让它在一张 H100 上跑到 90% 利用率。
- 怎么让 100 个并发请求共享一个 KV cache 池。
- 怎么把上下文从 4K 扩到 1M。
- 怎么把模型从 fp16 量到 int4 还不掉精度。

这些都是工程问题，但**核心算法**（attention、KV cache、RoPE、SwiGLU、采样）就是 `run.c` 这 700 行写的东西。

读懂这 700 行，你就有了那张可以查所有问题的"参照地图"。

---

## 附：常见疑问

**Q: 为什么 `run.c` 不支持 batch？**
A: 设计选择。Karpathy 的目标是"教学用 + 单用户本地推理"。加 batch 维度会让代码长不少。

**Q: 为什么不用 SIMD（AVX、NEON）？**
A: 同样是为了简单。靠 `-O3 -march=native` 让编译器自动向量化，已经能拿到不错的性能。要极致性能就上 `llama.cpp`。

**Q: 7B 以上为什么跑不动？**
A: README 里提到，13B+ 的指针运算会有 int 溢出 bug；另外 fp32 推理 13B 模型要 50GB 内存、跑得也慢。runq.c 把内存压到 1/4，但仍然慢——主要是单线程 CPU 推理本身的极限。

**Q: 这跟 ONNX Runtime / TensorRT 比有什么差距？**
A: 巨大。生产引擎有：fused kernel、显存复用、连续 batching、speculative decoding、PagedAttention、各种精度的量化。llama2.c 没有任何这些——它的价值是**让你看懂这些优化想解决什么问题**。

---

如果你看到这里，建议下一步：把 `run.c` 在编辑器里打开，对着这篇文章每节走一遍。代码就在那里，700 行，没有任何隐藏。
