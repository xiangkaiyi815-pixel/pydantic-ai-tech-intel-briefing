from search_assistant.verification.policy import (
    extract_key_claims,
    requires_calibration,
    requires_verification,
    verify_claims_against_sources,
)
from search_assistant.contracts import SourceEvidence


def test_current_version_claim_requires_verification():
    assert requires_verification("FastAPI 0.138.1 is the latest version.", "research")


def test_hard_and_high_stakes_require_calibration():
    assert requires_calibration("hard", has_unverified_claims=False)
    assert requires_calibration("high_stakes", has_unverified_claims=False)
    assert requires_calibration("simple", has_unverified_claims=True)


def test_extract_key_claims_keeps_numbered_claims():
    claims = extract_key_claims("FastAPI 0.138.1 was released in 2026. Use official docs.")

    assert "FastAPI 0.138.1 was released in 2026" in claims


def test_extract_key_claims_splits_chinese_markdown_into_compact_claims():
    text = """
### 证据核查
- NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡。
- DeepSeek-V4 的参数量未从搜索结果中确认。

搜索记录:
- 联网搜索: 已执行
- 搜索结果:
  1. [direct-official] NVIDIA DGX Spark official specifications - https://example.test/dgx-spark
"""

    claims = extract_key_claims(text)

    assert "NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡" in claims
    assert "DeepSeek-V4 的参数量未从搜索结果中确认" in claims
    assert "证据核查" not in claims
    assert all("搜索记录" not in claim for claim in claims)
    assert all(len(claim) < 120 for claim in claims)


def test_extract_key_claims_reads_markdown_evidence_table_rows():
    text = """
### 证据核查

| 来源 | 确认了什么 | 缺少什么 |
|------|-----------|----------|
| NVIDIA DGX Spark 官方规格 | 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡 | 无 DeepSeek-V4 性能实测 |
| DeepSeek-V4 ModelScope | 模型页面存在 | 参数量未确认 |
"""

    claims = extract_key_claims(text)

    assert (
        "NVIDIA DGX Spark 官方规格 确认 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡"
        in claims
    )
    assert "DeepSeek-V4 ModelScope 确认 模型页面存在" in claims
    assert all("缺少什么" not in claim for claim in claims)
    assert all("------" not in claim for claim in claims)


def test_extract_key_claims_filters_reasoning_scaffolding_but_keeps_technical_claims():
    text = """
### 推理路径：不同并行策略下节点依赖关系不同

下面我把这种“紧密耦合”具体拆开，同时说明从现有文档中能确认什么、哪些地方是推断。
你问的是推理部署场景下的分布式，而非训练。
根据搜索结果中的两套官方文档（NVIDIA Megatron Bridge 和 vLLM），分布式推理主要有几种并行方式：

- **张量并行（Tensor Parallelism）**：将同一层的参数切分到不同 GPU。
- **流水线并行（Pipeline Parallelism）**：将模型的不同层分配到不同节点，前一个节点的输出作为后一个节点的输入。

本次搜索返回了两份官方文档：
- **NVIDIA Megatron Bridge 并行化指南** —— 列出了张量并行、流水线并行、专家并行等多种并行策略。
"""

    claims = extract_key_claims(text)

    assert "推理路径：不同并行策略下节点依赖关系不同" not in claims
    assert all("下面我把" not in claim for claim in claims)
    assert all("你问的是" not in claim for claim in claims)
    assert all("根据搜索结果" not in claim for claim in claims)
    assert all("本次搜索返回" not in claim for claim in claims)
    assert any("张量并行" in claim and "将同一层的参数切分到不同 GPU" in claim for claim in claims)
    assert any("流水线并行" in claim and "将模型的不同层分配到不同节点" in claim for claim in claims)
    assert any("列出了张量并行、流水线并行、专家并行等多种并行策略" in claim for claim in claims)


def test_extract_key_claims_filters_next_step_action_items():
    text = """
### 下一步验证

- **确认你想部署的模型类型**：是 Dense 模型还是 MoE 模型。
- **确认硬件互联**：你的节点之间是高速网络（如 RDMA）还是普通以太网。
- Check accelerator vendor roadmaps: do NVIDIA, AMD, or Intel list CXL 3.0/4.0 support in their latest datasheets?

### 证据核查
- NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡。
"""

    claims = extract_key_claims(text)

    assert all("确认你想部署" not in claim for claim in claims)
    assert all("确认硬件互联" not in claim for claim in claims)
    assert all("vendor roadmaps" not in claim for claim in claims)
    assert "NVIDIA DGX Spark 官方规格确认 128GB 统一内存和 ConnectX-7 网卡" in claims


def test_extract_key_claims_filters_evidence_headings_and_next_verification_fragments():
    text = """
## 证据检查了什么

两篇文档都确认了“将模型拆分到多个计算单元，并依赖通信协同”这个基本事实。

## 下一步可验证

- 这些是后续验证的点。
- 模型的参数规模、专家数、激活参数比例；
- 节点间网络带宽能否承受 all-to-all 通信。
"""

    claims = extract_key_claims(text)

    assert "证据检查了什么" not in claims
    assert all("这些是后续验证的点" not in claim for claim in claims)
    assert all("模型的参数规模" not in claim for claim in claims)
    assert all("节点间网络带宽" not in claim for claim in claims)
    assert any("将模型拆分到多个计算单元" in claim for claim in claims)


def test_extract_key_claims_filters_user_framing_interpretation():
    text = """
你描述的“各负责一部分”最贴近模型并行（参数拆分）而非数据并行（数据拆分）。
- 单节点放不下时组合张量并行和流水线并行跨节点。
"""

    claims = extract_key_claims(text)

    assert all("你描述的" not in claim for claim in claims)
    assert any("张量并行和流水线并行跨节点" in claim for claim in claims)


def test_extract_key_claims_filters_user_assumption_scaffolding():
    text = """
关键假设：你关注的是AI服务器在大模型推理场景下的内存瓶颈。
你对PCIe协议有基本了解，想理解CXL增加的缓存一致性和内存池化。
CXL官方页面确认CXL面向处理器、内存扩展和加速器。
"""

    claims = extract_key_claims(text)

    assert all("你关注的是" not in claim for claim in claims)
    assert all("你对PCIe" not in claim for claim in claims)
    assert "CXL官方页面确认CXL面向处理器、内存扩展和加速器" in claims


def test_extract_key_claims_filters_live_eval_scaffolding_and_user_framing():
    text = """
根据当前搜索获得的CXL官方信息（[CXL About页面](https://www.computeexpresslink.org/about-cxl/)），CXL在AI服务器领域的价值主要体现在：
- **标准潜力**：CXL 4.0规范已发布，带宽翻倍至128 GT/s，并增强了捆绑端口和RAS特性。
但同时，搜索中**未找到**以下关键证据：
- AI推理或训练场景的实际部署案例、性能基准（延迟、吞吐量）。

“相对独立”意味着你默认节点之间耦合松散、可独立工作——这个假设不完全成立，取决于具体并行策略。
但“相对独立”这个说法需要增加一个重要限定条件：节点之间的耦合程度取决于具体采用的并行策略。
不过“相对独立”的程度取决于具体的并行拆分方式，不能一概而论。
你理解的“各负责一部分”可能涵盖模型并行和专家并行。
你的“各负责一部分”这个描述，最贴近模型并行而非数据并行。
我查阅了两份官方文档：NVIDIA Megatron Bridge 的并行策略文档和 vLLM 的分布式扩展指南。
搜索到的两份官方文档直接印证了“将模型拆分到多个计算单元，并依赖通信协同”这个基本框架：
这两份文档都印证了“将模型拆分到多个计算单元，并依赖通信协同”这个基本事实。
它们直接列出了数据并行、张量并行、流水线并行、专家并行等模式，并且都明确了“模型跨多个计算单元拆分、依赖通信协同”这一框架。
文档框架覆盖训练和推理场景。
文档明确面向推理。
这个假设不完全成立——在不同并行策略下，节点的依赖程度差别很大。
"""

    claims = extract_key_claims(text)

    assert all("根据当前搜索获得" not in claim for claim in claims)
    assert all("搜索中" not in claim for claim in claims)
    assert all("相对独立" not in claim for claim in claims)
    assert all("你理解的" not in claim for claim in claims)
    assert all("你的“各负责一部分”" not in claim for claim in claims)
    assert all("我查阅了" not in claim for claim in claims)
    assert all("搜索到的两份官方文档" not in claim for claim in claims)
    assert all("这两份文档" not in claim for claim in claims)
    assert all("它们直接列出了" not in claim for claim in claims)
    assert all("文档框架覆盖" not in claim for claim in claims)
    assert all("文档明确面向推理" not in claim for claim in claims)
    assert all("这个假设不完全成立" not in claim for claim in claims)
    assert any("CXL 4.0规范" in claim for claim in claims)


def test_extract_key_claims_filters_english_reasoning_scaffolding():
    text = """
**Reasoning snapshot**
- I checked the three most likely candidates for current major work: Google DeepMind's Genie blog, NVIDIA's Cosmos page, and NVIDIA's Isaac GR00T page.
- What remains unknown: the production readiness of Cosmos and the API availability of Genie 3.
- Look for recent papers on physics foundation models from academic labs or startups.

**Evidence basis**
- Google DeepMind Genie 2 is explicitly described as a large-scale foundation world model in 2026.
"""

    claims = extract_key_claims(text)

    assert all("I checked" not in claim for claim in claims)
    assert all("What remains unknown" not in claim for claim in claims)
    assert all("Look for recent papers" not in claim for claim in claims)
    assert any("Genie 2 is explicitly described" in claim for claim in claims)


def test_verifies_claim_when_source_text_conservatively_supports_it():
    source = SourceEvidence(
        title="Feishu message reply API 2026",
        url="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
        snippet="The Feishu message reply API supports replying to received messages from a bot.",
        provider="unit",
        checked_at="2026-06-27T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        ["Feishu message reply API supports replying to received messages in 2026"],
        [source],
    )

    assert len(verified) == 1
    assert verified[0].verdict == "verified"
    assert verified[0].source == source.url
    assert unverified == []


def test_verifies_chinese_parallelism_claim_against_english_source_aliases():
    source = SourceEvidence(
        title="vLLM distributed inference parallelism",
        url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
        snippet=(
            "Distributed inference supports tensor parallelism and pipeline parallelism. "
            "Expert parallelism can use all-to-all communication across nodes."
        ),
        provider="unit",
        checked_at="2026-07-03T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        ["分布式大模型推理可以使用张量并行、流水线并行和专家并行，并需要节点间通信"],
        [source],
    )

    assert len(verified) == 1
    assert verified[0].source == source.url
    assert unverified == []


def test_verifies_foundational_distributed_inference_claim_against_official_english_sources():
    sources = [
        SourceEvidence(
            title="NVIDIA Megatron Bridge official parallelisms documentation",
            url="https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
            snippet=(
                "Megatron Bridge supports various data-parallel and model-parallel deep learning "
                "workload deployment methods. Data Parallelism replicates the model across multiple GPUs. "
                "Data batches are evenly distributed between GPUs and inter-GPU communication is required "
                "to keep the model replicas consistent."
            ),
            provider="direct-official",
            checked_at="2026-07-04T00:00:00Z",
        ),
        SourceEvidence(
            title="vLLM official parallelism and scaling documentation",
            url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
            snippet=(
                "Distributed inference strategies for a single-model replica include tensor parallel "
                "inference. For multi-node multi-GPU serving, combine tensor parallelism with pipeline "
                "parallelism across nodes until there is enough GPU memory for the model."
            ),
            provider="direct-official",
            checked_at="2026-07-04T00:00:00Z",
        ),
    ]

    verified, unverified = verify_claims_against_sources(
        ["分布式大模型推理可以理解为多节点通过并行策略分担一部分推理工作，并通过节点间通信协同完成"],
        sources,
    )

    assert len(verified) == 1
    assert verified[0].source == sources[1].url
    assert unverified == []


def test_verifies_live_distributed_parallelism_detail_claims_against_official_sources():
    sources = [
        SourceEvidence(
            title="NVIDIA Megatron Bridge official parallelisms documentation",
            url="https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html",
            snippet=(
                "Tensor parallelism works best within a single node. Pipeline parallelism can work "
                "across nodes. Expert parallelism is specific to MoE models. DeepEP and HybridEP "
                "provide optimized MoE token dispatching. Token dropping requires alltoall or "
                "alltoall_seq token dispatcher."
            ),
            provider="direct-official",
            checked_at="2026-07-04T00:00:00Z",
        ),
        SourceEvidence(
            title="vLLM official parallelism and scaling documentation",
            url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
            snippet=(
                "Multi-node multi-GPU using tensor parallel and pipeline parallel inference: "
                "if the model is too large for a single node, combine tensor parallelism with "
                "pipeline parallelism."
            ),
            provider="direct-official",
            checked_at="2026-07-04T00:00:00Z",
        ),
    ]

    verified, unverified = verify_claims_against_sources(
        [
            "单节点放不下时组合张量并行和流水线并行跨节点",
            "MoE 模型则需要考虑专家并行和 token 分发",
        ],
        sources,
    )

    assert len(verified) == 2
    assert unverified == []


def test_verifies_chinese_cxl_definition_against_official_english_source_aliases():
    source = SourceEvidence(
        title="About CXL - Compute Express Link",
        url="https://www.computeexpresslink.org/about-cxl/",
        snippet=(
            "Compute Express Link (CXL) is an industry-supported Cache-Coherent Interconnect "
            "for Processors, Memory Expansion and Accelerators. CXL technology maintains "
            "memory coherency between the CPU memory space and memory on attached devices, "
            "which allows resource sharing for higher performance. CXL is designed to be "
            "an industry open standard interface for high-speed communications as accelerators "
            "are increasingly used to complement CPUs for Artificial Intelligence workloads."
        ),
        provider="unit",
        checked_at="2026-07-03T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        [
            "CXL 是一种开放标准的缓存一致高速互连，用于处理器、内存扩展和加速器，并支持 CPU 与设备内存一致性和资源共享",
            "CXL 面向加速器补充 CPU 的 AI 工作负载",
        ],
        [source],
    )

    assert len(verified) == 2
    assert [claim.source for claim in verified] == [source.url, source.url]
    assert unverified == []


def test_verifies_cxl_ai_server_relation_against_official_ai_ml_source_text():
    source = SourceEvidence(
        title="Compute Express Link Consortium About CXL official page",
        url="https://www.computeexpresslink.org/about-cxl/",
        snippet=(
            "CXL is an industry-supported Cache-Coherent Interconnect for Processors, Memory Expansion "
            "and Accelerators. CXL is designed to be an industry open standard interface for high-speed "
            "communications, as accelerators are increasingly used to complement CPUs in support of "
            "emerging applications such as Artificial Intelligence and Machine Learning."
        ),
        provider="direct-official",
        checked_at="2026-07-04T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        ["它与AI服务器的核心关系是：CXL官方明确将AI和机器学习列为新兴应用场景，因为加速器越来越多地用于补充CPU"],
        [source],
    )

    assert len(verified) == 1
    assert verified[0].source == source.url
    assert unverified == []


def test_verifies_cxl_version_bandwidth_and_feature_claim_with_unit_attached_number():
    source = SourceEvidence(
        title="Compute Express Link Consortium About CXL official page",
        url="https://www.computeexpresslink.org/about-cxl/",
        snippet=(
            "CXL 4.0 doubles bandwidth from 64GTs to 128GTs, adds support for bundled ports, "
            "and enhances memory RAS features."
        ),
        provider="direct-official",
        checked_at="2026-07-04T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        ["CXL 4.0 规范将带宽提升至 128 GT/s，并增加了捆绑端口和增强的内存 RAS 特性"],
        [source],
    )

    assert len(verified) == 1
    assert verified[0].source == source.url
    assert unverified == []


def test_verifies_world_model_and_robot_foundation_claims_against_official_sources():
    sources = [
        SourceEvidence(
            title="Google DeepMind Genie 2 official world model article",
            url="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/",
            snippet=(
                "Genie 2: A large-scale foundation world model. Our research paves the way "
                "for prototyping interactive experiences. Genie 2 can generate a vast diversity "
                "of rich 3D worlds. Genie 2 is a world model, meaning it can simulate virtual "
                "worlds. For every example, the model is prompted with a single image generated "
                "by Imagen 3, and people can step into and interact with the generated world."
            ),
            provider="direct-official",
            checked_at="2026-07-04T00:00:00Z",
        ),
        SourceEvidence(
            title="NVIDIA Isaac GR00T official page",
            url="https://developer.nvidia.com/isaac/gr00t",
            snippet=(
                "NVIDIA Isaac GR00T is an open reference platform for general-purpose humanoid robots. "
                "It comprises open data and data pipelines, an open robot foundation model, "
                "simulation frameworks, middleware, CUDA-X accelerated runtime libraries, and "
                "NVIDIA Jetson Thor for real-time robot inference and control. Download Isaac GR00T Models."
            ),
            provider="direct-official",
            checked_at="2026-07-04T00:00:00Z",
        ),
    ]

    verified, unverified = verify_claims_against_sources(
        [
            "Google DeepMind的Genie 2：官方文章明确将其描述为大规模基础世界模型，能从单张图像生成交互式3D环境",
            "Isaac GR00T则是一个面向人形机器人的开放参考平台，其中包含可下载的开放机器人基础模型",
            "Genie 2的官方角色已确认：其博文明确将其定义为基础世界模型，并明确为研究系统",
        ],
        sources,
    )

    assert len(verified) == 3
    assert unverified == []


def test_bilingual_aliases_do_not_verify_broad_comparison_claims():
    source = SourceEvidence(
        title="vLLM distributed inference parallelism",
        url="https://docs.vllm.ai/en/stable/serving/parallelism_scaling/",
        snippet="vLLM supports LLM distributed inference with model parallelism across nodes.",
        provider="unit",
        checked_at="2026-07-03T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        ["分布式大模型推理（模型并行）与常见的分布式数据库或微服务有本质区别"],
        [source],
    )

    assert verified == []
    assert unverified == ["分布式大模型推理（模型并行）与常见的分布式数据库或微服务有本质区别"]


def test_leaves_claim_unverified_when_sources_do_not_support_it():
    source = SourceEvidence(
        title="NVIDIA GB10 unified memory overview",
        url="https://example.test/gb10",
        snippet="GB10 systems include unified memory for local AI development.",
        provider="unit",
        checked_at="2026-06-27T00:00:00Z",
    )

    verified, unverified = verify_claims_against_sources(
        ["DeepSeek V4 can run at full precision across ten GB10 devices"],
        [source],
    )

    assert verified == []
    assert unverified == ["DeepSeek V4 can run at full precision across ten GB10 devices"]
