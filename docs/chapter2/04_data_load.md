# 第一节 数据加载

> 虽然本节内容在实际应用中非常重要，但是由于各种文档加载器的迭代更新，以及各类 AI 应用的不同需求，具体选择需要根据实际情况。本节仅作简单引入，但请务必**重视数据加载**环节，**“垃圾进，垃圾出 (Garbage In, Garbage Out)”** ——高质量输入是高质量输出的前提。

## 一、文档加载器

在RAG系统中，**数据加载**是整个流水线的第一步，也是不可或缺的一步。文档加载器负责将各种格式的非结构化文档（如PDF、Word、Markdown、HTML等）转换为程序可以处理的结构化数据。数据加载的质量会直接影响后续的索引构建、检索效果和最终的生成质量。

### 1.1 主要功能

- **文档格式解析**
将不同格式的文档（如PDF、Word、Markdown等）解析为文本内容。

- **元数据提取**
在解析文档内容的同时，提取相关的元数据信息，如文档来源、页码等。

- **统一数据格式**
将解析后的内容转换为统一的数据格式，便于后续处理。

### 1.2 当前主流RAG文档加载器

| 工具名称 | 特点 | 适用场景 | 性能表现 |
|---------|---------|---------|---------|
| **PyMuPDF4LLM** | PDF→Markdown转换，OCR+表格识别 | 科研文献、技术手册 | 开源免费，GPU加速 |
| **TextLoader** | 基础文本文件加载 | 纯文本处理 | 轻量高效 |
| **DirectoryLoader** | 批量目录文件处理 | 混合格式文档库 | 支持多格式扩展 |
| **Unstructured** | 多格式文档解析 | PDF、Word、HTML等 | 统一接口，智能解析 |
| **FireCrawlLoader** | 网页内容抓取 | 在线文档、新闻 | 实时内容获取 |
| **LlamaParse** | 深度PDF结构解析 | 法律合同、学术论文 | 解析精度高，商业API |
| **Docling** | 模块化企业级解析 | 企业合同、报告 | IBM生态兼容 |
| **Marker** | PDF→Markdown，GPU加速 | 科研文献、书籍 | 专注PDF转换 |
| **MinerU** | 多模态集成解析 | 学术文献、财务报表 | 集成LayoutLMv3+YOLOv8 |

## 二、Unstructured文档处理库

[**Unstructured**](https://docs.unstructured.io/open-source/) 是一个专业的文档处理库，专门设计用于RAG和AI微调场景的非结构化数据预处理。提供了统一的接口来处理多种文档格式，是目前应用较广泛的文档加载解决方案之一。

### 2.1 Unstructured的核心优势

**格式支持广泛**
- 支持多种文档格式：PDF、Word、Excel、HTML、Markdown等
- 统一的API接口，无需为不同格式编写不同代码

**智能内容解析**
- 自动识别文档结构：标题、段落、表格、列表等
- 保留文档元数据信息

### 2.2 [支持的文档元素类型](https://docs.unstructured.io/open-source/concepts/document-elements)

Unstructured能够识别和分类以下文档元素：

| 元素类型 | 描述 |
|---------|------|
| `Title` | 文档标题 |
| `NarrativeText` | 由多个完整句子组成的正文文本，不包括标题、页眉、页脚和说明文字 |
| `ListItem` | 列表项，属于列表的正文文本元素 |
| `Table` | 表格 |
| `Image` | 图像元数据 |
| `Formula` | 公式 |
| `Address` | 物理地址 |
| `EmailAddress` | 邮箱地址 |
| `FigureCaption` | 图片标题/说明文字 |
| `Header` | 文档页眉 |
| `Footer` | 文档页脚 |
| `CodeSnippet` | 代码片段 |
| `PageBreak` | 页面分隔符 |
| `PageNumber` | 页码 |
| `UncategorizedText` | 未分类的自由文本 |
| `CompositeElement` | 分块处理时产生的复合元素* |

> **注：** `CompositeElement` 是通过分块（chunking）处理产生的特殊元素类型，由一个或多个连续的文本元素组合而成。例如，多个列表项可能会被组合成一个单独的块。

## 三、从LangChain封装到原始Unstructured

在第一章的示例中，我们使用了LangChain的`UnstructuredMarkdownLoader`，它是LangChain对Unstructured库的封装。接下来展示如何直接使用Unstructured库，这样可以获得更大的灵活性和控制力。

### 3.1 代码示例

创建一个简单的示例，尝试使用Unstructured库加载并解析一个PDF文件：

> 若代码运行出现报错 `ImportError: libgl.so.1 cannot open shared object file no such file or directory`, 执行 `sudo apt-get install python3-opencv` 安装依赖库。

```python
from unstructured.partition.auto import partition

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
```

**partition函数参数解析：**

- `filename`: 文档文件路径，支持本地文件路径
- `content_type`: 可选参数，指定MIME类型（如"application/pdf"），可绕过自动文件类型检测
- `file`: 可选参数，文件对象，与filename二选一使用
- `url`: 可选参数，远程文档URL，支持直接处理网络文档
- `include_page_breaks`: 布尔值，是否在输出中包含页面分隔符
- `strategy`: 处理策略，可选"auto"、"fast"、"hi_res"等
- `encoding`: 文本编码格式，默认自动检测

`partition`函数使用自动文件类型检测，内部会根据文件类型路由到对应的专用函数（如PDF文件会调用`partition_pdf`）。如果需要更专业的PDF处理，可以直接使用`from unstructured.partition.pdf import partition_pdf`，它提供更多PDF特有的参数选项，如OCR语言设置、图像提取、表格结构推理等高级功能，同时性能更优。

> **完整代码文件**：[`01_unstructured_example.py`](https://github.com/datawhalechina/all-in-rag/blob/main/code/C2/01_unstructured_example.py)

> [**Unstructured官方文档**](https://docs.unstructured.io/open-source/core-functionality/partitioning)

## 练习

- 使用`partition_pdf`替换当前`partition`函数并分别尝试用`hi_res`和`ocr_only`进行解析，观察输出结果有何变化。
- 首次的输出（输出的部分节选）：
  Element 80 (Title):
播报
============================================================
Element 81 (Title):
权威合作编辑
============================================================
Element 82 (Title):
检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合检索和生成技术的模型。它通过引用外部知识库的信
============================================================
Element 83 (Title):
中国科学院大学计算机科
============================================================
Element 84 (UncategorizedText):
息来生成答案或内容，具有较强的可解释性和定制能力，适用于问答系统、文档生成、智能助手等多个自然语言处理任务中。 RAG模型的优势在于通用性强、可实现即时的知识更新，以及通过端到端评估方法提供更高效和精准的信息服务 [1]。
- 替换之后的输出（节选）：
- hi-res:
解析完成: 224 个元素, 共 8277 个字符
元素类型统计: {'Image': 21, 'UncategorizedText': 89, 'Header': 4, 'NarrativeText': 68, 'Table': 4, 'FigureCaption': 4, 'Title': 30, 'ListItem': 4}

前 5 个元素：
Element 1 (Image):

------------------------------------------------------------
Element 2 (UncategorizedText):
Bh fe Se «8 Be BR 8H
------------------------------------------------------------
Element 3 (UncategorizedText):
地图
------------------------------------------------------------
Element 4 (UncategorizedText):
Oke 6B CR OBS
------------------------------------------------------------
Element 5 (Header):
百度首页 登录 注册
- ocr_only:
解析完成: 138 个元素, 共 8266 个字符
元素类型统计: {'UncategorizedText': 50, 'Title': 61, 'NarrativeText': 26, 'ListItem': 1}

前 5 个元素：
Element 1 (UncategorizedText):
Bh fe Se «8 Be BR 8H 4 Oke 6B CR OBS HES SR ith
------------------------------------------------------------
Element 2 (Title):
Cy) Bai@ Bil | eee x SABER | my
------------------------------------------------------------
Element 3 (NarrativeText):
WBARAATERBEBANEAAR, VRE DBI HAER RE, CEL IEROR: REAR RRMA, NECA CRIRS, LM SER! iFIB>> Bn pias HEAR AAS a MAB BAAR RATE > 1 REA i oa E Ay ) Ha Otte Lewes
------------------------------------------------------------
Element 4 (UncategorizedText):
ARB WARAZ—
------------------------------------------------------------
Element 5 (Title):
MoingSh
------------------------------------------------------------
 ### 结论
在本练习中，我将partition_pdf的strategy参数分别设置为"hi_res"和"ocr_only"对同一份rag.pdf进行解析
1）元素数量与结构丰富度对比
使用hi_res时，共解析出224个元素（8277个字符），元素类型较为丰富，包括Image、Header、NarrativeText、Table、FigureCaption、Title和 ListItem等。例如，Image有21个，Table有4个，说明hi_res模式不仅提取文本，还对版面结构（图片、表格、标题、页眉等）进行了检测和分割。
使用ocr_only时，共解析出138个元素（8266个字符），整体元素数量明显减少，类型主要集中在UncategorizedText、Title 和 NarrativeText，不再区分图片、表格和图注等结构。这表明ocr_only更侧重对整页进行OCR识别，而不做复杂的版面分析。

2）文本粒度与布局信息的差异
在hi_res模式下，前几个元素中既包含单独的Image元素，也包含Header和较短的文本片段（大量UncategorizedText），文本被切得更细，能够较好地反映出页面上的布局结构。
在ocr_only模式下，前几个元素多为长串的OCR文本，往往将同一页上的多个区域“揉”成一个较大的文本块，同时仍然存在较多乱码。这说明ocr_only 更像是对页面整体做一次纯OCR，缺乏精细的块级分段和版面标签。

3）结论
综合来看，hi_res模式适合需要保留版面结构信息的场景（例如保留图片、表格、标题层次等），代价是元素数量多、粒度细、预处理开销更大；ocr_only模式则更偏向简单粗暴的文本提取，结构信息较弱，但实现逻辑相对简单。在本PDF中，两种策略字符总量相近，但hi_res 提供了更多关于布局和元素类型的结构化信号，而ocr_only输出则更接近“纯OCR文本”。
  
