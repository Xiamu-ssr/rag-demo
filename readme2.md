# RAG Demo 项目架构设计说明书（SQLite + 向量存储抽象层版，精简）

> 目标：实现一个可维护、可删改、可扩展的 RAG 核心（对标 Dify 知识库闭环）。本版聚焦 **SQLite + 向量存储抽象层**，覆盖：多知识库 → 文档上传 → 解析/分段 → 向量化与索引 → 检索（多库）→ 生成（可选）。

---

## 1. 能力概览

* **多知识库（Collection）**：一账户可建多库，检索可选单库/多库。
* **解析与分段**：支持 **PDF / TXT / HTML / Markdown**；提供 **通用分段** 与 **父子分段**（父块用于生成，子块用于检索）。
* **向量存储抽象**：支持 FAISS、Milvus、Qdrant 等多种向量数据库，通过统一接口管理。
* **模块化模型管理**：统一的模型管理中心，支持嵌入、生成、重排序等多种模型类型。
* **检索**：向量 Top‑K、相似度阈值、MMR 去冗余；多库并发检索与融合（Score/RRF，支持可选 Rerank）。
* **删除一致性**：DB 外键级联 + 向量存储在线删除或离线重建。

---

## 2. 目录结构（最小实现）

```
app/
 ├─ api/                # FastAPI 路由（collections/documents/index/query）
 ├─ core/               # 配置、依赖注入、日志
 ├─ db/                 # SQLAlchemy engine、ORM 实体、CRUD
 ├─ ingest/             # parser / preprocess / splitters / pipeline
 ├─ indexer/            # 索引构建器、id_allocator
 ├─ retrieval/          # vector（TopK/MMR/多库融合）、rerank（可选）、fuse
 ├─ llm/                # 模块化模型管理中心（embedder、model_manager、providers、config）
 └─ storage/            # 向量存储抽象层（base、faiss_store、milvus_store、qdrant_store）
front/                  # 前端项目代码
```
注意：前端项目代码在front目录下，使用react，node版本为22
注意：后端请使用conda activate rag环境

---

## 3. 数据模型（SQLite）

> 统一采用“**父子分段模型**”：即使是通用分段（flat），也为每个文档创建一个“**全文父块**”，所有子块挂在该父块下，便于复用一套检索/生成逻辑。

### 3.1 DDL（含必要索引）

```sql
-- 知识库（RAG三阶段配置分别存储）
CREATE TABLE IF NOT EXISTS collections (
  id             TEXT PRIMARY KEY,                       -- 库ID（UUID/ULID）
  name           TEXT NOT NULL,                          -- 库名（业务展示）
  description    TEXT,                                   -- 描述
  storage_type   TEXT DEFAULT 'faiss',                  -- 向量存储类型（faiss、milvus、qdrant）
  splitting_config TEXT,                                 -- 分段配置（JSON：分段策略和预处理）
  index_config   TEXT,                                   -- 索引配置（JSON：嵌入模型和向量存储参数）
  retrieval_config TEXT,                                 -- 检索配置（JSON：检索策略和融合方式）
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_collections_name ON collections(name);

-- 文档（一个库下可有多文档）
CREATE TABLE IF NOT EXISTS documents (
  id             TEXT PRIMARY KEY,                       -- 文档ID
  collection_id  TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
  title          TEXT,                                   -- 文档名/标题
  uri            TEXT,                                   -- 来源（上传路径/URL）
  meta_json      TEXT,                                   -- 文档元数据（hash/页数/作者等）
  status         TEXT CHECK(status IN ('ingesting','indexed','failed','deleted')),
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_documents_collection ON documents(collection_id);
CREATE INDEX IF NOT EXISTS ix_documents_status ON documents(status);

-- 父块（段落/全文）
CREATE TABLE IF NOT EXISTS parents (
  id             TEXT PRIMARY KEY,
  doc_id         TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  order_no       INTEGER,                                -- 在文档中的顺序
  text           TEXT NOT NULL,                          -- 父块全文
  token_count    INTEGER,                                -- 父块Token数（可选）
  headers        TEXT                                    -- 标题链/层级信息（可选）
);
CREATE INDEX IF NOT EXISTS ix_parents_doc ON parents(doc_id);
CREATE INDEX IF NOT EXISTS ix_parents_order ON parents(doc_id, order_no);

-- 子块（检索单元）
CREATE TABLE IF NOT EXISTS chunks (
  id             TEXT PRIMARY KEY,
  parent_id      TEXT NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
  order_no       INTEGER,                                -- 在父块中的顺序
  text           TEXT NOT NULL,                          -- 子块文本
  token_count    INTEGER                                 -- 子块Token数（可选）
);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS ix_chunks_order ON chunks(parent_id, order_no);

-- 向量映射（向量本体存FAISS文件；DB 仅保存映射与元信息）
CREATE TABLE IF NOT EXISTS embeddings (
  vector_id      INTEGER PRIMARY KEY,                    -- int64，写入FAISS的外部ID
  chunk_id       TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  collection_id  TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
  model          TEXT NOT NULL,                          -- 嵌入模型名（冗余用于自检）
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_embeddings_collection ON embeddings(collection_id);
CREATE INDEX IF NOT EXISTS ix_embeddings_chunk ON embeddings(chunk_id);

-- 模型配置表（模型管理中心）
CREATE TABLE IF NOT EXISTS model_configs (
  id             TEXT PRIMARY KEY,                       -- 模型配置ID
  name           TEXT NOT NULL UNIQUE,                   -- 模型名称
  provider       TEXT NOT NULL,                          -- 提供商（openai、huggingface等）
  model_type     TEXT NOT NULL,                          -- 模型类型（embedding、llm、rerank）
  config_json    TEXT NOT NULL,                          -- 模型配置（JSON格式）
  is_active      BOOLEAN DEFAULT FALSE,                  -- 是否为当前激活模型
  is_available   BOOLEAN DEFAULT TRUE,                   -- 是否可用
  created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_model_configs_type ON model_configs(model_type);
CREATE INDEX IF NOT EXISTS ix_model_configs_active ON model_configs(model_type, is_active);
```

### 3.2 字段说明（逐表）

**collections**

* `id`：库唯一ID。
* `name`：库名（唯一约束便于人读管理）。
* `storage_type`：向量存储类型，支持faiss、milvus、qdrant等。
* `splitting_config`：分段配置（JSON），包含分段模式、父子块参数、预处理策略等。
* `index_config`：索引配置（JSON），包含嵌入模型、向量维度、向量存储参数等。
* `retrieval_config`：检索配置（JSON），包含检索策略、MMR参数、融合方式、重排序等。

**documents**

* `collection_id`：所属库，删除库时级联删除文档。
* `uri`：文件来源（本地路径/URL）；可用于对象存储回源。
* `meta_json`：如 `{"sha256":"...","pages":10,"mime":"application/pdf"}`。
* `status`：`ingesting`（入库中）/`indexed`（索引激活）/`failed`/`deleted`。

**parents**

* `doc_id`：所属文档；**通用分段**时为“全文父块”。
* `order_no`：父块在文档中的顺序，便于生成时拼接。
* `headers`：标题链，如 `H1/H2/H3`。

**chunks**

* `parent_id`：所属父块。
* `order_no`：在父块中的顺序，便于“命中邻域窗口”裁剪。

**embeddings**

* `vector_id`：写入向量存储的外部 ID；用于删除/重建/多库掩码。
* `chunk_id`：向量对应的业务子块ID（检索后可回溯原文）。
* `collection_id`：冗余库ID，便于过滤/重建，兼作一致性自检键。

**model\_configs**

* 记录模型配置信息；`provider` 指定模型提供商；`model_type` 区分嵌入、生成、重排序模型；`is_active` 标识当前激活模型。

---

## 4. 解析与分段

* **文件类型**：PDF/TXT/HTML/Markdown。
* **预处理**：空白归一（空格/换行/制表符）、Unicode NFKC、全角→半角、去页眉/页脚/目录（可开关）、可选去重（simhash）。
* **通用分段（flat）**：递归分隔符 + `chunk_size` + `chunk_overlap`（建议按 Token）。
* **父子分段**：父块（段落/全文）→ 子块；检索命中子块，生成拉取父块并做“命中邻域窗口”裁剪。

---

## 5. 向量存储（抽象化设计）

* **存储抽象**：通过 `VectorStore` 基类统一接口，支持 FAISS、Milvus、Qdrant 等多种向量数据库。
* **FAISS 实现**：文件布局 `/data/indices/{collection_id}/index.faiss` 与 `meta.json`；索引类型可选 `IndexHNSWFlat` 或 `IndexFlatIP`。
* **Milvus 实现**：连接 Milvus 服务，支持分布式向量存储和高性能检索。
* **Qdrant 实现**：连接 Qdrant 服务，提供高效的向量相似性搜索。
* **ID 管理**：统一使用 `vector_id` 管理向量的增删改查操作。
* **索引构建**：直接构建并激活索引，简化部署和维护流程。
* **删除策略**：支持在线删除和批量重建，根据存储类型选择最优策略。

---

## 6. 配置示例（库级，极简）

> 目标：**尽量少的可见项**，其余走后端默认。以下 JSON 可直接写入 `collections.config_json`。

### 6.1 分段配置 `splitting_config`

#### A) 通用分段（flat）
```json
{
  "mode": "flat",
  "separators": ["\n\n", "\n"],
  "chunk_size": 1024,
  "chunk_overlap": 50,
  "preprocess": {
    "normalize_whitespace": true,
    "remove_urls_emails": false
  }
}
```

#### B) 父子分段（父=段落）
```json
{
  "mode": "parent_child",
  "parent": {
    "type": "paragraph",
    "separators": ["\n\n", "\n"],
    "max_tokens": 2000
  },
  "child": {
    "separators": ["\n\n", "\n"],
    "chunk_size": 512,
    "chunk_overlap": 80
  },
  "preprocess": {
    "normalize_whitespace": true,
    "remove_urls_emails": false
  }
}
```

#### C) 父子分段（父=全文，适合短文档）
```json
{
  "mode": "parent_child",
  "parent": {
    "type": "document"
  },
  "child": {
    "separators": ["\n\n", "\n"],
    "chunk_size": 512,
    "chunk_overlap": 80
  },
  "preprocess": {
    "normalize_whitespace": true,
    "remove_urls_emails": false
  }
}
```

> 说明：`unit` 与 `nfkc` 等高级项不暴露；默认按**字符长度**裁切，Unicode 统一为 UTF‑8；如需更精细可在以后引入 `advanced` 段。

---

### 6.2 索引策略 `index_config`
> 选择嵌入模型和向量存储类型，其他参数由后端按模型和存储类型自动设置。

```json
{
  "embedding_model": "bge-m3",
  "storage_type": "faiss"
}
```

---

### 6.3 检索设置 `retrieval_config`
> 固定默认：向量相似度=**cosine**；候选规模、分数归一化等细节走后端默认。下面四种策略互斥选择一种。

#### A) 向量检索
```json
{
  "mode": "vector",
  "vector": {
    "top_k": 3,
    "min_score": 0.5
  }
}
```

#### B) 全文检索（BM25）
```json
{
  "mode": "lexical",
  "lexical": {
    "top_k": 3
  }
}
```

#### C) 混合检索（权重设置）
```json
{
  "mode": "hybrid_weighted",
  "hybrid": {
    "top_k": 3,
    "weights": {
      "vector": 0.7,
      "lexical": 0.3
    }
  }
}
```

#### D) 混合检索（Rerank 设置）
```json
{
  "mode": "hybrid_rerank",
  "hybrid": {
    "top_k": 3
  },
  "rerank": {
    "model": "bge-reranker-base"
  }
}
```

> 说明：
> - `top_k` = 最终返回条数；`min_score` = 全局阈值；
> - 混合检索的候选规模、归一化与融合算法使用系统默认（无需配置）。

---
