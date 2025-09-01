# RAG Demo - 检索增强生成演示系统

一个可维护、可删改、可扩展的检索增强生成（RAG）核心系统，对标 Dify 知识库闭环功能。

## 项目结构

```
rag-demo/
├── front/                  # 前端React项目
│   ├── src/
│   ├── package.json
│   └── ...
├── app/                    # 后端FastAPI项目
│   ├── api/               # 路由薄层
│   ├── core/              # 全局配置、依赖注入、日志、错误
│   ├── db/                # ORM/DDL/CRUD（SQLite）
│   ├── ingest/            # 解析(parser)/预处理(preprocess)/分段(splitters)
│   ├── llm/               # 模型管理中心
│   │   ├── embedder.py    # 嵌入器接口适配层
│   │   ├── model_manager.py # 模型管理器
│   │   ├── providers/     # 模型提供商适配器
│   │   └── config.py      # 模型配置管理
│   ├── storage/           # 向量存储抽象层
│   │   ├── base.py        # 向量存储基类
│   │   ├── faiss_store.py # FAISS向量存储实现
│   │   ├── milvus_store.py # Milvus向量存储实现
│   │   └── qdrant_store.py # Qdrant向量存储实现
│   ├── indexer/           # index_builder + id_allocator
│   ├── retrieval/         # candidate_search + mmr + fuse + (rerank)
│   └── utils/             # token 计数、文本裁剪、度量统计
├── main.py                # FastAPI应用入口
├── requirements.txt       # Python依赖
├── .env.example          # 环境配置示例
└── readme.md             # 项目说明
```

## 技术栈

### 前端
- React 18 + TypeScript
- Vite 构建工具
- Ant Design UI组件库
- React Router 路由管理
- Zustand 状态管理

### 后端
- FastAPI + Python 3.11
- SQLAlchemy ORM + SQLite数据库
- 向量存储抽象层（支持FAISS、Milvus、Qdrant）
- 模块化模型管理中心（支持OpenAI、Ollama、BGE等）
- Loguru 日志管理

## 快速开始

### 环境要求
- Node.js 22
- Python 3.11
- Conda环境管理

### 前端开发

```bash
# 进入前端目录
cd front

# 安装依赖
pnpm install

# 启动开发服务器
pnpm run dev
```

前端将在 http://localhost:5173 启动

### 后端开发

```bash
# 激活conda环境
conda activate rag

# 安装Python依赖
pip install -r requirements.txt

# 启动后端服务
python main.py
```

后端将在 http://localhost:8000 启动

### 环境配置

1. 复制环境配置文件：
```bash
cp .env.example .env
```

2. 根据需要修改 `.env` 文件中的配置

## 核心功能

### 知识库管理
- 创建和管理多个知识库
- 配置分段策略、索引策略、检索策略
- 支持文档上传和批量处理
- 灵活的向量存储后端选择

### 模型管理
- 统一的模型管理中心
- 支持多种模型提供商（OpenAI、Ollama、BGE等）
- 动态模型切换和配置管理
- 模型性能监控和统计

### 文档处理
- 支持PDF、TXT、HTML、Markdown格式
- 智能文档解析和预处理
- 父子分段模型，平衡检索精度和生成质量

### 向量检索
- 多种向量数据库支持（FAISS、Milvus、Qdrant）
- 支持向量检索、全文检索、混合检索
- MMR去冗余和多库融合
- 统一的向量存储抽象接口

### 检索测试
- 实时检索测试和结果展示
- 多种检索策略对比
- 检索结果可视化分析

## API文档

启动后端服务后，访问以下地址查看API文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 开发指南

### 前端开发注意事项
- 前端代码必须在 `front/` 目录下运行
- 使用 TypeScript 进行类型安全开发
- 遵循 Ant Design 设计规范

### 后端开发注意事项
- 后端开发需要激活 `conda activate rag` 环境
- 遵循 FastAPI 最佳实践
- 使用异步编程模式
- 遵循简化的DDD设计原则

## 许可证

MIT License
