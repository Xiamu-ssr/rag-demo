"""API主路由"""

from fastapi import APIRouter

from app.api.endpoints import collections, documents, search, index, processing, system

# 创建主路由
api_router = APIRouter()

# 注册各模块路由
api_router.include_router(
    collections.router,
    prefix="/collections",
    tags=["collections"]
)

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    search.router,
    prefix="/search",
    tags=["search"]
)

api_router.include_router(
    index.router,
    prefix="/index",
    tags=["index"]
)

api_router.include_router(
    processing.router,
    prefix="/processing",
    tags=["processing"]
)

api_router.include_router(
    system.router,
    prefix="/system",
    tags=["system"]
)