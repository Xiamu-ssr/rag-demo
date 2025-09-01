"""检索查询API端点"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/query")
async def search_query():
    """智能检索"""
    return {"message": "智能检索功能待实现"}


@router.post("/test")
async def search_test():
    """检索测试"""
    return {"message": "检索测试功能待实现"}