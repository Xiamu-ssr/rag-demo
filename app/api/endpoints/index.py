"""索引管理API端点"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/build")
async def build_index():
    """构建索引"""
    return {"message": "索引构建功能待实现"}


@router.get("/status/{collection_id}")
async def get_index_status(collection_id: str):
    """查询索引状态"""
    return {"message": f"索引状态查询功能待实现: {collection_id}"}


@router.post("/query")
async def query_index():
    """直接查询索引"""
    return {"message": "索引查询功能待实现"}