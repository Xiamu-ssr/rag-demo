"""文档处理和测试API端点"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/split/test")
async def test_split():
    """分段测试"""
    return {"message": "分段测试功能待实现"}


@router.post("/parse/test")
async def test_parse():
    """文档解析测试"""
    return {"message": "文档解析测试功能待实现"}