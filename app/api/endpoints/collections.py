"""知识库管理API端点"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.logging import app_logger
from app.db.models import Collection
from app.api.schemas.collections import (
    CollectionCreate,
    CollectionUpdate,
    CollectionResponse,
    CollectionListResponse
)

router = APIRouter()


@router.post("/", response_model=CollectionResponse)
async def create_collection(
    collection_data: CollectionCreate,
    db: AsyncSession = Depends(get_async_session)
):
    """创建知识库"""
    try:
        # 检查名称是否已存在
        existing = await db.execute(
            select(Collection).where(Collection.name == collection_data.name)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="知识库名称已存在")
        
        # 创建新知识库
        collection = Collection(
            name=collection_data.name,
            description=collection_data.description,
            splitting_config=collection_data.splitting_config,
            index_config=collection_data.index_config,
            retrieval_config=collection_data.retrieval_config
        )
        
        db.add(collection)
        await db.commit()
        await db.refresh(collection)
        
        app_logger.info(f"Created collection: {collection.id}")
        return collection
        
    except Exception as e:
        app_logger.error(f"Error creating collection: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="创建知识库失败")


@router.get("/", response_model=CollectionListResponse)
async def list_collections(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_async_session)
):
    """获取知识库列表"""
    try:
        from sqlalchemy import select, func
        
        # 构建查询
        query = select(Collection)
        count_query = select(func.count(Collection.id))
        
        if search:
            search_filter = Collection.name.contains(search)
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)
        
        # 获取总数
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # 分页查询
        offset = (page - 1) * size
        query = query.offset(offset).limit(size).order_by(Collection.created_at.desc())
        
        result = await db.execute(query)
        collections = result.scalars().all()
        
        return CollectionListResponse(
            collections=collections,
            total=total,
            page=page,
            size=size
        )
        
    except Exception as e:
        app_logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail="获取知识库列表失败")


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    """获取知识库详情"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        collection = result.scalar_one_or_none()
        
        if not collection:
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        return collection
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error getting collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail="获取知识库详情失败")


@router.put("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    collection_data: CollectionUpdate,
    db: AsyncSession = Depends(get_async_session)
):
    """更新知识库"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        collection = result.scalar_one_or_none()
        
        if not collection:
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        # 更新字段
        update_data = collection_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(collection, field, value)
        
        await db.commit()
        await db.refresh(collection)
        
        app_logger.info(f"Updated collection: {collection_id}")
        return collection
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error updating collection {collection_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="更新知识库失败")


@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: str,
    db: AsyncSession = Depends(get_async_session)
):
    """删除知识库"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        collection = result.scalar_one_or_none()
        
        if not collection:
            raise HTTPException(status_code=404, detail="知识库不存在")
        
        await db.delete(collection)
        await db.commit()
        
        app_logger.info(f"Deleted collection: {collection_id}")
        return {"message": "知识库删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error deleting collection {collection_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="删除知识库失败")