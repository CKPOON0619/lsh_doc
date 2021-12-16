'''
Interfaces for app api
'''

from pydantic import BaseModel
from typing import Optional, List


class ItemInfo(BaseModel):
    itemId: str
    itemType: str
    contentType: str
    content: str


class TenantRegistry(BaseModel):
    tenantId: str


class UserRegistry(BaseModel):
    email: str


class TenantUserRegistry(BaseModel):
    tenantId: str
    userEmail: str


class TenantItemRegistry(BaseModel):
    tenantId: str
    userEmail: str
    items: List[ItemInfo]


class TenantTableItemRegistry(BaseModel):
    tenantId: str
    userEmail: str
    contentType: str
    itemType: str
    embeddingType: str
    hash_dim: int
    itemIds: Optional[List[str]]


class TenantItemDelete(BaseModel):
    tenantId: str
    userEmail: str
    contentType: str
    itemType: str
    itemIds: List[str]


class TenantTableRefresh(BaseModel):
    tenantId: str
    userEmail: str
    contentType: str
    itemType: str
    embeddingType: str
    hash_dim: int
    num_tables: int


class TenantItemQueryWithIds(BaseModel):
    tenantId: str
    userEmail: str
    contentType: str
    itemType: str
    embeddingType: str
    queryTableNum: int
    hash_dim: int
    itemIds: List[str]


class TenantItemQueryWithContents(BaseModel):
    tenantId: str
    userEmail: str
    contentType: str
    itemType: str
    embeddingType: str
    queryTableNum: int
    hash_dim: int
    contents: List[str]
