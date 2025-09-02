"use client"

import { useState } from "react"
import { Plus, Search, MoreHorizontal, Database, Clock, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { CreateKnowledgeBaseDialog } from "@/components/create-knowledge-base-dialog"

const mockKnowledgeBases = [
  {
    id: "1",
    name: "产品文档库",
    description: "包含所有产品相关的技术文档和用户手册",
    documentCount: 156,
    model: "text-embedding-3-large",
    updatedAt: "2024-01-15 14:30",
    status: "active" as const,
  },
  {
    id: "2",
    name: "客服知识库",
    description: "客户服务常见问题和解决方案",
    documentCount: 89,
    model: "text-embedding-3-small",
    updatedAt: "2024-01-14 09:15",
    status: "building" as const,
  },
  {
    id: "3",
    name: "法律合规文档",
    description: "公司法律文件和合规要求",
    documentCount: 34,
    model: "text-embedding-3-large",
    updatedAt: "2024-01-13 16:45",
    status: "failed" as const,
  },
]

const statusConfig = {
  active: { label: "可用", variant: "default" as const, color: "bg-green-500" },
  building: { label: "构建中", variant: "secondary" as const, color: "bg-yellow-500" },
  failed: { label: "失败", variant: "destructive" as const, color: "bg-red-500" },
}

export function KnowledgeBaseList() {
  const [searchQuery, setSearchQuery] = useState("")
  const [showCreateDialog, setShowCreateDialog] = useState(false)

  const filteredKnowledgeBases = mockKnowledgeBases.filter(
    (kb) =>
      kb.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      kb.description.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-semibold text-balance">知识库管理</h1>
            <p className="text-muted-foreground">创建和管理您的知识库，支持多种文档格式和检索策略</p>
          </div>
          <Button onClick={() => setShowCreateDialog(true)} className="gap-2">
            <Plus className="h-4 w-4" />
            新建知识库
          </Button>
        </div>

        {/* Search */}
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="搜索知识库..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      {/* Knowledge Base Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredKnowledgeBases.map((kb) => (
          <Card key={kb.id} className="hover:shadow-md transition-shadow cursor-pointer">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                    <Database className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <CardTitle className="text-base">{kb.name}</CardTitle>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant={statusConfig[kb.status].variant}>
                        <div className={`w-2 h-2 rounded-full ${statusConfig[kb.status].color} mr-1`} />
                        {statusConfig[kb.status].label}
                      </Badge>
                    </div>
                  </div>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem>查看详情</DropdownMenuItem>
                    <DropdownMenuItem>编辑配置</DropdownMenuItem>
                    <DropdownMenuItem>重建索引</DropdownMenuItem>
                    <DropdownMenuItem className="text-destructive">删除</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <CardDescription className="mb-4 line-clamp-2">{kb.description}</CardDescription>

              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <FileText className="h-3 w-3" />
                  <span>{kb.documentCount} 个文档</span>
                </div>
                <div className="flex items-center gap-2">
                  <Database className="h-3 w-3" />
                  <span>{kb.model}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-3 w-3" />
                  <span>更新于 {kb.updatedAt}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredKnowledgeBases.length === 0 && (
        <div className="text-center py-12">
          <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium mb-2">暂无知识库</h3>
          <p className="text-muted-foreground mb-4">
            {searchQuery ? "没有找到匹配的知识库" : "开始创建您的第一个知识库"}
          </p>
          {!searchQuery && (
            <Button onClick={() => setShowCreateDialog(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              新建知识库
            </Button>
          )}
        </div>
      )}

      <CreateKnowledgeBaseDialog open={showCreateDialog} onOpenChange={setShowCreateDialog} />
    </div>
  )
}
