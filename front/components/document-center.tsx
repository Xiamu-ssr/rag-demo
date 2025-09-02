"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FileText, Upload, Search, Filter, MoreHorizontal, Eye, Download, Trash2 } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

const documents = [
  {
    id: 1,
    name: "产品使用手册.pdf",
    type: "PDF",
    size: "2.3 MB",
    status: "已处理",
    chunks: 45,
    uploadTime: "2024-01-15 14:30",
    knowledgeBase: "产品文档",
  },
  {
    id: 2,
    name: "API 接口文档.md",
    type: "Markdown",
    size: "856 KB",
    status: "处理中",
    chunks: 0,
    uploadTime: "2024-01-15 13:45",
    knowledgeBase: "技术文档",
  },
  {
    id: 3,
    name: "常见问题解答.docx",
    type: "Word",
    size: "1.2 MB",
    status: "已处理",
    chunks: 28,
    uploadTime: "2024-01-14 16:20",
    knowledgeBase: "客服知识库",
  },
]

export function DocumentCenter() {
  const [searchQuery, setSearchQuery] = useState("")

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-balance">文档中心</h1>
          <p className="text-muted-foreground">管理和处理知识库文档</p>
        </div>
        <Button>
          <Upload className="h-4 w-4 mr-2" />
          上传文档
        </Button>
      </div>

      <Tabs defaultValue="all" className="space-y-4">
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="all">全部文档</TabsTrigger>
            <TabsTrigger value="processing">处理中</TabsTrigger>
            <TabsTrigger value="completed">已完成</TabsTrigger>
            <TabsTrigger value="failed">处理失败</TabsTrigger>
          </TabsList>

          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="搜索文档..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 w-64"
              />
            </div>
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <TabsContent value="all" className="space-y-4">
          <div className="grid gap-4">
            {documents.map((doc) => (
              <Card key={doc.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                        <FileText className="h-5 w-5" />
                      </div>
                      <div className="space-y-1">
                        <h3 className="font-medium">{doc.name}</h3>
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span>{doc.type}</span>
                          <span>{doc.size}</span>
                          <span>分段: {doc.chunks}</span>
                          <span>{doc.uploadTime}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Badge
                        variant={
                          doc.status === "已处理" ? "default" : doc.status === "处理中" ? "secondary" : "destructive"
                        }
                      >
                        {doc.status}
                      </Badge>
                      <Badge variant="outline">{doc.knowledgeBase}</Badge>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>
                            <Eye className="h-4 w-4 mr-2" />
                            预览
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Download className="h-4 w-4 mr-2" />
                            下载
                          </DropdownMenuItem>
                          <DropdownMenuItem className="text-destructive">
                            <Trash2 className="h-4 w-4 mr-2" />
                            删除
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
