"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Search, Clock, Database, Zap, Settings } from "lucide-react"

const searchHistory = [
  {
    id: 1,
    query: "如何配置API接口",
    results: 12,
    time: "2024-01-15 14:30",
    knowledgeBase: "技术文档",
  },
  {
    id: 2,
    query: "产品价格政策",
    results: 8,
    time: "2024-01-15 13:45",
    knowledgeBase: "产品文档",
  },
  {
    id: 3,
    query: "常见故障排除",
    results: 15,
    time: "2024-01-15 12:20",
    knowledgeBase: "客服知识库",
  },
]

export function SearchConsole() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)

  const handleSearch = async () => {
    setIsSearching(true)
    // Simulate search
    setTimeout(() => {
      setResults([
        {
          id: 1,
          title: "API 接口配置指南",
          content: "本文档详细介绍了如何配置和使用我们的API接口，包括认证、请求格式、响应处理等关键步骤...",
          score: 0.95,
          source: "技术文档/API文档.md",
          knowledgeBase: "技术文档",
        },
        {
          id: 2,
          title: "接口调用示例",
          content: "以下是一些常用的API调用示例，展示了不同场景下的请求参数和响应格式...",
          score: 0.87,
          source: "技术文档/示例代码.md",
          knowledgeBase: "技术文档",
        },
      ])
      setIsSearching(false)
    }, 1500)
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-balance">检索控制台</h1>
          <p className="text-muted-foreground">测试和优化知识库检索效果</p>
        </div>
        <Button variant="outline">
          <Settings className="h-4 w-4 mr-2" />
          检索设置
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Search Interface */}
        <div className="lg:col-span-2 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-4 w-4" />
                智能检索
              </CardTitle>
              <CardDescription>输入查询内容，测试知识库检索效果</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">查询内容</label>
                <Textarea
                  placeholder="请输入您要查询的问题..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={3}
                />
              </div>
              <Button onClick={handleSearch} disabled={!query || isSearching} className="w-full">
                {isSearching ? (
                  <>
                    <Zap className="h-4 w-4 mr-2 animate-spin" />
                    检索中...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    开始检索
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Search Results */}
          {results.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>检索结果</CardTitle>
                <CardDescription>找到 {results.length} 个相关结果</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {results.map((result) => (
                  <div key={result.id} className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium">{result.title}</h3>
                      <Badge variant="secondary">相似度: {(result.score * 100).toFixed(1)}%</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-2">{result.content}</p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Database className="h-3 w-3" />
                      <span>{result.knowledgeBase}</span>
                      <span>•</span>
                      <span>{result.source}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Search History */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-4 w-4" />
                检索历史
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {searchHistory.map((item) => (
                <div key={item.id} className="border rounded-lg p-3 space-y-2">
                  <p className="text-sm font-medium">{item.query}</p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{item.results} 个结果</span>
                    <span>{item.time}</span>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {item.knowledgeBase}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
