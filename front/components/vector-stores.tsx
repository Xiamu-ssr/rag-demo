"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { HardDrive, Database, Zap, Settings, Plus, BarChart3 } from "lucide-react"

const vectorStores = [
  {
    id: 1,
    name: "主向量库",
    type: "Pinecone",
    dimensions: 1536,
    vectors: 125000,
    capacity: 200000,
    status: "运行中",
    usage: 62.5,
  },
  {
    id: 2,
    name: "备份向量库",
    type: "Weaviate",
    dimensions: 768,
    vectors: 89000,
    capacity: 150000,
    status: "运行中",
    usage: 59.3,
  },
  {
    id: 3,
    name: "测试向量库",
    type: "Chroma",
    dimensions: 1536,
    vectors: 5000,
    capacity: 50000,
    status: "离线",
    usage: 10.0,
  },
]

export function VectorStores() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-balance">向量存储</h1>
          <p className="text-muted-foreground">管理向量数据库和存储配置</p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          添加向量库
        </Button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">总向量数</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">219K</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+2.1K</span> 本周新增
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">存储使用率</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">54.7%</div>
            <p className="text-xs text-muted-foreground">平均使用率</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">查询性能</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">45ms</div>
            <p className="text-xs text-muted-foreground">平均查询时间</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">活跃连接</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">当前活跃连接数</p>
          </CardContent>
        </Card>
      </div>

      {/* Vector Stores List */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">向量库列表</h2>
        <div className="grid gap-4">
          {vectorStores.map((store) => (
            <Card key={store.id}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                      <HardDrive className="h-6 w-6 text-primary" />
                    </div>
                    <div className="space-y-2">
                      <h3 className="font-semibold">{store.name}</h3>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <span>类型: {store.type}</span>
                        <span>维度: {store.dimensions}</span>
                        <span>向量数: {store.vectors.toLocaleString()}</span>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span>存储使用率</span>
                          <span>{store.usage}%</span>
                        </div>
                        <Progress value={store.usage} className="w-64" />
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <Badge variant={store.status === "运行中" ? "default" : "secondary"}>{store.status}</Badge>

                    <div className="flex gap-2">
                      <Button variant="outline" size="sm">
                        <Settings className="h-4 w-4 mr-2" />
                        配置
                      </Button>
                      <Button variant="outline" size="sm">
                        <BarChart3 className="h-4 w-4 mr-2" />
                        监控
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
