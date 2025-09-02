"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, Zap, Settings, Plus, Activity } from "lucide-react"

const models = [
  {
    id: 1,
    name: "GPT-4",
    provider: "OpenAI",
    type: "文本生成",
    status: "运行中",
    usage: "高",
    description: "强大的大语言模型，适用于复杂的文本理解和生成任务",
  },
  {
    id: 2,
    name: "text-embedding-ada-002",
    provider: "OpenAI",
    type: "文本嵌入",
    status: "运行中",
    usage: "中",
    description: "高质量的文本向量化模型，用于语义检索",
  },
  {
    id: 3,
    name: "Claude-3",
    provider: "Anthropic",
    type: "文本生成",
    status: "离线",
    usage: "低",
    description: "安全可靠的AI助手，擅长分析和推理",
  },
]

export function ModelCenter() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-balance">模型中心</h1>
          <p className="text-muted-foreground">管理和配置AI模型服务</p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          添加模型
        </Button>
      </div>

      <Tabs defaultValue="models" className="space-y-4">
        <TabsList>
          <TabsTrigger value="models">模型列表</TabsTrigger>
          <TabsTrigger value="usage">使用统计</TabsTrigger>
          <TabsTrigger value="settings">模型配置</TabsTrigger>
        </TabsList>

        <TabsContent value="models" className="space-y-4">
          <div className="grid gap-4">
            {models.map((model) => (
              <Card key={model.id}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                        <Brain className="h-6 w-6 text-primary" />
                      </div>
                      <div className="space-y-1">
                        <h3 className="font-semibold">{model.name}</h3>
                        <p className="text-sm text-muted-foreground">{model.description}</p>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{model.provider}</Badge>
                          <Badge variant="secondary">{model.type}</Badge>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="flex items-center gap-2">
                          <Activity className="h-4 w-4" />
                          <span className="text-sm font-medium">使用率: {model.usage}</span>
                        </div>
                        <Badge variant={model.status === "运行中" ? "default" : "secondary"} className="mt-1">
                          {model.status}
                        </Badge>
                      </div>

                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Settings className="h-4 w-4 mr-2" />
                          配置
                        </Button>
                        <Button variant="outline" size="sm">
                          <Zap className="h-4 w-4 mr-2" />
                          测试
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="usage" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium">今日调用</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">2,847</div>
                <p className="text-xs text-muted-foreground">
                  <span className="text-green-600">+12%</span> 较昨日
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium">本月消耗</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">$127.50</div>
                <p className="text-xs text-muted-foreground">预算剩余 $372.50</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium">平均响应时间</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">1.2s</div>
                <p className="text-xs text-muted-foreground">
                  <span className="text-green-600">-0.3s</span> 较上周
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
