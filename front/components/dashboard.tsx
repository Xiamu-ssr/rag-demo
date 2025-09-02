"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Database, FileText, Search, Brain, HardDrive, Activity, TrendingUp, Users, Clock } from "lucide-react"

export function Dashboard() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-balance">仪表盘</h1>
          <p className="text-muted-foreground">RAG 知识库系统概览</p>
        </div>
        <Button>
          <Activity className="h-4 w-4 mr-2" />
          系统状态
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">知识库总数</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+2</span> 本月新增
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">文档总数</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1,247</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+89</span> 本周新增
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">检索次数</CardTitle>
            <Search className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8,432</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+12%</span> 较上月
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">活跃用户</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">156</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+8</span> 本周活跃
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              最近活动
            </CardTitle>
            <CardDescription>系统最新操作记录</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 bg-green-500 rounded-full" />
                <div>
                  <p className="text-sm font-medium">新建知识库 "产品手册"</p>
                  <p className="text-xs text-muted-foreground">2 分钟前</p>
                </div>
              </div>
              <Badge variant="secondary">创建</Badge>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 bg-blue-500 rounded-full" />
                <div>
                  <p className="text-sm font-medium">上传文档 "API 文档.pdf"</p>
                  <p className="text-xs text-muted-foreground">15 分钟前</p>
                </div>
              </div>
              <Badge variant="secondary">上传</Badge>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 bg-purple-500 rounded-full" />
                <div>
                  <p className="text-sm font-medium">执行检索查询</p>
                  <p className="text-xs text-muted-foreground">1 小时前</p>
                </div>
              </div>
              <Badge variant="secondary">检索</Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              系统状态
            </CardTitle>
            <CardDescription>各组件运行状态</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Brain className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium">AI 模型服务</span>
              </div>
              <Badge className="bg-green-100 text-green-800">正常</Badge>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <HardDrive className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium">向量数据库</span>
              </div>
              <Badge className="bg-green-100 text-green-800">正常</Badge>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Search className="h-4 w-4 text-yellow-500" />
                <span className="text-sm font-medium">检索服务</span>
              </div>
              <Badge className="bg-yellow-100 text-yellow-800">维护中</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
