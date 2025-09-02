"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { Settings, Shield, Database, Bell, Users, Key } from "lucide-react"

export function SystemSettings() {
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-balance">系统设置</h1>
          <p className="text-muted-foreground">配置系统参数和安全选项</p>
        </div>
      </div>

      <Tabs defaultValue="general" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="general">常规设置</TabsTrigger>
          <TabsTrigger value="security">安全配置</TabsTrigger>
          <TabsTrigger value="database">数据库</TabsTrigger>
          <TabsTrigger value="notifications">通知</TabsTrigger>
          <TabsTrigger value="users">用户管理</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                基础配置
              </CardTitle>
              <CardDescription>系统基本参数设置</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="system-name">系统名称</Label>
                  <Input id="system-name" defaultValue="RAG 知识库系统" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="system-version">系统版本</Label>
                  <Input id="system-version" defaultValue="v1.0.0" disabled />
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>启用调试模式</Label>
                    <p className="text-sm text-muted-foreground">开启详细的系统日志记录</p>
                  </div>
                  <Switch />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>自动备份</Label>
                    <p className="text-sm text-muted-foreground">每日自动备份系统数据</p>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-4 w-4" />
                安全设置
              </CardTitle>
              <CardDescription>系统安全和访问控制配置</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>启用双因素认证</Label>
                    <p className="text-sm text-muted-foreground">为管理员账户启用2FA</p>
                  </div>
                  <Switch />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>API 访问限制</Label>
                    <p className="text-sm text-muted-foreground">限制API调用频率</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>数据加密</Label>
                    <p className="text-sm text-muted-foreground">对敏感数据进行加密存储</p>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label htmlFor="session-timeout">会话超时时间 (分钟)</Label>
                <Input id="session-timeout" type="number" defaultValue="30" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="database" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-4 w-4" />
                数据库配置
              </CardTitle>
              <CardDescription>数据库连接和性能设置</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="db-host">数据库主机</Label>
                  <Input id="db-host" defaultValue="localhost" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="db-port">端口</Label>
                  <Input id="db-port" type="number" defaultValue="5432" />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="db-name">数据库名称</Label>
                <Input id="db-name" defaultValue="rag_knowledge_base" />
              </div>

              <Separator />

              <div className="space-y-2">
                <Label htmlFor="connection-pool">连接池大小</Label>
                <Input id="connection-pool" type="number" defaultValue="10" />
              </div>

              <Button variant="outline">测试连接</Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-4 w-4" />
                通知设置
              </CardTitle>
              <CardDescription>配置系统通知和警报</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>邮件通知</Label>
                    <p className="text-sm text-muted-foreground">系统事件邮件提醒</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>错误警报</Label>
                    <p className="text-sm text-muted-foreground">系统错误实时通知</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>性能监控</Label>
                    <p className="text-sm text-muted-foreground">性能指标异常提醒</p>
                  </div>
                  <Switch />
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label htmlFor="admin-email">管理员邮箱</Label>
                <Input id="admin-email" type="email" defaultValue="admin@example.com" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="users" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-4 w-4" />
                用户管理
              </CardTitle>
              <CardDescription>用户权限和访问控制</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>允许用户注册</Label>
                    <p className="text-sm text-muted-foreground">开放用户自主注册</p>
                  </div>
                  <Switch />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>邮箱验证</Label>
                    <p className="text-sm text-muted-foreground">注册时需要邮箱验证</p>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <Label htmlFor="max-users">最大用户数</Label>
                <Input id="max-users" type="number" defaultValue="100" />
              </div>

              <Button>
                <Key className="h-4 w-4 mr-2" />
                重置API密钥
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
