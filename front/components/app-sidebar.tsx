"use client"

import { Database, FileText, Search, Settings, HardDrive, Brain, BarChart3 } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar"

const menuItems = [
  {
    title: "仪表盘",
    url: "/dashboard",
    icon: BarChart3,
  },
  {
    title: "知识库",
    url: "/kb",
    icon: Database,
  },
  {
    title: "文档中心",
    url: "/docs",
    icon: FileText,
  },
  {
    title: "检索控制台",
    url: "/search",
    icon: Search,
  },
  {
    title: "模型中心",
    url: "/models",
    icon: Brain,
  },
  {
    title: "向量存储",
    url: "/vector-stores",
    icon: HardDrive,
  },
  {
    title: "系统设置",
    url: "/settings",
    icon: Settings,
  },
]

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader className="border-b border-sidebar-border p-4">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Database className="h-4 w-4" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold">RAG 知识库</span>
            <span className="text-xs text-muted-foreground">企业版</span>
          </div>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>主要功能</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <a href={item.url}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}
