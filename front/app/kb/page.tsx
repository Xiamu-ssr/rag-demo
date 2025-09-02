import { AppSidebar } from "@/components/app-sidebar"
import { KnowledgeBaseList } from "@/components/knowledge-base-list"
import { SidebarProvider } from "@/components/ui/sidebar"

export default function KnowledgeBasePage() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-auto">
          <KnowledgeBaseList />
        </main>
      </div>
    </SidebarProvider>
  )
}
