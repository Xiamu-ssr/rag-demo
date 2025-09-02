import { AppSidebar } from "@/components/app-sidebar"
import { ModelCenter } from "@/components/model-center"
import { SidebarProvider } from "@/components/ui/sidebar"

export default function ModelCenterPage() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-auto">
          <ModelCenter />
        </main>
      </div>
    </SidebarProvider>
  )
}
