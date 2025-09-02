import { AppSidebar } from "@/components/app-sidebar"
import { DocumentCenter } from "@/components/document-center"
import { SidebarProvider } from "@/components/ui/sidebar"

export default function DocumentCenterPage() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-auto">
          <DocumentCenter />
        </main>
      </div>
    </SidebarProvider>
  )
}
