import { AppSidebar } from "@/components/app-sidebar"
import { VectorStores } from "@/components/vector-stores"
import { SidebarProvider } from "@/components/ui/sidebar"

export default function VectorStoresPage() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-auto">
          <VectorStores />
        </main>
      </div>
    </SidebarProvider>
  )
}
