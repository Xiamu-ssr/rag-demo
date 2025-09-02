import { AppSidebar } from "@/components/app-sidebar"
import { SearchConsole } from "@/components/search-console"
import { SidebarProvider } from "@/components/ui/sidebar"

export default function SearchConsolePage() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-auto">
          <SearchConsole />
        </main>
      </div>
    </SidebarProvider>
  )
}
