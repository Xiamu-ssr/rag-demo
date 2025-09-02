import { AppSidebar } from "@/components/app-sidebar"
import { SystemSettings } from "@/components/system-settings"
import { SidebarProvider } from "@/components/ui/sidebar"

export default function SystemSettingsPage() {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <main className="flex-1 overflow-auto">
          <SystemSettings />
        </main>
      </div>
    </SidebarProvider>
  )
}
