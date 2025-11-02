import { MapPin } from "lucide-react"

export default function LiveMapPage() {
  return (
    <div className="flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center p-8">
      <div className="mx-auto max-w-2xl text-center">
        <div className="mb-6 flex justify-center">
          <div className="rounded-full bg-primary/10 p-6">
            <MapPin className="h-12 w-12 text-primary" />
          </div>
        </div>
        <h1 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl mb-4">Live Plastic Drift Map</h1>
        <p className="text-pretty text-lg text-muted-foreground leading-relaxed">
          The live map visualization will be displayed here. This feature is currently under development and will show
          real-time ocean plastic drift predictions based on volunteer computing results.
        </p>
        <div className="mt-8 rounded-lg border-2 border-dashed border-border bg-muted/30 p-12">
          <p className="text-sm text-muted-foreground">Map visualization coming soon</p>
        </div>
      </div>
    </div>
  )
}
