"use client"

import { OceanCurrentMap } from "@/components/ocean-current-map"

export default function LiveMapPage() {
  return (
    <div className="min-h-screen pt-20 relative overflow-hidden bg-black">
      <div className="container mx-auto px-4 py-12 relative z-10">
        <div className="mb-12 text-center">
          <h1 className="text-5xl md:text-6xl font-sentient mb-4">
            Live <i className="font-light">Plastic Drift</i> Map
          </h1>
          <p className="text-xl text-foreground/60 max-w-3xl mx-auto leading-relaxed">
            Real-time ocean plastic drift predictions powered by distributed volunteer computing
          </p>
        </div>

        <div className="relative">
          <div className="relative overflow-hidden rounded-xl border border-foreground/10" style={{ height: "70vh", minHeight: "500px" }}>
            <OceanCurrentMap />
          </div>
        </div>

        <div className="mt-8 grid md:grid-cols-3 gap-6">
          <div className="bg-foreground/5 border border-foreground/10 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse" />
              <h3 className="font-bold">Live simulation</h3>
            </div>
            <p className="text-sm text-foreground/70">3,000+ trajectory simulations running in real-time across the North Atlantic</p>
          </div>

          <div className="bg-foreground/5 border border-foreground/10 rounded-lg p-6">
            <h3 className="font-bold mb-4">Current Flow Legend</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-8 h-1 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full" />
                <span className="text-foreground/70">Slow (0-1 mph)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-1.5 bg-gradient-to-r from-cyan-400 to-blue-300 rounded-full" />
                <span className="text-foreground/70">Medium (1-2 mph)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-2 bg-gradient-to-r from-blue-300 to-white rounded-full" />
                <span className="text-foreground/70">Fast (2+ mph)</span>
              </div>
            </div>
          </div>

          <div className="bg-foreground/5 border border-foreground/10 rounded-lg p-6">
            <h3 className="font-bold mb-4">Active Regions</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-foreground/70">North Atlantic Gyre</span>
                <span className="text-red-400">High</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground/70">Caribbean Basin</span>
                <span className="text-green-400">Normal</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
