"use client"

import { MapPin, Activity, Radio, Layers, Info, Zap } from "lucide-react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { AnimatedParticles } from "@/components/animated-particles"
import { Badge } from "@/components/ui/badge"
import { OceanCurrentMap } from "@/components/ocean-current-map"

export default function LiveMapPage() {
  const activeRegions = [
    { name: "North Pacific Gyre", particles: 2847, status: "High Activity", color: "text-red-400" },
    { name: "Mediterranean Sea", particles: 1523, status: "Moderate", color: "text-yellow-400" },
    { name: "Indian Ocean", particles: 3142, status: "High Activity", color: "text-red-400" },
    { name: "Caribbean Basin", particles: 891, status: "Normal", color: "text-green-400" },
  ]

  return (
    <div className="min-h-screen pt-20 relative overflow-hidden">
      <AnimatedParticles />
      <div className="absolute inset-0 bg-gradient-to-b from-blue-950/20 via-cyan-950/10 to-background" />

      <div className="container mx-auto px-4 py-12 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-12 text-center"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 backdrop-blur-sm mb-6">
            <Radio className="h-4 w-4 text-green-400 animate-pulse" />
            <span className="text-sm text-green-300">Live Data Feed Active</span>
          </div>

          <h1 className="text-5xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Live Plastic Drift Map
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Real-time ocean plastic drift predictions powered by distributed volunteer computing
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-6 mb-8">
          {[
            { icon: Activity, label: "Simulations Running", value: "2,847", color: "from-blue-500/20 to-cyan-500/20" },
            { icon: MapPin, label: "Tracked Regions", value: "98", color: "from-purple-500/20 to-pink-500/20" },
            { icon: Layers, label: "Data Layers", value: "12", color: "from-green-500/20 to-emerald-500/20" },
            { icon: Zap, label: "Update Rate", value: "15s", color: "from-yellow-500/20 to-orange-500/20" },
          ].map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ scale: 1.05, y: -5 }}
            >
              <Card className={`p-6 bg-gradient-to-br ${stat.color} backdrop-blur-xl border-border/50`}>
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${stat.color}`}>
                    <stat.icon className="h-6 w-6 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-3xl font-bold text-white">{stat.value}</div>
                    <div className="text-sm text-muted-foreground">{stat.label}</div>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="relative"
        >
          <Card className="relative overflow-hidden bg-gradient-to-br from-blue-950/40 to-cyan-950/40 backdrop-blur-xl border-blue-500/20">
            <div className="relative" style={{ height: "70vh", minHeight: "500px" }}>
              <OceanCurrentMap />

              {/* Overlay controls and legend */}
              <div className="absolute top-4 left-4 z-10 space-y-3">
                <div className="bg-background/80 backdrop-blur-md rounded-lg p-4 border border-blue-500/20">
                  <h3 className="text-sm font-semibold text-blue-300 mb-3">Current Flow Legend</h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-1 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full" />
                      <span className="text-muted-foreground">Slow (0-1 mph)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-1.5 bg-gradient-to-r from-cyan-400 to-blue-300 rounded-full" />
                      <span className="text-muted-foreground">Medium (1-2 mph)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-2 bg-gradient-to-r from-blue-300 to-white rounded-full" />
                      <span className="text-muted-foreground">Fast (2+ mph)</span>
                    </div>
                  </div>
                </div>

                <div className="bg-background/80 backdrop-blur-md rounded-lg p-3 border border-blue-500/20">
                  <div className="flex items-center gap-2 text-xs text-green-400">
                    <Radio className="h-3 w-3 animate-pulse" />
                    <span>Live simulation</span>
                  </div>
                </div>
              </div>

              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-4 flex-wrap justify-center">
                <Badge variant="secondary" className="px-4 py-2 bg-blue-500/20 text-blue-300 border-blue-400/30 backdrop-blur-md">
                  <Layers className="h-4 w-4 mr-2" />
                  North Atlantic Gyre
                </Badge>
                <Badge variant="secondary" className="px-4 py-2 bg-cyan-500/20 text-cyan-300 border-cyan-400/30 backdrop-blur-md">
                  <Activity className="h-4 w-4 mr-2" />
                  3,000+ Trajectories
                </Badge>
                <Badge variant="secondary" className="px-4 py-2 bg-purple-500/20 text-purple-300 border-purple-400/30 backdrop-blur-md">
                  <MapPin className="h-4 w-4 mr-2" />
                  Plastic Accumulation
                </Badge>
              </div>
            </div>
          </Card>
        </motion.div>

        <div className="mt-8 grid md:grid-cols-2 gap-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-xl border-border/50">
              <div className="flex items-start gap-3 mb-4">
                <Info className="h-6 w-6 text-blue-400 shrink-0 mt-1" />
                <div>
                  <h3 className="font-bold text-xl mb-2">About the Map</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    This visualization will display real-time plastic drift predictions based on Monte Carlo simulations
                    running on volunteer computers worldwide. Data is updated every 15 seconds.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.7 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-xl border-border/50">
              <h3 className="font-bold text-xl mb-4">Active Regions</h3>
              <div className="space-y-3">
                {activeRegions.map((region, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: 0.8 + index * 0.1 }}
                    className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-blue-500/5 to-cyan-500/5 border border-border/50"
                  >
                    <div>
                      <div className="font-semibold">{region.name}</div>
                      <div className="text-sm text-muted-foreground">{region.particles} active simulations</div>
                    </div>
                    <Badge className={`${region.color} bg-transparent border-current`}>{region.status}</Badge>
                  </motion.div>
                ))}
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
