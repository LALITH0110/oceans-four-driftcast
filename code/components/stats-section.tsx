"use client"

import { useEffect, useRef, useState } from "react"
import { motion, useInView } from "framer-motion"
import CountUp from "react-countup"
import { Activity, Database, Globe, Users } from "lucide-react"

const stats = [
  {
    icon: Users,
    value: 15420,
    label: "Active Volunteers",
    suffix: "+",
    description: "Computing power contributors worldwide",
  },
  {
    icon: Database,
    value: 2847,
    label: "Simulations Running",
    suffix: "",
    description: "Real-time trajectory calculations",
  },
  {
    icon: Globe,
    value: 98,
    label: "Coverage Areas",
    suffix: "%",
    description: "Ocean regions monitored",
  },
  {
    icon: Activity,
    value: 456,
    label: "Cleanup Operations",
    suffix: "+",
    description: "Assisted by our forecasts",
  },
]

export function StatsSection() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  const [hasAnimated, setHasAnimated] = useState(false)

  useEffect(() => {
    if (isInView && !hasAnimated) {
      setHasAnimated(true)
    }
  }, [isInView, hasAnimated])

  return (
    <section ref={ref} className="relative py-24 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-blue-950/20 via-cyan-950/10 to-transparent" />

      <div className="container mx-auto px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Real-Time Impact
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Live statistics from our global volunteer computing network
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ scale: 1.05, y: -5 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl blur-xl group-hover:blur-2xl transition-all duration-300" />
              <div className="relative bg-card/50 backdrop-blur-xl border border-blue-500/20 rounded-xl p-6 hover:border-blue-400/40 transition-all duration-300">
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-500/20">
                    <stat.icon className="h-6 w-6 text-blue-400" />
                  </div>
                  <div className="h-2 w-2 rounded-full bg-green-400 animate-pulse" />
                </div>

                <div className="space-y-2">
                  <div className="text-3xl font-bold text-white">
                    {hasAnimated ? (
                      <CountUp end={stat.value} duration={2.5} separator="," suffix={stat.suffix} />
                    ) : (
                      "0"
                    )}
                  </div>
                  <div className="font-semibold text-blue-400">{stat.label}</div>
                  <div className="text-sm text-muted-foreground">{stat.description}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
