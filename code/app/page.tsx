"use client"

import { Download, Cpu, Globe, Users, Waves, TrendingDown, Sparkles, Shield, Zap, ArrowRight } from "lucide-react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollVideoBackground } from "@/components/scroll-video-background"
import { AnimatedParticles, FloatingBubbles } from "@/components/animated-particles"
import { StatsSection } from "@/components/stats-section"

const fadeInUp = {
  initial: { opacity: 0, y: 60 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 },
}

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
}

export default function HomePage() {
  return (
    <div className="flex flex-col overflow-hidden">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <ScrollVideoBackground />
        <AnimatedParticles />
        <FloatingBubbles />

        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/50 to-background" />

        <motion.div
          initial="initial"
          animate="animate"
          variants={staggerContainer}
          className="container mx-auto px-4 relative z-10 text-center"
        >
          <motion.div
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 backdrop-blur-sm mb-8"
          >
            <Sparkles className="h-4 w-4 text-blue-400" />
            <span className="text-sm text-blue-300">Powered by Distributed Computing</span>
          </motion.div>

          <motion.h1
            variants={fadeInUp}
            className="text-balance text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight mb-6"
          >
            <span className="text-white">Predict </span>
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Ocean Plastic
            </span>
            <br />
            <span className="text-white">Drift in Real-Time</span>
          </motion.h1>

          <motion.p
            variants={fadeInUp}
            className="mt-6 text-xl md:text-2xl leading-relaxed text-blue-100/80 max-w-4xl mx-auto"
          >
            Join thousands of volunteers lending computing power to track ocean plastics and protect marine life through
            advanced drift forecasting
          </motion.p>

          <motion.div variants={fadeInUp} className="mt-12 flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              className="gap-2 text-lg px-8 py-6 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 border-0 shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 transition-all"
            >
              <Download className="h-5 w-5" />
              Download Desktop App
              <ArrowRight className="h-5 w-5" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="gap-2 text-lg px-8 py-6 bg-background/20 backdrop-blur-sm border-blue-400/30 hover:bg-background/30 hover:border-blue-400/50"
            >
              Watch Demo
            </Button>
          </motion.div>

          <motion.div
            variants={fadeInUp}
            className="mt-16 flex items-center justify-center gap-8 text-sm text-muted-foreground"
          >
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-green-400" />
              <span>100% Open Source</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-yellow-400" />
              <span>Zero Setup Required</span>
            </div>
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-blue-400" />
              <span>15K+ Contributors</span>
            </div>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 border-2 border-blue-400/50 rounded-full p-1"
          >
            <div className="w-1.5 h-3 bg-blue-400 rounded-full mx-auto" />
          </motion.div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <StatsSection />

      {/* Problem Section */}
      <section className="py-24 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-red-950/10 to-transparent" />
        <div className="container mx-auto px-4 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-4xl"
          >
            <div className="mb-12 text-center">
              <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent">
                The Crisis
              </h2>
              <p className="text-xl text-muted-foreground leading-relaxed">
                8 million tons of plastic enter our oceans annually, threatening marine ecosystems
              </p>
            </div>

            <motion.div whileHover={{ scale: 1.02 }} transition={{ duration: 0.3 }}>
              <Card className="border-red-500/20 bg-gradient-to-br from-red-950/20 to-orange-950/20 backdrop-blur-xl">
                <CardContent className="pt-8">
                  <div className="flex flex-col md:flex-row items-start gap-6">
                    <div className="p-4 rounded-xl bg-gradient-to-br from-red-500/20 to-orange-500/20 shrink-0">
                      <TrendingDown className="h-8 w-8 text-red-400" />
                    </div>
                    <div>
                      <h3 className="font-bold text-2xl mb-3 text-red-300">Cleanup Crews Need Forecasts</h3>
                      <p className="text-lg text-muted-foreground leading-relaxed">
                        Conservation teams require real-time predictions of plastic accumulation zones. Traditional
                        large-scale ocean-drift simulations are computationally expensive and inaccessible to most
                        organizations, leaving them unable to optimize cleanup efforts.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-950/20 via-cyan-950/20 to-transparent" />
        <FloatingBubbles />

        <div className="container mx-auto px-4 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-6xl"
          >
            <div className="mb-16 text-center">
              <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Our Solution
              </h2>
              <p className="text-xl text-muted-foreground leading-relaxed max-w-3xl mx-auto">
                Democratizing ocean-drift forecasting through distributed volunteer computing
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              {[
                {
                  icon: Cpu,
                  title: "Distributed Computing",
                  description:
                    "Complex simulations divided into thousands of small Monte Carlo trajectories that run on volunteers' devices",
                  color: "from-purple-500/20 to-pink-500/20",
                  iconColor: "text-purple-400",
                },
                {
                  icon: Globe,
                  title: "Real-Time Forecasts",
                  description:
                    "Aggregated results create high-resolution plastic drift forecasts, enabling near-real-time predictions",
                  color: "from-blue-500/20 to-cyan-500/20",
                  iconColor: "text-blue-400",
                },
                {
                  icon: Users,
                  title: "Community Powered",
                  description:
                    "Uses community resources instead of expensive supercomputers, democratizing environmental modeling",
                  color: "from-green-500/20 to-emerald-500/20",
                  iconColor: "text-green-400",
                },
                {
                  icon: Waves,
                  title: "Physics & AI Informed",
                  description:
                    "Lightweight kernels combine ocean-current data, wind patterns, and AI models for accurate predictions",
                  color: "from-cyan-500/20 to-teal-500/20",
                  iconColor: "text-cyan-400",
                },
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 40 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="group relative"
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} rounded-2xl blur-xl group-hover:blur-2xl transition-all duration-300 opacity-50`} />
                  <Card className="relative bg-card/50 backdrop-blur-xl border-border/50 hover:border-border transition-all duration-300 h-full">
                    <CardHeader className="space-y-4">
                      <div className={`p-3 rounded-xl bg-gradient-to-br ${feature.color} w-fit`}>
                        <feature.icon className={`h-8 w-8 ${feature.iconColor}`} />
                      </div>
                      <CardTitle className="text-2xl">{feature.title}</CardTitle>
                      <CardDescription className="text-base leading-relaxed">{feature.description}</CardDescription>
                    </CardHeader>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Impact Section */}
      <section className="py-24 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-green-950/10 to-transparent" />
        <div className="container mx-auto px-4 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-4xl"
          >
            <div className="mb-16 text-center">
              <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                Global Impact
              </h2>
              <p className="text-xl text-muted-foreground leading-relaxed">
                Protecting marine ecosystems and coastal communities worldwide
              </p>
            </div>

            <div className="space-y-6">
              {[
                {
                  title: "Intercept Plastic Early",
                  description:
                    "Rapid forecasts help governments, NGOs, and cleanup teams intercept plastic before it reaches sensitive habitats or shorelines.",
                  color: "from-blue-500/20 to-cyan-500/20",
                  borderColor: "border-l-blue-400",
                },
                {
                  title: "Support Marine Life",
                  description:
                    "By reducing the volume of plastic reaching coasts, we support marine biodiversity, sustainable fisheries, and healthier coastal communities.",
                  color: "from-green-500/20 to-emerald-500/20",
                  borderColor: "border-l-green-400",
                },
                {
                  title: "Measurable Results",
                  description:
                    "Success measured by forecast accuracy compared to satellite data, geographic coverage achieved, and direct adoption by marine cleanup initiatives.",
                  color: "from-purple-500/20 to-pink-500/20",
                  borderColor: "border-l-purple-400",
                },
              ].map((impact, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -40 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, x: 10 }}
                >
                  <Card className={`border-l-4 ${impact.borderColor} bg-gradient-to-br ${impact.color} backdrop-blur-xl`}>
                    <CardHeader>
                      <CardTitle className="text-2xl">{impact.title}</CardTitle>
                      <CardDescription className="text-base leading-relaxed text-muted-foreground">
                        {impact.description}
                      </CardDescription>
                    </CardHeader>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-950/10 via-purple-950/10 to-transparent" />
        <AnimatedParticles />

        <div className="container mx-auto px-4 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-4xl"
          >
            <div className="mb-16 text-center">
              <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Get Started in 3 Steps
              </h2>
              <p className="text-xl text-muted-foreground leading-relaxed">
                Join the fight against ocean plastic pollution today
              </p>
            </div>

            <div className="space-y-12">
              {[
                {
                  number: 1,
                  title: "Download the App",
                  description:
                    "Install our lightweight desktop application on your computer. It runs securely in the background without affecting your work.",
                  color: "from-blue-500/20 to-cyan-500/20",
                },
                {
                  number: 2,
                  title: "Run Simulations",
                  description:
                    "Your device receives small ocean-drift trajectory simulations using real ocean-current, wind, and wave data from NOAA and Copernicus Marine Service.",
                  color: "from-purple-500/20 to-pink-500/20",
                },
                {
                  number: 3,
                  title: "Make an Impact",
                  description:
                    "Results are aggregated into global plastic-drift probability maps that help cleanup crews focus their efforts where they're needed most.",
                  color: "from-green-500/20 to-emerald-500/20",
                },
              ].map((step, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -40 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ duration: 0.6, delay: index * 0.15 }}
                  className="flex gap-6 group"
                >
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 360 }}
                    transition={{ duration: 0.6 }}
                    className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br ${step.color} font-bold text-2xl text-white shadow-lg`}
                  >
                    {step.number}
                  </motion.div>
                  <div className="flex-1">
                    <h3 className="font-bold text-2xl mb-3 group-hover:text-blue-400 transition-colors">
                      {step.title}
                    </h3>
                    <p className="text-lg text-muted-foreground leading-relaxed">{step.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="mt-16 text-center"
            >
              <Button
                size="lg"
                className="gap-2 text-lg px-8 py-6 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 border-0 shadow-lg shadow-blue-500/50"
              >
                <Download className="h-5 w-5" />
                Download Desktop App
                <ArrowRight className="h-5 w-5" />
              </Button>
              <p className="mt-6 text-sm text-muted-foreground">
                Available for Windows, macOS, and Linux • Free and Open Source
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-border/50 py-16 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-t from-blue-950/20 to-transparent" />
        <AnimatedParticles />

        <div className="container mx-auto px-4 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-4xl text-center"
          >
            <div className="flex items-center justify-center gap-3 mb-6">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
                className="p-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20"
              >
                <Waves className="h-8 w-8 text-blue-400" />
              </motion.div>
              <span className="font-bold text-2xl bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Ocean Plastic Tracker
              </span>
            </div>

            <p className="text-muted-foreground mb-6">
              Computing for Social Good • Marine Conservation • Volunteer Network
            </p>

            <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-muted-foreground">
              <span>Illinois Institute of Technology</span>
              <span>•</span>
              <span>MIT Licensed</span>
              <span>•</span>
              <span>Open Source</span>
            </div>

            <p className="mt-8 text-sm text-muted-foreground/70">
              Democratizing ocean-cleanup intelligence through distributed computing
            </p>
          </motion.div>
        </div>
      </footer>
    </div>
  )
}
