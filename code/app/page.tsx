import { Download, Cpu, Globe, Users, Waves, TrendingDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollVideoBackground } from "@/components/scroll-video-background"

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 md:py-32">
        <ScrollVideoBackground />

        <div className="container mx-auto px-4 relative z-10">
          <div className="mx-auto max-w-3xl text-center">
            <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl text-white drop-shadow-lg">
              Predict Ocean Plastic Drift with <span className="text-primary drop-shadow-lg">Volunteer Computing</span>
            </h1>
            <p className="mt-6 text-pretty text-lg sm:text-xl leading-relaxed text-white/90 drop-shadow-md">
              Help cleanup crews intercept plastic before it reaches sensitive marine habitats. Contribute your
              computer's idle power to run ocean-drift simulations and protect our oceans.
            </p>
            <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:justify-center">
              <Button size="lg" className="gap-2 text-base">
                <Download className="h-5 w-5" />
                Download Desktop App
              </Button>
              <Button size="lg" variant="outline" className="gap-2 text-base bg-transparent">
                Learn More
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-16 md:py-24">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-3xl">
            <div className="mb-12 text-center">
              <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">The Problem</h2>
              <p className="mt-4 text-pretty text-lg text-muted-foreground leading-relaxed">
                Millions of tons of plastic enter the ocean every year, forming harmful accumulation zones that kill
                marine life and disrupt coastal economies.
              </p>
            </div>

            <Card className="border-2">
              <CardContent className="pt-6">
                <div className="flex flex-col gap-6">
                  <div className="flex items-start gap-4">
                    <div className="rounded-lg bg-destructive/10 p-3">
                      <TrendingDown className="h-6 w-6 text-destructive" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-lg">Cleanup Crews Need Forecasts</h3>
                      <p className="mt-2 text-muted-foreground leading-relaxed">
                        Cleanup teams need short-term predictions of where floating plastic will concentrate, but
                        running large-scale ocean-drift simulations in real time is computationally expensive and
                        inaccessible to most conservation groups.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="bg-muted/50 py-16 md:py-24">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-3xl">
            <div className="mb-12 text-center">
              <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">Our Innovation</h2>
              <p className="mt-4 text-pretty text-lg text-muted-foreground leading-relaxed">
                A volunteer-computing platform that makes ocean-drift forecasting fast, affordable, and accessible.
              </p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2">
              <Card>
                <CardHeader>
                  <Cpu className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>Distributed Computing</CardTitle>
                  <CardDescription>
                    Complex simulations divided into thousands of small Monte Carlo trajectories that run on volunteers'
                    devices
                  </CardDescription>
                </CardHeader>
              </Card>

              <Card>
                <CardHeader>
                  <Globe className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>Real-Time Forecasts</CardTitle>
                  <CardDescription>
                    Aggregated results create high-resolution plastic drift forecasts, enabling near-real-time
                    predictions
                  </CardDescription>
                </CardHeader>
              </Card>

              <Card>
                <CardHeader>
                  <Users className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>Community Powered</CardTitle>
                  <CardDescription>
                    Uses community resources instead of expensive supercomputers, democratizing environmental modeling
                  </CardDescription>
                </CardHeader>
              </Card>

              <Card>
                <CardHeader>
                  <Waves className="h-8 w-8 text-primary mb-2" />
                  <CardTitle>Physics & AI Informed</CardTitle>
                  <CardDescription>
                    Lightweight kernels combine ocean-current data, wind patterns, and AI models for accurate
                    predictions
                  </CardDescription>
                </CardHeader>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Impact Section */}
      <section className="py-16 md:py-24">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-3xl">
            <div className="mb-12 text-center">
              <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">Social Impact</h2>
              <p className="mt-4 text-pretty text-lg text-muted-foreground leading-relaxed">
                Helping protect marine biodiversity and coastal communities worldwide.
              </p>
            </div>

            <div className="space-y-6">
              <Card className="border-l-4 border-l-primary">
                <CardHeader>
                  <CardTitle>Intercept Plastic Early</CardTitle>
                  <CardDescription className="text-base leading-relaxed">
                    Rapid forecasts help governments, NGOs, and cleanup teams intercept plastic before it reaches
                    sensitive habitats or shorelines.
                  </CardDescription>
                </CardHeader>
              </Card>

              <Card className="border-l-4 border-l-secondary">
                <CardHeader>
                  <CardTitle>Support Marine Life</CardTitle>
                  <CardDescription className="text-base leading-relaxed">
                    By reducing the volume of plastic reaching coasts, we support marine biodiversity, sustainable
                    fisheries, and healthier coastal communities.
                  </CardDescription>
                </CardHeader>
              </Card>

              <Card className="border-l-4 border-l-accent">
                <CardHeader>
                  <CardTitle>Measurable Results</CardTitle>
                  <CardDescription className="text-base leading-relaxed">
                    Success measured by forecast accuracy compared to satellite data, geographic coverage achieved, and
                    direct adoption by marine cleanup initiatives.
                  </CardDescription>
                </CardHeader>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-muted/50 py-16 md:py-24">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-3xl">
            <div className="mb-12 text-center">
              <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">How You Can Contribute</h2>
              <p className="mt-4 text-pretty text-lg text-muted-foreground leading-relaxed">
                Join thousands of volunteers lending their computing power to track ocean plastics.
              </p>
            </div>

            <div className="space-y-8">
              <div className="flex gap-6">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold text-lg">
                  1
                </div>
                <div>
                  <h3 className="font-semibold text-xl mb-2">Download the App</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Install our lightweight desktop application on your computer. It runs securely in the background
                    without affecting your work.
                  </p>
                </div>
              </div>

              <div className="flex gap-6">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold text-lg">
                  2
                </div>
                <div>
                  <h3 className="font-semibold text-xl mb-2">Run Simulations</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Your device receives small ocean-drift trajectory simulations using real ocean-current, wind, and
                    wave data from NOAA and Copernicus Marine Service.
                  </p>
                </div>
              </div>

              <div className="flex gap-6">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold text-lg">
                  3
                </div>
                <div>
                  <h3 className="font-semibold text-xl mb-2">Make an Impact</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Results are aggregated into global plastic-drift probability maps that help cleanup crews focus
                    their efforts where they're needed most.
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-12 text-center">
              <Button size="lg" className="gap-2 text-base">
                <Download className="h-5 w-5" />
                Download Desktop App
              </Button>
              <p className="mt-4 text-sm text-muted-foreground">Available for Windows, macOS, and Linux</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12">
        <div className="container mx-auto px-4">
          <div className="mx-auto max-w-3xl text-center">
            <div className="flex items-center justify-center gap-2 mb-4">
              <Waves className="h-6 w-6 text-primary" />
              <span className="font-semibold text-lg">Ocean Plastic Tracker</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Computing for Social Good • Marine Conservation • Volunteer Network
            </p>
            <p className="mt-4 text-xs text-muted-foreground">
              Democratizing ocean-cleanup intelligence through distributed computing
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
