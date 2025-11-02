"use client";

import Link from "next/link";
import { GL } from "./gl";
import { Pill } from "./pill";
import { Button } from "./ui/button";
import { useState } from "react";
import { EducationalSurvey } from "./educational-survey";

export function Hero() {
  const [hovering, setHovering] = useState(false);
  return (
    <>
      <EducationalSurvey />
      <div className="flex flex-col h-svh justify-between">
        <GL hovering={hovering} />

        <div className="pb-16 mt-auto text-center relative">
          <Pill className="mb-6">VOLUNTEER COMPUTING</Pill>
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-sentient">
            Predict <br />
            <i className="font-light">Ocean Plastic</i> Drift
          </h1>
          <p className="font-mono text-sm sm:text-base text-foreground/60 text-balance mt-8 max-w-[520px] mx-auto">
            Help cleanup crews intercept plastic before it reaches sensitive marine habitats. Contribute your computer's idle power to run ocean-drift simulations
          </p>

          <Link className="contents max-sm:hidden" href="/#download">
            <Button
              className="mt-14"
              onMouseEnter={() => setHovering(true)}
              onMouseLeave={() => setHovering(false)}
            >
              [Download Desktop App]
            </Button>
          </Link>
          <Link className="contents sm:hidden" href="/#download">
            <Button
              size="sm"
              className="mt-14"
              onMouseEnter={() => setHovering(true)}
              onMouseLeave={() => setHovering(false)}
            >
              [Download App]
            </Button>
          </Link>
        </div>
      </div>

      {/* Problem Section */}
      <section className="py-24 relative" id="problem">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-sentient mb-6">
              The <i className="font-light">Crisis</i>
            </h2>
            <p className="text-lg text-foreground/70 leading-relaxed mb-12">
              8 million tons of plastic enter our oceans annually, forming harmful accumulation zones that kill marine life and disrupt coastal economies
            </p>
            <div className="bg-foreground/5 border border-foreground/10 rounded-lg p-8 text-left">
              <h3 className="text-2xl font-bold mb-4 text-primary">Cleanup Crews Need Forecasts</h3>
              <p className="text-foreground/70 leading-relaxed">
                Conservation teams require real-time predictions of plastic accumulation zones. Traditional large-scale ocean-drift simulations are computationally expensive and inaccessible to most organizations, leaving them unable to optimize cleanup efforts.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-24 bg-foreground/5 relative" id="solution">
        <div className="container mx-auto px-4">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-sentient mb-6">
                Our <i className="font-light">Innovation</i>
              </h2>
              <p className="text-lg text-foreground/70 leading-relaxed">
                Democratizing ocean-drift forecasting through distributed volunteer computing
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {[
                {
                  title: "Distributed Computing",
                  description: "Complex simulations divided into thousands of small Monte Carlo trajectories running on volunteers' devices"
                },
                {
                  title: "Real-Time Forecasts",
                  description: "Aggregated results create high-resolution plastic drift forecasts, enabling near-real-time predictions"
                },
                {
                  title: "Community Powered",
                  description: "Uses community resources instead of expensive supercomputers, democratizing environmental modeling"
                },
                {
                  title: "Physics & AI Informed",
                  description: "Lightweight kernels combine ocean-current data, wind patterns, and AI models for accurate predictions"
                }
              ].map((feature, i) => (
                <div key={i} className="bg-background border border-foreground/10 rounded-lg p-6 hover:border-primary/50 transition-colors">
                  <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                  <p className="text-foreground/70 leading-relaxed">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 relative" id="how-it-works">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-sentient mb-6">
                Get Started in <i className="font-light">3 Steps</i>
              </h2>
              <p className="text-lg text-foreground/70 leading-relaxed">
                Join the fight against ocean plastic pollution today
              </p>
            </div>

            <div className="space-y-12">
              {[
                {
                  number: "01",
                  title: "Download the App",
                  description: "Install our lightweight desktop application. It runs securely in the background without affecting your work."
                },
                {
                  number: "02",
                  title: "Run Simulations",
                  description: "Your device receives small ocean-drift trajectory simulations using real ocean-current, wind, and wave data from NOAA and Copernicus."
                },
                {
                  number: "03",
                  title: "Make an Impact",
                  description: "Results are aggregated into global plastic-drift probability maps helping cleanup crews focus efforts where needed most."
                }
              ].map((step, i) => (
                <div key={i} className="flex gap-6">
                  <div className="text-6xl font-bold text-primary/20">{step.number}</div>
                  <div>
                    <h3 className="text-2xl font-bold mb-3">{step.title}</h3>
                    <p className="text-foreground/70 leading-relaxed">{step.description}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-16 text-center" id="download">
              <Button className="text-lg px-8">
                [Download Desktop App]
              </Button>
              <p className="mt-4 text-sm text-foreground/60">Available for Windows, macOS, and Linux • Free and Open Source</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-foreground/10">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-foreground/60">
            Computing for Social Good • Marine Conservation • Volunteer Network
          </p>
          <p className="mt-4 text-xs text-foreground/40">
            MIT Licensed • Illinois Institute of Technology
          </p>
        </div>
      </footer>
    </>
  );
}
