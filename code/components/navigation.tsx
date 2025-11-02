"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Waves } from "lucide-react"
import { cn } from "@/lib/utils"

export function Navigation() {
  const pathname = usePathname()

  return (
    <nav className="absolute top-0 z-50 w-full">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold text-lg text-white drop-shadow-md">
          <Waves className="h-6 w-6 text-primary drop-shadow-md" />
          <span className="text-balance">Ocean Plastic Tracker</span>
        </Link>

        <div className="flex items-center gap-6">
          <Link
            href="/"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary drop-shadow-md",
              pathname === "/" ? "text-white" : "text-white/80",
            )}
          >
            Home
          </Link>
          <Link
            href="/live-map"
            className={cn(
              "text-sm font-medium transition-colors hover:text-primary drop-shadow-md",
              pathname === "/live-map" ? "text-white" : "text-white/80",
            )}
          >
            Live Map
          </Link>
        </div>
      </div>
    </nav>
  )
}
