"use client"

import { useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  age: number
  maxAge: number
  speed: number
}

export function OceanCurrentMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d", { alpha: true })
    if (!ctx) return

    const resizeCanvas = () => {
      const container = canvas.parentElement
      if (container) {
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight
      }
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    const particles: Particle[] = []
    const particleCount = 3000

    // North Atlantic Gyre flow field (simplified)
    const getFlow = (x: number, y: number) => {
      const centerX = canvas.width * 0.5
      const centerY = canvas.height * 0.5

      // Create circular gyre pattern
      const dx = x - centerX
      const dy = y - centerY
      const distance = Math.sqrt(dx * dx + dy * dy)
      const angle = Math.atan2(dy, dx)

      // Circular flow with some variation
      const flowAngle = angle + Math.PI / 2 + Math.sin(distance * 0.01) * 0.3
      const flowSpeed = Math.min(distance * 0.0008, 2) + Math.sin(x * 0.005) * 0.5

      // Add some turbulence
      const turbulenceX = Math.sin(x * 0.01 + y * 0.005) * 0.3
      const turbulenceY = Math.cos(y * 0.01 + x * 0.005) * 0.3

      return {
        vx: Math.cos(flowAngle) * flowSpeed + turbulenceX,
        vy: Math.sin(flowAngle) * flowSpeed + turbulenceY,
      }
    }

    // Initialize particles
    for (let i = 0; i < particleCount; i++) {
      const x = Math.random() * canvas.width
      const y = Math.random() * canvas.height
      const flow = getFlow(x, y)

      particles.push({
        x,
        y,
        vx: flow.vx,
        vy: flow.vy,
        age: Math.random() * 100,
        maxAge: 50 + Math.random() * 50,
        speed: 0.5 + Math.random() * 1.5,
      })
    }

    let animationId: number

    const animate = () => {
      // Fade out effect for trails
      ctx.fillStyle = "rgba(12, 31, 49, 0.05)"
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      particles.forEach((particle) => {
        // Update particle position based on flow field
        const flow = getFlow(particle.x, particle.y)
        particle.vx = flow.vx * particle.speed
        particle.vy = flow.vy * particle.speed

        particle.x += particle.vx
        particle.y += particle.vy

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width
        if (particle.x > canvas.width) particle.x = 0
        if (particle.y < 0) particle.y = canvas.height
        if (particle.y > canvas.height) particle.y = 0

        particle.age++

        // Calculate opacity based on age
        const lifeRatio = particle.age / particle.maxAge
        const opacity = lifeRatio < 0.3 ? lifeRatio / 0.3 : 1 - (lifeRatio - 0.3) / 0.7

        // Draw particle trail
        if (opacity > 0) {
          const speedFactor = Math.sqrt(particle.vx * particle.vx + particle.vy * particle.vy)
          const blue = 150 + Math.floor(speedFactor * 30)
          const green = 180 + Math.floor(speedFactor * 20)

          ctx.strokeStyle = `rgba(${blue}, ${green}, 255, ${opacity * 0.6})`
          ctx.lineWidth = 1 + speedFactor * 0.3

          ctx.beginPath()
          ctx.moveTo(particle.x, particle.y)
          ctx.lineTo(particle.x - particle.vx * 2, particle.y - particle.vy * 2)
          ctx.stroke()
        }

        // Reset particle if too old
        if (particle.age > particle.maxAge) {
          particle.x = Math.random() * canvas.width
          particle.y = Math.random() * canvas.height
          particle.age = 0
          particle.maxAge = 50 + Math.random() * 50
        }
      })

      animationId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      cancelAnimationFrame(animationId)
    }
  }, [mounted])

  if (!mounted) {
    return (
      <div className="w-full h-full bg-gradient-to-br from-blue-950 to-cyan-950 flex items-center justify-center">
        <div className="text-blue-400 text-lg">Loading map...</div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full bg-gradient-to-br from-[#1e4a66] via-[#1a3d58] to-[#1e4a66]">
      {/* Coastline SVG outlines */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid slice">
        <defs>
          <filter id="coastlineGlow">
            <feGaussianBlur stdDeviation="0.5" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* North America - East Coast */}
        <path
          d="M 0,0 L 0,600 L 50,600 Q 60,595 70,585 L 85,565 Q 95,545 100,525 L 105,505 Q 108,485 110,465 L 112,445 Q 113,425 113,405 L 113,385 Q 112,365 110,345 L 107,325 Q 103,305 98,287 L 92,270 Q 85,253 77,237 L 68,222 Q 58,208 47,195 L 35,183 Q 22,172 8,163 L 0,157 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Florida peninsula */}
        <path
          d="M 50,600 L 55,580 Q 58,565 60,550 L 62,535 Q 63,525 65,515 L 67,500 Q 68,490 68,480 L 67,470 Q 65,460 62,450 L 58,442 Q 52,435 45,432 L 38,430 Q 32,432 28,438 L 25,448 Q 24,458 25,468 L 28,480 Q 32,492 35,504 L 38,520 Q 40,535 42,550 L 45,570 Q 47,585 49,595 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Greenland */}
        <path
          d="M 280,0 L 290,2 Q 305,6 318,14 L 332,25 Q 345,38 355,54 L 363,72 Q 370,90 375,110 L 378,130 Q 380,150 380,170 L 379,190 Q 377,210 372,228 L 366,245 Q 358,261 348,275 L 336,287 Q 322,297 306,304 L 289,308 Q 271,310 254,308 L 237,303 Q 222,296 209,286 L 198,274 Q 189,260 183,244 L 179,227 Q 176,209 176,190 L 177,171 Q 180,153 185,135 L 192,118 Q 200,102 210,88 L 222,75 Q 235,64 250,55 L 265,48 Q 280,43 295,40 L 280,0 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Iceland */}
        <path
          d="M 380,140 Q 392,138 403,140 L 414,145 Q 423,152 428,162 L 432,174 Q 434,186 432,198 L 427,209 Q 419,218 408,222 L 396,224 Q 384,223 373,218 L 363,210 Q 356,199 353,186 L 352,173 Q 354,161 359,151 L 367,143 Q 375,138 385,137 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.2"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* British Isles */}
        <path
          d="M 485,220 Q 492,218 499,220 L 507,225 Q 513,232 516,241 L 518,251 Q 518,261 515,270 L 510,279 Q 503,286 494,289 L 484,290 Q 474,288 466,283 L 459,275 Q 455,266 454,256 L 455,246 Q 458,237 463,229 L 470,223 Q 477,219 485,218 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.2"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Scandinavia */}
        <path
          d="M 520,0 L 530,5 Q 542,12 552,22 L 561,35 Q 568,50 573,67 L 576,85 Q 578,104 577,123 L 574,142 Q 569,160 561,177 L 551,192 Q 539,205 525,215 L 509,223 Q 492,228 475,230 L 458,229 Q 442,225 428,218 L 520,0 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Western Europe (Iberia, France, Low Countries) */}
        <path
          d="M 428,218 L 415,227 Q 403,238 393,251 L 385,266 Q 379,282 375,299 L 373,317 Q 372,335 374,353 L 378,371 Q 384,388 392,404 L 402,419 Q 414,432 428,443 L 443,452 Q 460,459 478,463 L 497,465 Q 516,464 534,459 L 551,451 Q 566,441 579,428 L 590,413 Q 599,397 605,379 L 609,360 Q 611,341 610,322 L 607,303 Q 602,285 594,268 L 584,252 Q 572,238 558,226 L 542,216 Q 525,208 507,203 L 488,200 Q 470,200 452,203 L 435,208 Q 420,214 428,218 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Mediterranean (Iberia continuation) */}
        <path
          d="M 392,404 L 385,420 Q 380,436 377,453 L 375,470 Q 375,487 377,504 L 381,521 Q 387,537 395,552 L 405,565 Q 417,576 431,585 L 1000,585 L 1000,400 L 610,400 Q 608,408 606,416 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* North Africa */}
        <path
          d="M 431,585 L 447,593 Q 464,598 482,600 L 1000,600 L 1000,585 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1.5"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Caribbean Islands */}
        <path
          d="M 120,540 Q 128,538 135,540 L 142,544 Q 147,550 149,558 L 150,566 Q 149,574 145,581 L 139,587 Q 131,590 123,589 L 115,586 Q 109,580 107,572 L 106,564 Q 108,556 113,549 L 118,543 Q 125,539 132,538 Z"
          fill="rgba(30, 50, 70, 0.6)"
          stroke="#87b5d5"
          strokeWidth="1"
          strokeLinejoin="round"
          filter="url(#coastlineGlow)"
        />

        {/* Top edge fill */}
        <rect x="0" y="0" width="1000" height="1" fill="rgba(30, 50, 70, 0.6)" />
        <rect x="1000" y="0" width="1" height="600" fill="rgba(30, 50, 70, 0.6)" />
      </svg>

      {/* Current flow canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

      {/* Map grid overlay */}
      <div className="absolute inset-0 opacity-5 pointer-events-none">
        <div className="w-full h-full bg-[linear-gradient(to_right,rgba(59,130,246,0.3)_1px,transparent_1px),linear-gradient(to_bottom,rgba(59,130,246,0.3)_1px,transparent_1px)] bg-[size:40px_40px]" />
      </div>

      {/* Geographic labels */}
      <div className="absolute inset-0 pointer-events-none text-shadow">
        <div className="absolute top-[20%] left-[15%] text-amber-200/60 text-xs md:text-sm font-semibold tracking-wider">
          NORTH<br/>AMERICA
        </div>
        <div className="absolute top-[15%] left-[35%] text-amber-200/60 text-xs md:text-sm font-semibold tracking-wider">
          GREENLAND
        </div>
        <div className="absolute top-[20%] left-[55%] text-amber-200/60 text-xs md:text-sm font-semibold tracking-wider">
          ICELAND
        </div>
        <div className="absolute top-[30%] right-[20%] text-amber-200/60 text-xs md:text-sm font-semibold tracking-wider">
          EUROPE
        </div>
        <div className="absolute top-[50%] left-1/2 -translate-x-1/2 -translate-y-1/2 text-cyan-300/50 text-base md:text-xl font-light tracking-[0.3em]">
          NORTH ATLANTIC OCEAN
        </div>
        <div className="absolute bottom-[25%] left-[20%] text-cyan-300/40 text-xs md:text-sm font-light tracking-wider">
          Caribbean<br/>Sea
        </div>
        <div className="absolute bottom-[20%] right-[15%] text-amber-200/50 text-xs md:text-sm font-semibold tracking-wider">
          AFRICA
        </div>
      </div>

      {/* Gyre indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="absolute top-[45%] left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
      >
        <div className="relative">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
            className="w-48 h-48 md:w-64 md:h-64 rounded-full border-2 border-cyan-400/10 border-dashed"
          />
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
            className="absolute inset-4 rounded-full border border-blue-400/15 border-dashed"
          />
        </div>
      </motion.div>
    </div>
  )
}
