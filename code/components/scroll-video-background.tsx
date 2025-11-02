import Image from "next/image"

export function ScrollVideoBackground() {
  return (
    <Image
      src="/water.webp"
      alt="Ocean background"
      fill
      className="object-cover"
      priority
      quality={100}
    />
  )
}
