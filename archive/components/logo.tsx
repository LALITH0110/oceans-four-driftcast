export const Logo = (props: React.SVGProps<SVGSVGElement>) => {
  return (
    <svg
      viewBox="0 0 80 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <text
        x="0"
        y="30"
        fill="white"
        fontSize="32"
        fontWeight="200"
        fontFamily="Sentient, sans-serif"
      >
        vArI
      </text>
    </svg>
  );
};
