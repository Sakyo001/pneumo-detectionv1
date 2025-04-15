import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow any host in dev environment
  devIndicators: {
    autoPrerender: false,
  },
};

export default nextConfig;
