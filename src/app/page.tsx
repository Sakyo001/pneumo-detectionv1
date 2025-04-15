"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState, useRef } from "react";

export default function Home() {
  const [scrollY, setScrollY] = useState(0);
  const featuresRef = useRef<HTMLElement | null>(null);
  const howItWorksRef = useRef<HTMLElement | null>(null);
  const [visibleItems, setVisibleItems] = useState<Record<string, boolean>>({});
  const [isMounted, setIsMounted] = useState(false);
  
  useEffect(() => {
    setIsMounted(true);
    
    const handleScroll = () => {
      setScrollY(window.scrollY);
      
      // Check all elements with data-animate attribute
      const animatedElements = document.querySelectorAll('[data-animate]');
      const updatedVisibility: Record<string, boolean> = {...visibleItems};
      
      animatedElements.forEach((el) => {
        const id = el.getAttribute('data-animate');
        if (!id) return;
        
        const rect = el.getBoundingClientRect();
        const isVisible = rect.top < window.innerHeight * 0.85 && rect.bottom > 0;
        
        // Only update if there's a change to prevent unnecessary re-renders
        if (updatedVisibility[id] !== isVisible) {
          updatedVisibility[id] = isVisible;
        }
      });
      
      setVisibleItems(updatedVisibility);
    };
    
    // Add event listener
    window.addEventListener("scroll", handleScroll, { passive: true });
    
    // Call handler right away to update initial position
    setTimeout(handleScroll, 100); // Small timeout to ensure DOM is loaded
    
    // Remove event listener on cleanup
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <main className="flex flex-col min-h-screen bg-gray-50 scroll-smooth">
      {/* Add smooth scrolling to whole document */}
      <style jsx global>{`
        html {
          scroll-behavior: smooth;
        }
        
        @keyframes slideLeft {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(calc(-50% - 0.5rem));
          }
        }
        
        .infinite-slider {
          animation: slideLeft 30s linear infinite;
          width: fit-content;
        }
        
        .infinite-slider:hover {
          animation-play-state: paused;
        }
        
        @keyframes slideInFromLeft {
          0% {
            transform: translateX(-100px);
            opacity: 0;
          }
          100% {
            transform: translateX(0);
            opacity: 1;
          }
        }
        
        @keyframes slideInFromRight {
          0% {
            transform: translateX(100px);
            opacity: 0;
          }
          100% {
            transform: translateX(0);
            opacity: 1;
          }
        }
        
        .slide-in-left {
          transform: translateX(-60px);
          opacity: 0;
          transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.6s ease-out;
        }
        
        .slide-in-left.visible {
          transform: translateX(0);
          opacity: 1;
        }
        
        .slide-in-right {
          transform: translateX(60px);
          opacity: 0;
          transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.6s ease-out;
        }
        
        .slide-in-right.visible {
          transform: translateX(0);
          opacity: 1;
        }
        
        .fade-in-up {
          transform: translateY(30px);
          opacity: 0;
          transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.6s ease-out;
        }
        
        .fade-in-up.visible {
          transform: translateY(0);
          opacity: 1;
        }
      `}</style>

      {/* Navigation Bar - Upgraded with shadow and transition */}
      <nav className="flex justify-between items-center px-6 py-3 bg-white sticky top-0 z-50 shadow-sm transition-all duration-300">
        <div className="flex items-center">
          <div className="mr-2 flex items-center justify-center w-7 h-7">
            <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
          </div>
          <h1 className="font-semibold text-gray-800">MedRecord Hub</h1>
        </div>
        <div className="flex space-x-3">
          <button 
            className="border border-gray-300 text-indigo-600 px-4 py-1.5 rounded-md text-sm flex items-center hover:bg-indigo-50 hover:border-indigo-300 transition-colors active:bg-indigo-100"
            type="button"
            suppressHydrationWarning
          >
            <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Sign Up
          </button>
          <Link href="/auth/login">
            <button 
              className="bg-indigo-600 text-white px-5 py-1.5 rounded-md text-sm flex items-center hover:bg-indigo-700 transition-colors active:bg-indigo-800 shadow-sm hover:shadow"
              type="button"
              suppressHydrationWarning
            >
              <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
              </svg>
              Sign In
            </button>
          </Link>
        </div>
      </nav>

      {/* Hero Section - Enhanced with gradient overlay */}
      <section className="relative h-[600px] w-full overflow-hidden">
        {/* Dashboard UI background image */}
        <div className="absolute inset-0 w-full h-full">
          <Image 
            src="/images/bg.png" 
            alt="Medical Dashboard Interface" 
            fill 
            priority
            style={{ objectFit: 'cover', objectPosition: 'center' }}
          />
          {/* Add gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-900/70 to-purple-800/50"></div>
        </div>
        
        {/* Text content - with animation */}
        <div className="absolute inset-0 flex flex-start items-center">
          <div className="container mx-auto px-8">
            <div className="max-w-full text-center animate-[fadeIn_1s_ease-in]">
              <h1 className="text-6xl font-bold text-white leading-tight mb-3 drop-shadow-md">
                Revolutionizing Your Medical Records
              </h1>
              <p className="text-white text-lg mb-8 max-w-3xl mx-auto">
                Streamlined, secure, and user-friendly medical record management with advanced AI analysis
              </p>
              <div className="flex justify-center gap-4">
                <Link href="/dashboard/patient">
                  <button 
                    className="bg-white text-indigo-700 px-6 py-2.5 rounded-md text-sm font-medium hover:bg-gray-100 transition-colors active:bg-gray-200 shadow-md hover:shadow-lg"
                    suppressHydrationWarning
                  >
                    Get Started
                  </button>
                </Link>
                <a href="#features" suppressHydrationWarning>
                  <button 
                    className="bg-transparent text-white border border-white px-6 py-2.5 rounded-md text-sm font-medium hover:bg-white/10 transition-colors"
                    suppressHydrationWarning
                  >
                    Learn More
                  </button>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Sliding Attributes Section - IMPROVED */}
      <section className="py-12 bg-white border-y border-gray-100 overflow-hidden relative">
        <div className="absolute inset-y-0 left-0 w-20 z-10 bg-gradient-to-r from-white via-white to-transparent"></div>
        <div className="absolute inset-y-0 right-0 w-20 z-10 bg-gradient-to-l from-white via-white to-transparent"></div>
        
        <div className="flex overflow-hidden">
          <div className="flex infinite-slider">
            <div className="flex items-center space-x-16 pl-8 pr-16">
              <div className="flex items-center">
                <div className="h-12 w-12 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 16.17l-3.59-3.59L4 14l5 5 10-10-1.41-1.42L9 16.17z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Efficient</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-purple-100 rounded-full flex items-center justify-center text-purple-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M13 2.05v3.03c3.39.49 6 3.39 6 6.92 0 .9-.18 1.75-.48 2.54l2.6 1.53c.56-1.24.88-2.62.88-4.07 0-5.18-3.95-9.45-9-9.95zM12 19c-3.87 0-7-3.13-7-7 0-3.53 2.61-6.43 6-6.92V2.05c-5.06.5-9 4.76-9 9.95 0 5.52 4.47 10 9.99 10 3.31 0 6.24-1.61 8.06-4.09l-2.6-1.53C16.17 17.98 14.21 19 12 19z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Fast</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM9 6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9V6zm9 14H6V10h12v10zm-6-3c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Secure</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-green-100 rounded-full flex items-center justify-center text-green-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 11.75c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm6 0c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8 0-.29.02-.58.05-.86 2.36-1.05 4.23-2.98 5.21-5.37C11.07 8.33 14.05 10 17.42 10c.78 0 1.53-.09 2.25-.26.21.71.33 1.47.33 2.26 0 4.41-3.59 8-8 8z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">User-Friendly</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-red-100 rounded-full flex items-center justify-center text-red-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Cloud-Based</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19.3 16.9c.4-.7.7-1.5.7-2.4 0-2.5-2-4.5-4.5-4.5S11 12 11 14.5s2 4.5 4.5 4.5c.9 0 1.7-.3 2.4-.7l3.2 3.2 1.4-1.4-3.2-3.2zm-3.8.1c-1.4 0-2.5-1.1-2.5-2.5s1.1-2.5 2.5-2.5 2.5 1.1 2.5 2.5-1.1 2.5-2.5 2.5zM12 20v2C6.48 22 2 17.52 2 12S6.48 2 12 2c4.84 0 8.87 3.44 9.8 8h-2.07c-.64-2.46-2.4-4.47-4.73-5.41V5c0 1.1-.9 2-2 2h-2v2c0 .55-.45 1-1 1H8v2h2v3H9l-4.79-4.79C4.08 10.79 4 11.38 4 12c0 4.41 3.59 8 8 8z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">AI-Powered</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-teal-100 rounded-full flex items-center justify-center text-teal-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm-2 14l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Reliable</span>
              </div>
            </div>
            
            {/* Duplicate set for seamless loop */}
            <div className="flex items-center space-x-16 pl-8 pr-16">
              <div className="flex items-center">
                <div className="h-12 w-12 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 16.17l-3.59-3.59L4 14l5 5 10-10-1.41-1.42L9 16.17z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Efficient</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-purple-100 rounded-full flex items-center justify-center text-purple-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M13 2.05v3.03c3.39.49 6 3.39 6 6.92 0 .9-.18 1.75-.48 2.54l2.6 1.53c.56-1.24.88-2.62.88-4.07 0-5.18-3.95-9.45-9-9.95zM12 19c-3.87 0-7-3.13-7-7 0-3.53 2.61-6.43 6-6.92V2.05c-5.06.5-9 4.76-9 9.95 0 5.52 4.47 10 9.99 10 3.31 0 6.24-1.61 8.06-4.09l-2.6-1.53C16.17 17.98 14.21 19 12 19z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Fast</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM9 6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9V6zm9 14H6V10h12v10zm-6-3c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Secure</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-green-100 rounded-full flex items-center justify-center text-green-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 11.75c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm6 0c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8 0-.29.02-.58.05-.86 2.36-1.05 4.23-2.98 5.21-5.37C11.07 8.33 14.05 10 17.42 10c.78 0 1.53-.09 2.25-.26.21.71.33 1.47.33 2.26 0 4.41-3.59 8-8 8z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">User-Friendly</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-red-100 rounded-full flex items-center justify-center text-red-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Cloud-Based</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19.3 16.9c.4-.7.7-1.5.7-2.4 0-2.5-2-4.5-4.5-4.5S11 12 11 14.5s2 4.5 4.5 4.5c.9 0 1.7-.3 2.4-.7l3.2 3.2 1.4-1.4-3.2-3.2zm-3.8.1c-1.4 0-2.5-1.1-2.5-2.5s1.1-2.5 2.5-2.5 2.5 1.1 2.5 2.5-1.1 2.5-2.5 2.5zM12 20v2C6.48 22 2 17.52 2 12S6.48 2 12 2c4.84 0 8.87 3.44 9.8 8h-2.07c-.64-2.46-2.4-4.47-4.73-5.41V5c0 1.1-.9 2-2 2h-2v2c0 .55-.45 1-1 1H8v2h2v3H9l-4.79-4.79C4.08 10.79 4 11.38 4 12c0 4.41 3.59 8 8 8z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">AI-Powered</span>
              </div>
              
              <div className="flex items-center">
                <div className="h-12 w-12 bg-teal-100 rounded-full flex items-center justify-center text-teal-600">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm-2 14l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z" />
                  </svg>
                </div>
                <span className="ml-4 text-gray-800 font-medium whitespace-nowrap">Reliable</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section with modern cards and individual animations */}
      <section 
        id="features" 
        ref={featuresRef}
        className="py-20 px-4 bg-gradient-to-b from-white to-indigo-50 overflow-hidden"
      >
        <div className="max-w-6xl mx-auto">
          <div 
            data-animate="features-title"
            className={`text-center mb-16 ${isMounted ? `slide-in-left ${visibleItems['features-title'] ? 'visible' : ''}` : ''}`}
          >
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Powerful Healthcare Features</h2>
            <p className="text-gray-600 max-w-2xl mx-auto">Our platform combines cutting-edge technology with user-friendly design to revolutionize healthcare management</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div 
              data-animate="feature-1"
              className={`bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-all duration-300 border border-gray-100 group ${isMounted ? `fade-in-up ${visibleItems['feature-1'] ? 'visible' : ''}` : ''}`}
              style={isMounted ? { transitionDelay: '0.1s' } : {}}
            >
              <div className="h-12 w-12 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg mb-5 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">Secure Records</h3>
              <p className="text-gray-600">Advanced encryption and security protocols ensure your medical data remains private and protected at all times.</p>
            </div>
            
            {/* Feature 2 */}
            <div 
              data-animate="feature-2"
              className={`bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-all duration-300 border border-gray-100 group ${isMounted ? `fade-in-up ${visibleItems['feature-2'] ? 'visible' : ''}` : ''}`}
              style={isMounted ? { transitionDelay: '0.2s' } : {}}
            >
              <div className="h-12 w-12 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg mb-5 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">AI Analysis</h3>
              <p className="text-gray-600">Powered by EfficientNet model, our platform can detect pneumonia from X-ray images with up to 96% accuracy.</p>
            </div>
            
            {/* Feature 3 */}
            <div 
              data-animate="feature-3"
              className={`bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-all duration-300 border border-gray-100 group ${isMounted ? `fade-in-up ${visibleItems['feature-3'] ? 'visible' : ''}` : ''}`}
              style={isMounted ? { transitionDelay: '0.3s' } : {}}
            >
              <div className="h-12 w-12 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg mb-5 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-3">Instant Results</h3>
              <p className="text-gray-600">Get quick and accurate pneumonia detection results with comprehensive analysis and recommended actions.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section - Enhanced with individual animations from right */}
      <section 
        ref={howItWorksRef}
        className="py-20 px-4 bg-white overflow-hidden"
      >
        <div 
          data-animate="how-it-works-title" 
          className={`${isMounted ? `slide-in-right ${visibleItems['how-it-works-title'] ? 'visible' : ''}` : ''}`}
        >
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">How It Works</h2>
          <p className="text-center text-gray-600 mb-12 max-w-2xl mx-auto">Our streamlined process makes healthcare management easier for everyone involved</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div 
            data-animate="how-1"
            className={`text-center relative group ${isMounted ? `slide-in-right ${visibleItems['how-1'] ? 'visible' : ''}` : ''}`}
            style={isMounted ? { transitionDelay: '0.1s' } : {}}
          >
            <div className="absolute -top-4 -left-4 w-8 h-8 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white flex items-center justify-center font-bold">1</div>
            <div className="flex justify-center mb-5 transform transition-transform group-hover:scale-105 duration-300">
              <div className="w-20 h-20 bg-gradient-to-b from-gray-50 to-gray-100 rounded-xl border border-gray-200 flex items-center justify-center shadow-sm">
                <Image src="/images/patient.png" alt="Patient" width={48} height={48} />
              </div>
            </div>
            <h3 className="font-medium text-indigo-600 mb-2">Patient Portal</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              Patients can easily register, schedule appointments, and view their medical records through our intuitive interface.
            </p>
          </div>
          
          <div 
            data-animate="how-2"
            className={`text-center relative group ${isMounted ? `slide-in-right ${visibleItems['how-2'] ? 'visible' : ''}` : ''}`}
            style={isMounted ? { transitionDelay: '0.2s' } : {}}
          >
            <div className="absolute -top-4 -left-4 w-8 h-8 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white flex items-center justify-center font-bold">2</div>
            <div className="flex justify-center mb-5 transform transition-transform group-hover:scale-105 duration-300">
              <div className="w-20 h-20 bg-gradient-to-b from-gray-50 to-gray-100 rounded-xl border border-gray-200 flex items-center justify-center shadow-sm">
                <Image src="/images/doctor.png" alt="Doctor" width={48} height={48} />
              </div>
            </div>
            <h3 className="font-medium text-indigo-600 mb-2">Doctor Analysis</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              Doctors use our AI-powered tools to analyze X-rays and provide accurate diagnoses, improving patient outcomes.
            </p>
          </div>
          
          <div 
            data-animate="how-3"
            className={`text-center relative group ${isMounted ? `slide-in-right ${visibleItems['how-3'] ? 'visible' : ''}` : ''}`}
            style={isMounted ? { transitionDelay: '0.3s' } : {}}
          >
            <div className="absolute -top-4 -left-4 w-8 h-8 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white flex items-center justify-center font-bold">3</div>
            <div className="flex justify-center mb-5 transform transition-transform group-hover:scale-105 duration-300">
              <div className="w-20 h-20 bg-gradient-to-b from-gray-50 to-gray-100 rounded-xl border border-gray-200 flex items-center justify-center shadow-sm">
                <Image src="/images/admin.png" alt="Admin" width={48} height={48} />
              </div>
            </div>
            <h3 className="font-medium text-indigo-600 mb-2">Admin Dashboard</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              Administrators manage the entire platform, ensuring data security and system efficiency for all users.
            </p>
          </div>
        </div>
      </section>

      {/* Technology Partners Section - New gradient section highlighting the tech */}
      <section className="px-4 py-16 bg-gradient-to-r from-indigo-50 to-purple-50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">Powered By Cutting-Edge Technology</h2>
          <p className="text-center text-gray-600 mb-12 max-w-2xl mx-auto">Our platform leverages advanced technologies to deliver accurate and reliable healthcare solutions</p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 flex flex-col items-center justify-center">
              <div className="h-16 w-16 mb-4 flex items-center justify-center">
                <svg className="w-12 h-12 text-indigo-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/>
                </svg>
              </div>
              <h3 className="font-medium text-gray-800 mb-1 text-center">Machine Learning</h3>
              <p className="text-xs text-gray-500 text-center">Advanced algorithms for accurate diagnosis</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 flex flex-col items-center justify-center">
              <div className="h-16 w-16 mb-4 flex items-center justify-center">
                <svg className="w-12 h-12 text-indigo-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
                </svg>
              </div>
              <h3 className="font-medium text-gray-800 mb-1 text-center">Secure Cloud</h3>
              <p className="text-xs text-gray-500 text-center">End-to-end encrypted storage solutions</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 flex flex-col items-center justify-center">
              <div className="h-16 w-16 mb-4 flex items-center justify-center">
                <svg className="w-12 h-12 text-indigo-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/>
                </svg>
              </div>
              <h3 className="font-medium text-gray-800 mb-1 text-center">EfficientNet</h3>
              <p className="text-xs text-gray-500 text-center">State-of-the-art image classification</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 flex flex-col items-center justify-center">
              <div className="h-16 w-16 mb-4 flex items-center justify-center">
                <svg className="w-12 h-12 text-indigo-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                </svg>
              </div>
              <h3 className="font-medium text-gray-800 mb-1 text-center">Global Access</h3>
              <p className="text-xs text-gray-500 text-center">Available anywhere, anytime</p>
            </div>
          </div>
        </div>
      </section>

      {/* Transform Healthcare Today Section - Enhanced with gradient and animation */}
      <section className="px-4 py-20 mb-16">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white overflow-hidden rounded-2xl shadow-md transform transition-all duration-500 hover:shadow-xl">
            <div className="flex flex-col md:flex-row">
              {/* Left Content */}
              <div className="md:w-1/2 bg-gradient-to-br from-indigo-600 to-purple-700 p-12 text-white flex flex-col justify-center">
                <h2 className="text-4xl font-bold leading-tight mb-6">Transform<br/>Healthcare Today</h2>
                <p className="text-white/90 mb-8 text-base leading-relaxed max-w-md">
                  Join MedRecord Hub to simplify health record management and access cutting-edge AI diagnosis tools. Sign up now to experience seamless healthcare solutions.
                </p>
                <div>
                  <Link href="/dashboard/doctor/upload-xray">
                    <button 
                      className="bg-white text-indigo-600 px-6 py-3 rounded-md text-sm font-medium hover:bg-gray-100 transition-colors shadow-md hover:shadow-lg"
                      suppressHydrationWarning
                    >
                      Try X-Ray Analysis
                    </button>
                  </Link>
                </div>
              </div>
              
              {/* Right Image */}
              <div className="md:w-1/2 relative overflow-hidden">
                <Image
                  src="/images/bg2.jpg" 
                  alt="Healthcare Professionals"
                  width={600}
                  height={350}
                  className="h-full w-full object-cover transition-transform duration-700 hover:scale-105"
                  priority
                />
                <div className="absolute inset-0 bg-gradient-to-l from-indigo-900/30 to-transparent"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer with gradient top border */}
      <footer className="bg-white p-8 mt-auto border-t border-t-4 border-t-indigo-500 border-gradient-to-r from-indigo-500 to-purple-500">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-center items-center mb-8">
            <div className="text-center">
              <div className="flex items-center justify-center mb-3">
                <div className="mr-2 flex items-center justify-center w-7 h-7">
                  <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
                </div>
                <h2 className="font-medium text-gray-800">MedRecord Hub</h2>
              </div>
              <div className="mb-6">
                <p className="text-sm text-gray-600 mb-2">Subscribe to our newsletter</p>
                <div className="flex mt-2 justify-center">
                  <input 
                    type="email" 
                    placeholder="Your email" 
                    className="border border-gray-300 p-2 rounded-l-md w-48 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    suppressHydrationWarning
                  />
                  <button 
                    className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-4 py-2 rounded-r-md text-sm hover:opacity-90 transition-opacity"
                    suppressHydrationWarning
                  >
                    Subscribe
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-sm">
            <div>
              <h3 className="font-medium mb-3 text-gray-800">Product</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Features</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Pricing</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Demo</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium mb-3 text-gray-800">Resources</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Documentation</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>API</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Guides</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium mb-3 text-gray-800">Company</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>About Us</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Careers</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Contact</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium mb-3 text-gray-800">Plans & Pricing</h3>
              <ul className="space-y-2">
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Free Trial</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Pro</a></li>
                <li><a href="#" className="text-gray-600 hover:text-indigo-600 transition-colors" suppressHydrationWarning>Enterprise</a></li>
              </ul>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t border-gray-200 text-center text-sm text-gray-500">
            <p>© 2023 MedRecord Hub. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </main>
  );
}
