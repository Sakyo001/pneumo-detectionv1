"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function LogoutPage() {
  const router = useRouter();
  const [redirecting, setRedirecting] = useState(false);
  const [countdown, setCountdown] = useState(3);
  
  useEffect(() => {
    if (redirecting) {
      const timer = setTimeout(() => {
        if (countdown > 1) {
          setCountdown(countdown - 1);
        } else {
          router.push("/auth/login");
        }
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [redirecting, countdown, router]);
  
  function handleLogout() {
    setRedirecting(true);
    // API call to destroy session is handled by the server logout action
    fetch("/api/auth/logout", { method: "POST" })
      .catch(err => console.error("Error calling logout API:", err));
  }
  
  function handleCancel() {
    router.back();
  }
  
  return (
    <main className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center">
            <div className="mr-3 flex items-center justify-center w-8 h-8">
              <Image src="/icons/logo.png" alt="Logo" width={20} height={20} />
            </div>
            <h1 className="font-semibold text-gray-800 text-lg">MedRecord Hub</h1>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-grow items-center justify-center p-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-50 p-8 max-w-md w-full">
          <div className="text-center mb-6">
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
              </div>
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-1">Confirm Logout</h2>
            {redirecting ? (
              <p className="text-gray-600">
                Logged out successfully. Redirecting to login in {countdown} seconds...
              </p>
            ) : (
              <p className="text-gray-600">
                Are you sure you want to log out of your account?
              </p>
            )}
          </div>
          
          {!redirecting ? (
            <div className="flex flex-col space-y-3">
              <button
                onClick={handleLogout}
                className="w-full py-2.5 px-4 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                Yes, Log Out
              </button>
              <button
                onClick={handleCancel}
                className="w-full py-2.5 px-4 bg-white text-indigo-600 font-medium rounded-md border border-indigo-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex justify-center">
              <Link 
                href="/auth/login"
                className="py-2.5 px-4 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
              >
                Return to Login
              </Link>
            </div>
          )}
        </div>
      </div>
      
      {/* Footer */}
      <footer className="bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex justify-center">
            <div className="flex items-center">
              <div className="mr-2 flex items-center justify-center w-7 h-7">
                <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
              </div>
              <span className="text-gray-500 text-sm">© 2023 MedRecord Hub. All rights reserved.</span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
} 