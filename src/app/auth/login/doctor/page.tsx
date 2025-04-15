"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";
import { login } from "@/lib/auth";

export default function DoctorLogin() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(searchParams.get("error") || "");
  
  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setIsLoading(true);
    setError("");
    
    const formData = new FormData(e.currentTarget);
    
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({
          doctorId: formData.get("doctorId"),
          password: formData.get("password"),
          role: "DOCTOR"
        }),
        headers: {
          "Content-Type": "application/json"
        }
      });
      
      // Check if response is OK and has the correct content type
      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
          const errorData = await response.json();
          throw new Error(errorData.message || "Login failed");
        } else {
          throw new Error(`Server error: ${response.status}`);
        }
      }
      
      const contentType = response.headers.get("content-type");
      if (!contentType || contentType.indexOf("application/json") === -1) {
        throw new Error("Received non-JSON response from server");
      }
      
      const data = await response.json();
      
      if (data.success) {
        // Store user info in localStorage for faster access in other components
        if (data.user) {
          try {
            localStorage.setItem('userInfo', JSON.stringify({
              id: data.user.id,
              name: data.user.name,
              role: data.user.role
            }));
            console.log('User info cached in localStorage');
          } catch (storageError) {
            console.error('Failed to cache user info:', storageError);
          }
        }
        
        router.push("/dashboard/doctor");
      } else {
        setError(data.message || "Login failed");
      }
    } catch (err: any) {
      console.error("Login error:", err);
      setError(err.message || "An error occurred during login");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="flex flex-col min-h-screen bg-white">
      {/* Navigation Bar */}
      <nav className="flex justify-between items-center px-6 py-3 bg-white">
        <div className="flex items-center">
          <div className="mr-2 flex items-center justify-center w-7 h-7">
            <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
          </div>
          <h1 className="font-semibold text-gray-800">MedRecord Hub</h1>
        </div>
        <div>
          <Link href="/auth/login" className="text-indigo-600 text-sm hover:text-indigo-800">
            Back to Login
          </Link>
        </div>
      </nav>

      {/* Doctor Login Content */}
      <div className="flex flex-1 justify-center items-center px-4 py-12">
        <div className="w-full max-w-md">
          <div className="bg-white shadow-md rounded-xl p-8">
            <div className="text-center mb-6">
              <div className="flex justify-center mb-4">
                <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center">
                  <Image src="/images/doctor.png" alt="Doctor" width={40} height={40} />
                </div>
              </div>
              <h2 className="text-2xl font-bold text-gray-800 mb-1">Doctor Login</h2>
              <p className="text-gray-600 text-sm">Access your patient records and schedules</p>
            </div>

            {/* Login Form */}
            <form onSubmit={handleSubmit} className="mt-8 space-y-5">
              {error && (
                <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">
                  {error}
                </div>
              )}
              
              <div>
                <label htmlFor="doctorId" className="block text-sm font-medium text-gray-700 mb-1">
                  Doctor ID
                </label>
                <input 
                  type="text" 
                  id="doctorId" 
                  name="doctorId" 
                  placeholder="Enter your doctor ID" 
                  defaultValue="DOC12345"
                  className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                  Password
                </label>
                <input 
                  type="password" 
                  id="password" 
                  name="password" 
                  placeholder="Enter your password" 
                  className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  required
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
                    Remember me
                  </label>
                </div>
                
                <div className="text-sm">
                  <a href="#" className="text-indigo-600 hover:text-indigo-800">
                    Forgot password?
                  </a>
                </div>
              </div>
              
              <div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  {isLoading ? "Signing in..." : "Sign in to Doctor Portal"}
                </button>
              </div>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                For new doctors, please contact hospital administration for credentials
              </p>
              {/* Sample login info for demo purposes */}
              <div className="mt-4 p-3 bg-gray-50 rounded-md text-xs text-gray-500 text-left">
                <p className="font-medium mb-1">Demo Credentials:</p>
                <p>Doctor ID: DOC12345</p>
                <p>Password: doctor123</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Simple Footer */}
      <footer className="bg-white py-4 border-t text-center text-sm text-gray-600">
        <div className="container mx-auto">
          <p>© 2023 MedRecord Hub • <a href="#" className="text-indigo-600">Privacy</a> • <a href="#" className="text-indigo-600">Terms</a></p>
        </div>
      </footer>
    </main>
  );
}
