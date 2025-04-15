"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";

// LogoutModal Component
function LogoutModal({ 
  isOpen, 
  onClose, 
  onConfirm 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  onConfirm: () => void;
}) {
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Semi-transparent backdrop */}
      <div 
        className="fixed inset-0 bg-black/30 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      ></div>
      
      {/* Modal */}
      <div className="relative z-10 bg-white rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden">
        <div className="p-6">
          <div className="text-center mb-6">
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
              </div>
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-1">Confirm Logout</h2>
            <p className="text-gray-600">
              Are you sure you want to log out of your account?
            </p>
          </div>
          
          <div className="flex flex-col space-y-3">
            <button
              onClick={onConfirm}
              className="w-full py-2.5 px-4 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              Yes, Log Out
            </button>
            <button
              onClick={onClose}
              className="w-full py-2.5 px-4 bg-white text-indigo-600 font-medium rounded-md border border-indigo-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// New Stat Card Component
function StatCard({ 
  title, 
  value, 
  icon, 
  color, 
  percentage, 
  increasing = true 
}: { 
  title: string; 
  value: number; 
  icon: React.ReactNode; 
  color: string;
  percentage?: number;
  increasing?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={`bg-white rounded-xl shadow-sm border border-gray-50 p-6 hover:shadow-md transition-shadow relative overflow-hidden`}
    >
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm text-gray-500 mb-1">{title}</p>
          <motion.h3 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.4 }}
            className="text-2xl font-bold text-gray-800"
          >
            {value}
          </motion.h3>
          {percentage !== undefined && (
            <div className="flex items-center mt-2">
              <div className={`flex items-center ${increasing ? 'text-green-600' : 'text-red-600'}`}>
                <svg 
                  className="w-4 h-4 mr-1" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24" 
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    strokeWidth={2} 
                    d={increasing ? "M5 10l7-7m0 0l7 7m-7-7v18" : "M19 14l-7 7m0 0l-7-7m7 7V3"} 
                  />
                </svg>
                <span className="text-xs font-medium">{percentage}%</span>
              </div>
              <span className="text-xs text-gray-500 ml-2">vs last month</span>
            </div>
          )}
        </div>
        <div className={`p-3 bg-${color}-50 rounded-lg`}>
          {icon}
        </div>
      </div>
      
      {/* Background decoration */}
      <div className={`absolute -bottom-6 -right-6 w-24 h-24 rounded-full bg-${color}-50 opacity-30`}></div>
    </motion.div>
  );
}

// Donut Chart Component
function DonutChart({ normalCount, pneumoniaCount }: { normalCount: number; pneumoniaCount: number }) {
  const total = normalCount + pneumoniaCount;
  const normalPercentage = total > 0 ? Math.round((normalCount / total) * 100) : 0;
  const pneumoniaPercentage = total > 0 ? Math.round((pneumoniaCount / total) * 100) : 0;
  
  // Calculate the circumference of the circle
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  
  // Calculate the dash offset for each segment
  const normalDashOffset = circumference * (1 - normalPercentage / 100);
  const pneumoniaDashOffset = circumference; // Start at full circle
  
  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg className="w-full h-full" viewBox="0 0 100 100">
        {/* Background circle */}
        <circle 
          cx="50" 
          cy="50" 
          r={radius} 
          fill="none" 
          stroke="#e5e7eb" 
          strokeWidth="10" 
        />
        
        {/* Normal patients segment */}
        <motion.circle 
          cx="50" 
          cy="50" 
          r={radius} 
          fill="none" 
          stroke="#10b981" 
          strokeWidth="10" 
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: normalDashOffset }}
          transition={{ duration: 1, delay: 0.5 }}
          transform="rotate(-90 50 50)"
        />
        
        {/* Pneumonia patients segment - only showing the part that's not normal */}
        <motion.circle 
          cx="50" 
          cy="50" 
          r={radius} 
          fill="none" 
          stroke="#ef4444" 
          strokeWidth="10" 
          strokeDasharray={circumference} 
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference * (normalPercentage / 100) }}
          transition={{ duration: 1, delay: 0.8 }}
          transform={`rotate(${360 * (1 - normalPercentage / 100) - 90} 50 50)`}
        />
        
        {/* Center text */}
        <text 
          x="50" 
          y="45" 
          textAnchor="middle" 
          dominantBaseline="middle" 
          className="text-lg font-bold text-gray-700"
        >
          {total}
        </text>
        <text 
          x="50" 
          y="60" 
          textAnchor="middle" 
          dominantBaseline="middle" 
          className="text-xs text-gray-500"
        >
          Total scans
        </text>
      </svg>
      
      {/* Legend */}
      <div className="absolute -bottom-16 left-0 right-0">
        <div className="flex justify-center space-x-6 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
            <span className="text-gray-600">Normal ({normalPercentage}%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
            <span className="text-gray-600">Pneumonia ({pneumoniaPercentage}%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Recent Scans Table Component
function RecentScansTable({ scans }: { scans: any[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-gray-200">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
            <th scope="col" className="relative px-6 py-3">
              <span className="sr-only">View</span>
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {scans.map((scan, index) => (
            <motion.tr 
              key={scan.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex items-center">
                  <div className="text-sm font-medium text-gray-900">{scan.patientName}</div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-500">{scan.date}</div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${scan.result === 'Pneumonia' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                  {scan.result}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : scan.confidence}%
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <Link href={`/dashboard/doctor/scans/${scan.id}`} className="text-indigo-600 hover:text-indigo-900">View</Link>
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DoctorDashboardClient({ user }: { user: { id: string; role: string; name: string } }) {
  const router = useRouter();
  const [isLogoutModalOpen, setIsLogoutModalOpen] = useState(false);
  const [dashboardData, setDashboardData] = useState({
    totalScans: 0,
    pneumoniaCases: 0,
    normalCases: 0,
    todayScans: 0,
    recentScans: [] as any[]
  });
  const [isLoading, setIsLoading] = useState(true);
  
  // Fetch dashboard data
  const fetchData = async () => {
    setIsLoading(true);
    try {
      // Fetch data from our API endpoint
      const response = await fetch('/api/stats');
      
      if (!response.ok) {
        throw new Error('Failed to fetch statistics');
      }
      
      const result = await response.json();
      
      if (result.success && result.data) {
        console.log('Received dashboard data:', result.data);
        setDashboardData({
          totalScans: result.data.totalScans || 0,
          pneumoniaCases: result.data.pneumoniaCases || 0,
          normalCases: result.data.normalCases || 0,
          todayScans: result.data.todayScans || 0,
          recentScans: result.data.recentScans.map((scan: any) => ({
            ...scan,
            // Format date for display
            date: new Date(scan.date).toLocaleDateString()
          }))
        });
      } else {
        // Fallback to mock data if API call fails
        console.warn('Using fallback data for dashboard');
        generateMockData();
      }
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
      // Use mock data when API fails
      generateMockData();
    } finally {
      setIsLoading(false);
    }
  };
  
  // Generate mock data for demo/fallback purposes
  const generateMockData = () => {
    // Generate some sample data
    const totalScans = Math.floor(Math.random() * 50) + 50; // 50-100
    const pneumoniaCases = Math.floor(Math.random() * 35) + 15; // 15-50
    const normalCases = totalScans - pneumoniaCases;
    const todayScans = Math.floor(Math.random() * 8) + 1; // 1-8
    
    // Generate recent scans
    const recentScans = Array.from({ length: 5 }, (_, i) => {
      const isPneumonia = Math.random() > 0.5;
      // Use 0.80-0.99 range for confidence to match API format (which is multiplied by 100)
      const rawConfidence = 0.80 + ((Math.floor(Math.random() * 20)) / 100);
      return {
        id: `scan-${i}`,
        patientName: `Patient ${String.fromCharCode(65 + i)}`,
        date: new Date(Date.now() - i * 86400000).toLocaleDateString(),
        result: isPneumonia ? 'Pneumonia' : 'Normal',
        confidence: Math.round(rawConfidence * 100), // Convert to percentage (80-99%)
      };
    });
    
    setDashboardData({
      totalScans,
      pneumoniaCases,
      normalCases,
      todayScans,
      recentScans
    });
  };

  useEffect(() => {
    fetchData();
  }, []);
  
  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
      router.push("/auth/login");
    } catch (err) {
      console.error("Error logging out:", err);
    }
  };

  return (
    <main className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="mr-3 flex items-center justify-center w-8 h-8">
                <Image src="/icons/logo.png" alt="Logo" width={20} height={20} />
              </div>
              <h1 className="font-semibold text-gray-800 text-lg">MedRecord Hub</h1>
            </div>
            <div className="flex items-center space-x-6">
              <span className="text-gray-600 text-sm">Welcome, {user.name}</span>
              <button 
                onClick={() => setIsLogoutModalOpen(true)}
                className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center transition-colors"
              >
                <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-grow max-w-7xl mx-auto w-full px-6 py-10">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">Pneumonia Detection Dashboard</h2>
          <Link 
            href="/dashboard/doctor/upload-xray" 
            className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Upload New X-Ray
          </Link>
        </div>
        
        <AnimatePresence>
          {isLoading ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex justify-center items-center h-64"
            >
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <StatCard 
                  title="Total X-Ray Scans" 
                  value={dashboardData.totalScans} 
                  icon={
                    <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  } 
                  color="indigo"
                  percentage={8}
                  increasing={true}
                />
                
                <StatCard 
                  title="Pneumonia Cases" 
                  value={dashboardData.pneumoniaCases} 
                  icon={
                    <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  } 
                  color="red"
                  percentage={5}
                  increasing={true}
                />
                
                <StatCard 
                  title="Normal Cases" 
                  value={dashboardData.normalCases} 
                  icon={
                    <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  } 
                  color="green"
                  percentage={12}
                  increasing={true}
                />
                
                <StatCard 
                  title="Today's Scans" 
                  value={dashboardData.todayScans} 
                  icon={
                    <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  } 
                  color="yellow"
                />
              </div>
              
              {/* Charts and Tables Section */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Donut Chart */}
                <motion.div 
                  className="lg:col-span-1 bg-white rounded-xl shadow-sm border border-gray-50 p-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                >
                  <h3 className="text-lg font-medium text-gray-800 mb-6">Case Distribution</h3>
                  <DonutChart 
                    normalCount={dashboardData.normalCases} 
                    pneumoniaCount={dashboardData.pneumoniaCases} 
                  />
                </motion.div>
                
                {/* Recent Scans */}
                <motion.div 
                  className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-50 p-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.4 }}
                >
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-lg font-medium text-gray-800">Recent X-Ray Scans</h3>
                    <Link href="/dashboard/doctor/all-scans" className="text-sm text-indigo-600 hover:text-indigo-800">
                      View all
                    </Link>
                  </div>
                  <RecentScansTable scans={dashboardData.recentScans} />
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      {/* Footer */}
      <footer className="bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="md:flex md:items-center md:justify-between">
            <div className="flex justify-center md:justify-start">
              <div className="flex items-center">
                <div className="mr-2 flex items-center justify-center w-7 h-7">
                  <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
                </div>
                <span className="text-gray-500 text-sm">© 2025 MedRecord Hub. All rights reserved.</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
      
      {/* Logout Modal */}
      <LogoutModal 
        isOpen={isLogoutModalOpen} 
        onClose={() => setIsLogoutModalOpen(false)} 
        onConfirm={handleLogout} 
      />
    </main>
  );
} 