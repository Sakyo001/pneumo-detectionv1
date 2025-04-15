"use client";

import React from "react";
import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, Legend
} from "recharts";
import UserManagement from "./components/UserManagement";
import PneumoniaManagement from "./components/PneumoniaManagement";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"];

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
      <div 
        className="fixed inset-0 bg-black/30 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      ></div>
      
      <div className="relative z-10 bg-white rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden">
        <div className="p-6">
          <div className="text-center mb-6">
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

export default function AdminDashboard({ user }: { 
  user: { 
    id: string; 
    email: string;
    name: string; 
    role: string;
  } 
}) {
  const router = useRouter();
  const [isLogoutModalOpen, setIsLogoutModalOpen] = useState(false);
  const [stats, setStats] = useState({
    totalScans: 0,
    pneumoniaCases: 0,
    activeDoctors: 0,
    avgConfidence: 0
  });
  const [scanData, setScanData] = useState([]);
  const [pneumoniaData, setPneumoniaData] = useState([]);
  const [doctorData, setDoctorData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState('dashboard');

  const formatModelAccuracy = (value: number | null) => {
    if (value === null || isNaN(value)) return '0%';
    // Ensure the value is capped at 100%
    const cappedValue = Math.min(value, 100);
    return `${cappedValue.toFixed(1)}%`;
  };

  const refreshData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/admin/stats');
      if (!response.ok) {
        throw new Error('Failed to fetch dashboard data');
      }
      const data = await response.json();
      
      console.log('Admin dashboard received data:', data);
      
      // Validate and format the data
      if (data && typeof data === 'object') {
        const formattedData = {
          stats: {
            totalScans: Number(data.stats?.totalScans) || 0,
            pneumoniaCases: Number(data.stats?.pneumoniaCases) || 0,
            activeDoctors: Number(data.stats?.activeDoctors) || 0,
            avgConfidence: Math.min(Number(data.stats?.avgConfidence) || 0, 100)
          },
          scanData: Array.isArray(data.scanData) ? data.scanData : [],
          pneumoniaData: Array.isArray(data.pneumoniaData) ? data.pneumoniaData : [],
          doctorData: Array.isArray(data.doctorData) ? data.doctorData : []
        };
        
        console.log('Formatted admin dashboard data:', formattedData);
        
        setStats(formattedData.stats);
        setScanData(formattedData.scanData);
        setPneumoniaData(formattedData.pneumoniaData);
        setDoctorData(formattedData.doctorData);
      }
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
      router.push("/auth/login");
    } catch (err) {
      console.error("Error logging out:", err);
    }
  };

  const renderSection = () => {
    switch (activeSection) {
      case 'users':
        return <UserManagement />;
      case 'pneumonia':
        return <PneumoniaManagement />;
      default:
        return (
          <>
            {/* Stats Overview */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Total Scans</p>
                    <h3 className="text-2xl font-bold text-gray-800 mt-1">{stats.totalScans}</h3>
                  </div>
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Pneumonia Cases</p>
                    <h3 className="text-2xl font-bold text-gray-800 mt-1">{stats.pneumoniaCases}</h3>
                  </div>
                  <div className="p-3 bg-red-50 rounded-lg">
                    <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Active Doctors</p>
                    <h3 className="text-2xl font-bold text-gray-800 mt-1">{stats.activeDoctors}</h3>
                  </div>
                  <div className="p-3 bg-green-50 rounded-lg">
                    <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Model Accuracy</p>
                    <h3 className="text-2xl font-bold text-gray-800 mt-1">
                      {formatModelAccuracy(stats.avgConfidence)}
                    </h3>
                  </div>
                  <div className="p-3 bg-purple-50 rounded-lg">
                    <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Scans Over Time Chart */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6"
              >
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Scans Over Time</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={scanData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="scans" stroke="#3B82F6" name="Total Scans" />
                      <Line type="monotone" dataKey="pneumonia" stroke="#EF4444" name="Pneumonia Cases" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Pneumonia Distribution Chart */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6"
              >
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Pneumonia Distribution</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pneumoniaData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {pneumoniaData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>

              {/* Doctor Activity Chart */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="bg-white rounded-xl shadow-sm border border-gray-50 p-6 lg:col-span-2"
              >
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Doctor Activity</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={doctorData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis yAxisId="left" orientation="left" stroke="#3B82F6" />
                      <YAxis yAxisId="right" orientation="right" stroke="#10B981" />
                      <Tooltip />
                      <Legend />
                      <Bar yAxisId="left" dataKey="scans" fill="#3B82F6" name="Total Scans" />
                      <Bar yAxisId="right" dataKey="accuracy" fill="#10B981" name="Accuracy %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            </div>
          </>
        );
    }
  };

  return (
    <React.Fragment>
      <main className="flex flex-col min-h-screen bg-gray-50">
        {/* Header */}
        <motion.header 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="bg-white border-b border-gray-100 sticky top-0 z-50"
        >
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex justify-between items-center">
              <motion.div 
                className="flex items-center"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="mr-3 flex items-center justify-center w-8 h-8">
                  <Image src="/icons/logo.png" alt="Logo" width={20} height={20} />
                </div>
                <h1 className="font-semibold text-gray-800 text-lg">MedRecord Hub</h1>
              </motion.div>
              <div className="flex items-center space-x-6">
                <span className="text-gray-600 text-sm">Welcome, {user.name}</span>
                <motion.button 
                  onClick={() => setIsLogoutModalOpen(true)}
                  className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  Sign Out
                </motion.button>
              </div>
            </div>
          </div>
        </motion.header>

        {/* Navigation Tabs */}
        <div className="bg-white border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-6">
            <nav className="flex space-x-8">
              <button
                onClick={() => setActiveSection('dashboard')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeSection === 'dashboard'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveSection('users')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeSection === 'users'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                User Management
              </button>
              <button
                onClick={() => setActiveSection('pneumonia')}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeSection === 'pneumonia'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Pneumonia Data
              </button>
            </nav>
          </div>
        </div>

        {/* Main content */}
        <div className="flex-grow max-w-7xl mx-auto w-full px-6 py-10">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl font-bold text-gray-800 mb-6"
          >
            {activeSection === 'dashboard' && 'Admin Dashboard'}
            {activeSection === 'users' && 'User Management'}
            {activeSection === 'pneumonia' && 'Pneumonia Data Management'}
          </motion.h2>

          {isLoading && activeSection === 'dashboard' ? (
            <div className="flex justify-center py-20">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
            </div>
          ) : (
            renderSection()
          )}
        </div>
      </main>

      {isLogoutModalOpen && (
        <LogoutModal
          isOpen={isLogoutModalOpen}
          onClose={() => setIsLogoutModalOpen(false)}
          onConfirm={handleLogout}
        />
      )}
    </React.Fragment>
  );
} 