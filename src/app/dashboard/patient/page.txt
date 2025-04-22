"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

interface AnalysisResult {
  id: string;
  reference_number: string;
  image_url: string;
  analysis_result: string;
  confidence_score: number;
  created_at: string;
  doctor_name: string;
  pneumonia_type?: string | null;
  severity?: string | null;
  recommended_action?: string | null;
  patient_name?: string;
}

// Error boundary component for catching runtime errors
function ErrorFallback({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) {
  return (
    <div className="p-6 bg-red-50 border border-red-100 rounded-lg">
      <h3 className="text-lg font-medium text-red-800 mb-2">Something went wrong</h3>
      <p className="text-sm text-red-600 mb-4">{error.message}</p>
      <button
        onClick={resetErrorBoundary}
        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
      >
        Try again
      </button>
    </div>
  );
}

export default function PatientDashboard() {
  const [referenceNumber, setReferenceNumber] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [hasError, setHasError] = useState(false);

  // Reset error state when component mounts
  useEffect(() => {
    setHasError(false);
  }, []);

  const handleTrackScan = async () => {
    if (!referenceNumber.trim()) {
      setError("Please enter a reference number");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);
    setHasError(false);

    try {
      const response = await fetch(`/api/analysis/${referenceNumber}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to fetch analysis results");
      }

      const data = await response.json();
      if (!data) {
        throw new Error("No data received from server");
      }
      
      // Validate important fields
      if (!data.reference_number || !data.analysis_result) {
        throw new Error("Incomplete data received from server");
      }
      
      setResult(data);
    } catch (err) {
      console.error('Error details:', err);
      setError(err instanceof Error ? err.message : "An error occurred");
      setHasError(true);
    } finally {
      setIsLoading(false);
    }
  };

  // If a fatal error occurred, show a simple error message
  if (hasError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
        <div className="max-w-md w-full bg-white rounded-xl shadow-sm border border-gray-50 p-8">
          <h2 className="text-xl font-medium text-gray-800 mb-4">Error Loading Dashboard</h2>
          <p className="text-gray-600 mb-6">
            We encountered an error while loading your data. Please try again or contact support if the problem persists.
          </p>
          <div className="text-red-600 p-4 bg-red-50 rounded-lg mb-6">
            {error}
          </div>
          <button
            onClick={() => window.location.reload()}
            className="w-full bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="flex flex-col min-h-screen bg-gray-50">
      {/* Header/Navigation */}
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
                <Image src="/icons/logo.png" alt="Logo" width={20} height={20} priority />
              </div>
              <h1 className="font-semibold text-gray-800 text-lg">MedRecord Hub</h1>
            </motion.div>
            <div className="flex items-center space-x-6">
              <motion.a 
                href="#" 
                className="text-gray-500 hover:text-indigo-600 text-sm font-medium transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Help
              </motion.a>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link href="/" className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center transition-colors">
                  <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  Sign Out
                </Link>
              </motion.div>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Welcome Banner */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-gradient-to-r from-indigo-600 to-indigo-500 text-white"
      >
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h2 className="text-3xl font-bold">Welcome to Patient Portal</h2>
          <p className="mt-3 text-indigo-100 opacity-90 text-lg">Track your X-ray scans and medical records in one place</p>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="flex-grow max-w-7xl mx-auto w-full px-6 py-10">
        {/* Search Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-xl shadow-sm border border-gray-50 p-8 mb-10"
        >
          <h3 className="text-xl font-medium text-gray-800 mb-5">Track Your X-Ray Analysis</h3>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-grow">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <svg className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <input
                  type="text"
                  value={referenceNumber}
                  onChange={(e) => setReferenceNumber(e.target.value)}
                  placeholder="Enter reference number provided by your doctor"
                  className="block w-full pl-12 pr-4 py-3.5 border text-gray-700 border-gray-200 rounded-lg leading-5 bg-gray-50 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                />
              </div>
              {error && (
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-2 text-sm text-red-600"
                >
                  {error}
                </motion.p>
              )}
            </div>
            <motion.button 
              onClick={handleTrackScan}
              disabled={isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="bg-indigo-600 text-white px-8 py-3.5 rounded-lg hover:bg-indigo-700 transition-colors font-medium shadow-sm hover:shadow disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Searching...
                </>
              ) : (
                "Track Scan"
              )}
            </motion.button>
          </div>
          <p className="text-sm text-gray-500 mt-4">
            Enter the reference number provided by your doctor to view your X-ray analysis results
          </p>
        </motion.div>

        {/* Analysis Results Section */}
        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-xl shadow-sm border border-gray-50 p-8 mb-10"
          >
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-medium text-gray-800">Analysis Results</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* X-ray Image */}
              <motion.div 
                whileHover={{ scale: 1.02 }}
                className="relative aspect-square rounded-lg overflow-hidden bg-gray-50"
              >
                {result.image_url ? (
                  <Image
                    src={result.image_url}
                    alt="X-ray Image"
                    fill
                    className="object-contain"
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-gray-400">No image available</p>
                  </div>
                )}
              </motion.div>
              
              {/* Analysis Details */}
              <div className="space-y-6">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <h4 className="text-lg font-medium text-gray-800 mb-2">Diagnosis</h4>
                  <div className="flex items-center space-x-2">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      result.analysis_result === "Normal" 
                        ? "bg-green-100 text-green-800"
                        : "bg-red-100 text-red-800"
                    }`}>
                      {result.analysis_result}
                    </span>
                    <span className="text-sm text-gray-500">
                      Confidence: {(result.confidence_score !== undefined && result.confidence_score !== null) ? 
                        ((result.confidence_score >= 1 ? result.confidence_score : result.confidence_score * 100).toFixed(2) + "%") : 
                        "N/A"}
                    </span>
                  </div>
                  
                  {/* Additional diagnosis details */}
                  {result.pneumonia_type && (
                    <div className="mt-2">
                      <span className="text-sm font-medium text-gray-700">Type:</span>
                      <span className="ml-2 text-sm text-gray-600">{result.pneumonia_type}</span>
                    </div>
                  )}
                  
                  {result.severity && (
                    <div className="mt-1">
                      <span className="text-sm font-medium text-gray-700">Severity:</span>
                      <span className="ml-2 text-sm text-gray-600">{result.severity}</span>
                    </div>
                  )}
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 }}
                >
                  <h4 className="text-lg font-medium text-gray-800 mb-2">Details</h4>
                  <div className="space-y-2 text-sm text-gray-600">
                    <p><span className="font-medium">Reference Number:</span> {result.reference_number || 'N/A'}</p>
                    <p><span className="font-medium">Patient Name:</span> {result.patient_name || 'Not provided'}</p>
                    <p><span className="font-medium">Analyzed By:</span> {result.doctor_name || 'N/A'}</p>
                    <p><span className="font-medium">Date:</span> {result.created_at ? new Date(result.created_at).toLocaleDateString() : 'N/A'}</p>
                  </div>
                </motion.div>
                
                {result.recommended_action && (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.7 }}
                  >
                    <h4 className="text-lg font-medium text-gray-800 mb-2">Recommended Action</h4>
                    <div className="p-3 bg-indigo-50 rounded-lg text-sm text-gray-700">
                      {result.recommended_action}
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </main>
  );
}
