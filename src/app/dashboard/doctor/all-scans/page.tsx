"use client";

import React, { useState, useEffect, useMemo } from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from 'next/dynamic';

// Use a single lazy-loaded component for PDF functionality
const PDFExport = dynamic(
  () => import('./pdf-export').then(mod => mod.default),
  { ssr: false, loading: () => (
    <button className="inline-flex items-center px-4 py-2 bg-gray-400 text-white font-medium rounded-md">
      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      Loading PDF Export...
    </button>
  )}
);

// Define the ScanData interface for type safety
interface ScanData {
  id: string;
  patientName: string;
  date: string;
  result: string;
  confidence: number;
  [key: string]: any; // Allow additional properties
}

export default function AllScansPage() {
  const router = useRouter();
  // Add custom CSS for the pagination transitions
  useEffect(() => {
    // Create a style element for the pagination transitions
    const style = document.createElement('style');
    style.innerHTML = `
      .pagination-container {
        transition: opacity 0.2s ease-in-out;
      }
      .pagination-container.changing-page {
        opacity: 0.7;
        pointer-events: none;
      }
      .pagination-container button, 
      .pagination-container span {
        min-width: 2.5rem;
        height: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background-color 0.2s ease, color 0.2s ease;
      }
      .pagination-controls {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background-color: #f9fafb;
        border-radius: 0.375rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      }
      .pagination-controls button {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.375rem;
        padding: 0.5rem 0.75rem;
        font-weight: 500;
        color: #4b5563;
        transition: all 0.2s ease;
      }
      .pagination-controls button:hover:not(:disabled) {
        background-color: #f3f4f6;
        color: #1f2937;
      }
      .pagination-controls button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .pagination-controls .active-page {
        background-color: #4f46e5;
        color: white;
        border-color: #4f46e5;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);
  
  const [dashboardData, setDashboardData] = useState({
    totalScans: 0,
    pneumoniaCases: 0,
    normalCases: 0,
    todayScans: 0,
    recentScans: [] as any[]
  });
  const [allScans, setAllScans] = useState<ScanData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<{
    key: string;
    direction: 'ascending' | 'descending';
  }>({ key: 'date', direction: 'descending' });
  const [isBrowser, setIsBrowser] = useState(false);

  // Ensure ITEMS_PER_PAGE is never zero
  const ITEMS_PER_PAGE = 8;
  
  // Set isBrowser flag once component mounts to avoid hydration issues
  useEffect(() => {
    setIsBrowser(true);
    console.log("Component mounted, checking pagination visibility");
    
    // Force a re-render after a short delay to ensure pagination is visible
    const timer = setTimeout(() => {
      console.log("Forcing pagination visibility check");
      setCurrentPage(prev => prev);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);
  
  useEffect(() => {
    const fetchAllScans = async () => {
      setIsLoading(true);
      try {
        // Fetch all scans from API endpoint
        const response = await fetch('/api/scans/all');
        
        if (!response.ok) {
          throw new Error('Failed to fetch scans');
        }
        
        const result = await response.json();
        
        if (result.success && result.data) {
          setDashboardData({
            totalScans: result.data.totalScans,
            pneumoniaCases: result.data.pneumoniaCases,
            normalCases: result.data.normalCases,
            todayScans: result.data.todayScans,
            recentScans: []
          });
          
          setAllScans(result.data.scans.map((scan: any) => ({
            ...scan,
            // Format date for display
            date: new Date(scan.date).toLocaleDateString()
          })));
        } else {
          // Fallback to mock data
          generateMockData();
        }
      } catch (error) {
        console.error("Error fetching scans:", error);
        // Use mock data when API fails
        generateMockData();
      } finally {
        setIsLoading(false);
      }
    };
    
    // Generate mock data with consistent ids for demo purposes
    const generateMockData = () => {
      const totalScans = 50; // Generate 50 records to ensure multiple pages with ITEMS_PER_PAGE = 8
      const pneumoniaCases = 20; // Fixed number to avoid hydration mismatch
      const normalCases = totalScans - pneumoniaCases;
      
      // Generate all scans with deterministic data
      const mockScans = Array.from({ length: totalScans }, (_, i) => {
        const isPneumonia = i < pneumoniaCases;
        const mockDate = new Date(2023, 0, 1 + i); // Fixed date pattern
        // Use 0.80-0.99 range for confidence to match API format (which is multiplied by 100)
        const rawConfidence = 0.80 + ((i % 20) / 100); 
        return {
          id: `scan-${i}`,
          patientName: `Patient ${String.fromCharCode(65 + (i % 26))}${Math.floor(i / 26) + 1}`,
          date: mockDate.toLocaleDateString(),
          result: isPneumonia ? 'Pneumonia' : 'Normal',
          confidence: Math.round(rawConfidence * 100), // Convert to percentage (80-99%)
        };
      });
      
      setDashboardData({
        totalScans,
        pneumoniaCases,
        normalCases,
        todayScans: 3, // Fixed number
        recentScans: []
      });
      
      setAllScans(mockScans);
      console.log(`Mock data generated: ${mockScans.length} scans`);
    };
    
    fetchAllScans();
  }, []);
  
  // Sort scans based on current config - using safe type handling
  const sortedScans = useMemo(() => {
    if (!allScans || allScans.length === 0) return [];
    
    // Create a safe copy to avoid modifying original
    let sortableScans = [...allScans];
    
    if (sortConfig.key) {
      try {
        sortableScans.sort((a, b) => {
          // Safely handle potentially undefined values
          const aValue = a[sortConfig.key] || '';
          const bValue = b[sortConfig.key] || '';
          
          if (aValue < bValue) {
            return sortConfig.direction === 'ascending' ? -1 : 1;
          }
          if (aValue > bValue) {
            return sortConfig.direction === 'ascending' ? 1 : -1;
          }
          return 0;
        });
      } catch (error) {
        console.error("Sort error:", error);
        // Return unsorted if error
        return sortableScans;
      }
    }
    
    return sortableScans;
  }, [allScans, sortConfig]);
  
  // Filter scans based on search term - with improved error handling
  const filteredScans = sortedScans;
  
  // Paginate scans - with validation
  const paginatedScans = useMemo(() => {
    if (!filteredScans || filteredScans.length === 0) return [];
    
    try {
      const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
      return filteredScans.slice(startIndex, startIndex + ITEMS_PER_PAGE);
    } catch (error) {
      console.error("Pagination error:", error);
      return [];
    }
  }, [filteredScans, currentPage, ITEMS_PER_PAGE]);
  
  // Handle sorting with error prevention
  const requestSort = (key: string) => {
    if (!key) return;
    
    let direction: 'ascending' | 'descending' = 'ascending';
    try {
      if (sortConfig.key === key && sortConfig.direction === 'ascending') {
        direction = 'descending';
      }
      setSortConfig({ key, direction });
    } catch (error) {
      console.error("Sort config error:", error);
    }
  };
  
  // Calculate total pages safely
  const calculateTotalPages = () => {
    if (filteredScans.length === 0) return 1;
    return Math.max(1, Math.ceil(filteredScans.length / ITEMS_PER_PAGE));
  };
  
  const totalPages = calculateTotalPages();
  
  // Debug log for development
  useEffect(() => {
    console.log(`Debug: Total Pages: ${totalPages}, Items Per Page: ${ITEMS_PER_PAGE}, Total Items: ${filteredScans.length}`);
  }, [totalPages, filteredScans.length]);
  
  // Handle page changes - safely
  const handlePageChange = (newPage: number) => {
    // Ensure page is within valid range
    if (newPage < 1 || newPage > totalPages) return;
    
    // Prevent rapid clicks
    const paginationContainer = document.querySelector('.pagination-container');
    if (paginationContainer) {
      paginationContainer.classList.add('changing-page');
      setTimeout(() => {
        paginationContainer.classList.remove('changing-page');
      }, 300);
    }
    
    // Update current page
    setCurrentPage(newPage);
    
    // Scroll to top for better UX - with smooth behavior
    window.scrollTo({ 
      top: 0, 
      behavior: 'smooth' 
    });
  };
  
  // Get array of page numbers for pagination
  const getPageRange = () => {
    const range = [];
    
    // Always make sure we display at least page 1
    range.push(1);
    
    // If we have multiple pages, add more as needed
    if (totalPages > 1) {
      // Add ellipsis if needed
      if (currentPage > 3) {
        range.push('...');
      }
      
      // Add pages around current page
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      
      for (let i = start; i <= end && i < totalPages; i++) {
        range.push(i);
      }
      
      // Add ellipsis if needed
      if (currentPage < totalPages - 2) {
        range.push('...');
      }
      
      // Include last page if it's different from first
      if (totalPages > 1) {
        range.push(totalPages);
      }
    }
    
    // Log the range for debugging
    console.log(`Page range: ${range.join(', ')}`);
    
    return range;
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
              <Link 
                href="/dashboard/doctor" 
                className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center"
              >
                <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Dashboard
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-grow max-w-7xl mx-auto w-full px-6 py-10">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800">All X-Ray Scans</h2>
          
          <div className="flex space-x-4">
            {/* PDF Export - only rendered on client side */}
            {isBrowser && !isLoading && (
              <PDFExport scans={filteredScans} dashboardData={dashboardData} />
            )}
          </div>
        </div>
        
        {/* Stats overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-50">
            <p className="text-sm text-gray-500 mb-1">Total Scans</p>
            <h3 className="text-xl font-bold text-gray-800">{dashboardData.totalScans}</h3>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-50">
            <p className="text-sm text-gray-500 mb-1">Pneumonia Cases</p>
            <h3 className="text-xl font-bold text-red-600">{dashboardData.pneumoniaCases}</h3>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-50">
            <p className="text-sm text-gray-500 mb-1">Normal Cases</p>
            <h3 className="text-xl font-bold text-green-600">{dashboardData.normalCases}</h3>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-50">
            <p className="text-sm text-gray-500 mb-1">Today's Scans</p>
            <h3 className="text-xl font-bold text-indigo-600">{dashboardData.todayScans}</h3>
          </div>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
          </div>
        ) : (
          <>
            {/* Scans Table */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-50 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                        onClick={() => requestSort('patientName')}
                      >
                        <div className="flex items-center">
                          Patient
                          {sortConfig.key === 'patientName' && (
                            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortConfig.direction === 'ascending' ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                            </svg>
                          )}
                        </div>
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                        onClick={() => requestSort('date')}
                      >
                        <div className="flex items-center">
                          Date
                          {sortConfig.key === 'date' && (
                            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortConfig.direction === 'ascending' ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                            </svg>
                          )}
                        </div>
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                        onClick={() => requestSort('result')}
                      >
                        <div className="flex items-center">
                          Result
                          {sortConfig.key === 'result' && (
                            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortConfig.direction === 'ascending' ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                            </svg>
                          )}
                        </div>
                      </th>
                      <th 
                        scope="col" 
                        className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                        onClick={() => requestSort('confidence')}
                      >
                        <div className="flex items-center">
                          Confidence
                          {sortConfig.key === 'confidence' && (
                            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortConfig.direction === 'ascending' ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                            </svg>
                          )}
                        </div>
                      </th>
                      <th scope="col" className="relative px-6 py-3">
                        <span className="sr-only">Actions</span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {paginatedScans.length > 0 ? (
                      paginatedScans.map((scan, index) => (
                        <tr key={scan.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900">{scan.patientName || 'Unknown'}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-500">{scan.date || 'Unknown'}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${scan.result === 'Pneumonia' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                              {scan.result || 'Unknown'}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <div className="flex items-center">
                              <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2 max-w-[100px]">
                                <div 
                                  className={`h-2.5 rounded-full ${scan.result === 'Pneumonia' ? 'bg-red-600' : 'bg-green-600'}`} 
                                  style={{ width: `${scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : scan.confidence}%` }}
                                ></div>
                              </div>
                              <span>{scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : scan.confidence}%</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <Link 
                              href={`/dashboard/doctor/scans/${scan.id}`} 
                              className="text-indigo-600 hover:text-indigo-900"
                            >
                              View
                            </Link>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={5} className="px-6 py-4 text-center text-sm text-gray-500">
                          No scans found matching your search criteria.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              
              {/* Fixed height container for pagination */}
              <div className="h-[72px] border-t border-gray-200 bg-gray-50">
                <div className="flex items-center justify-between px-6 py-4">
                  <div className="flex-1 flex items-center justify-between">
                    <div className="w-[280px]">
                      <p className="text-sm text-gray-700">
                        Showing <span className="font-medium">{(currentPage - 1) * ITEMS_PER_PAGE + 1}</span> to <span className="font-medium">{Math.min(currentPage * ITEMS_PER_PAGE, filteredScans.length)}</span> of <span className="font-medium">{filteredScans.length}</span> results
                      </p>
                    </div>
                    <div className="min-w-[320px] flex justify-center">
                      {/* Simple pagination controls that are always visible */}
                      <div className="flex items-center space-x-2">
                        {/* Previous button */}
                        <button
                          onClick={() => handlePageChange(currentPage - 1)}
                          disabled={currentPage === 1}
                          className={`flex items-center justify-center w-10 h-10 rounded-md ${
                            currentPage === 1 
                              ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                              : 'bg-indigo-600 text-white hover:bg-indigo-700'
                          }`}
                        >
                          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M12.79 5.23a.75.75 0 01-.02 1.06L8.832 10l3.938 3.71a.75.75 0 11-1.04 1.08l-4.5-4.25a.75.75 0 010-1.08l4.5-4.25a.75.75 0 011.06.02z" clipRule="evenodd" />
                          </svg>
                        </button>
                        
                        {/* Page numbers */}
                        <div className="flex items-center space-x-1">
                          {getPageRange().map((page, index) => (
                            page === '...' ? (
                              <span
                                key={`ellipsis-${index}`}
                                className="flex items-center justify-center w-10 h-10 text-gray-500"
                              >
                                ...
                              </span>
                            ) : (
                              <button
                                key={`page-${page}`}
                                onClick={() => handlePageChange(Number(page))}
                                className={`flex items-center justify-center w-10 h-10 rounded-md ${
                                  currentPage === page
                                    ? 'bg-indigo-600 text-white'
                                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-indigo-50'
                                }`}
                              >
                                {page}
                              </button>
                            )
                          ))}
                        </div>
                        
                        {/* Next button */}
                        <button
                          onClick={() => handlePageChange(currentPage + 1)}
                          disabled={currentPage === totalPages}
                          className={`flex items-center justify-center w-10 h-10 rounded-md ${
                            currentPage === totalPages 
                              ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                              : 'bg-indigo-600 text-white hover:bg-indigo-700'
                          }`}
                        >
                          <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
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
                <span className="text-gray-500 text-sm">© 2023 MedRecord Hub. All rights reserved.</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
} 