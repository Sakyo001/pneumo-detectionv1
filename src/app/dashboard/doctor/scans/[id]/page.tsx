"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { PDFDownloadLink, Document, Page, Text, View, StyleSheet, Image as PDFImage } from "@react-pdf/renderer";

// Create styles for PDF
const styles = StyleSheet.create({
  page: {
    flexDirection: 'column',
    backgroundColor: '#FFFFFF',
    padding: 30,
  },
  header: {
    fontSize: 24,
    marginBottom: 5,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  subheader: {
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  section: {
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 14,
    marginBottom: 5,
    fontWeight: 'bold',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    paddingBottom: 5,
  },
  row: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  label: {
    width: '30%',
    fontSize: 12,
    fontWeight: 'bold',
  },
  value: {
    width: '70%',
    fontSize: 12,
  },
  imageContainer: {
    marginBottom: 20,
    alignItems: 'center',
  },
  image: {
    width: 300,
    height: 300,
    objectFit: 'contain',
    marginBottom: 10,
  },
  resultSection: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f9fafb',
    borderRadius: 5,
  },
  footer: {
    marginTop: 20,
    fontSize: 10,
    textAlign: 'center',
  },
  confidence: {
    height: 10,
    backgroundColor: '#e5e7eb',
    borderRadius: 5,
    marginTop: 5,
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 5,
  },
});

// PDF Document Component
const ScanDetailsPDF = ({ scan }: { scan: any }) => (
  <Document>
    <Page size="A4" style={styles.page}>
      <Text style={styles.header}>X-Ray Scan Report</Text>
      <Text style={styles.subheader}>Reference: {scan.referenceNumber}</Text>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Patient Information</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Name:</Text>
          <Text style={styles.value}>{scan.patientName}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Date:</Text>
          <Text style={styles.value}>{new Date(scan.date).toLocaleDateString()}</Text>
        </View>
      </View>
      
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Scan Results</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Diagnosis:</Text>
          <Text style={styles.value}>{scan.result}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Confidence:</Text>
          <Text style={styles.value}>{scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : scan.confidence}%</Text>
        </View>
        {scan.pneumoniaType && (
          <View style={styles.row}>
            <Text style={styles.label}>Type:</Text>
            <Text style={styles.value}>{scan.pneumoniaType}</Text>
          </View>
        )}
        {scan.severity && (
          <View style={styles.row}>
            <Text style={styles.label}>Severity:</Text>
            <Text style={styles.value}>{scan.severity}</Text>
          </View>
        )}
      </View>
      
      {scan.recommendedAction && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recommendations</Text>
          <Text style={styles.value}>{scan.recommendedAction}</Text>
        </View>
      )}
      
      <Text style={styles.footer}>
        Generated on {new Date().toLocaleDateString()} • MedRecord Hub
      </Text>
    </Page>
  </Document>
);

export default function ScanDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const scanId = params.id as string;
  
  const [scan, setScan] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchScanDetails = async () => {
      setIsLoading(true);
      try {
        // Attempt to fetch scan data from API
        const response = await fetch(`/api/scans/${scanId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch scan details');
        }
        
        const result = await response.json();
        
        if (result.success && result.data) {
          setScan(result.data);
        } else {
          // If API call fails to return data, use mock data
          generateMockData();
        }
      } catch (error) {
        console.error("Error fetching scan details:", error);
        setError("Could not load scan details. Please try again later.");
        // Use mock data when API fails
        generateMockData();
      } finally {
        setIsLoading(false);
      }
    };
    
    // Generate mock data for demo purposes
    const generateMockData = () => {
      // Create random result - more likely to be pneumonia since they're viewing details
      const isPneumonia = Math.random() > 0.3;
      const rawConfidence = 0.80 + ((Math.floor(Math.random() * 20)) / 100); // 0.80-0.99
      const mockScan = {
        id: scanId,
        patientName: `Patient ${String.fromCharCode(65 + Math.floor(Math.random() * 26))}`,
        date: new Date(Date.now() - Math.floor(Math.random() * 90) * 86400000).toISOString(),
        result: isPneumonia ? 'Pneumonia' : 'Normal',
        confidence: Math.round(rawConfidence * 100), // Convert to percentage (80-99%)
        pneumoniaType: isPneumonia ? (Math.random() > 0.5 ? 'Bacterial' : 'Viral') : null,
        severity: isPneumonia ? (Math.random() > 0.66 ? 'Severe' : Math.random() > 0.33 ? 'Moderate' : 'Mild') : null,
        recommendedAction: isPneumonia ? 'Recommend antibiotic treatment and follow-up chest X-ray in 2 weeks.' : 'No action required. Regular check-up advised.',
        referenceNumber: `XR-${Math.floor(Math.random() * 1000000).toString().padStart(6, '0')}`,
        imageUrl: '/images/xray.png' // Default image path
      };
      
      setScan(mockScan);
    };
    
    fetchScanDetails();
  }, [scanId]);
  
  // Calculate severity level style
  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'mild': return 'text-green-600';
      case 'moderate': return 'text-yellow-600';
      case 'severe': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };
  
  const getSeverityBgColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'mild': return 'bg-green-600';
      case 'moderate': return 'bg-yellow-600';
      case 'severe': return 'bg-red-600';
      default: return 'bg-gray-600';
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
              <Link 
                href="/dashboard/doctor/" 
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
          <h2 className="text-2xl font-bold text-gray-800">X-Ray Scan Details</h2>
          <div className="flex space-x-4">
            <PDFDownloadLink 
              document={<ScanDetailsPDF scan={scan} />} 
              fileName={`xray-scan-${scan?.referenceNumber || 'report'}.pdf`}
              className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
            >
              {({ loading, error }) => (
                <>
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  {loading ? 'Generating PDF...' : 'Export to PDF'}
                </>
              )}
            </PDFDownloadLink>
          </div>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
            <svg className="w-12 h-12 text-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h3 className="text-lg font-medium text-red-800 mb-2">Error Loading Data</h3>
            <p className="text-red-600">{error}</p>
            <button 
              onClick={() => router.back()}
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              Go Back
            </button>
          </div>
        ) : scan ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left column: X-ray image */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-50 p-6 lg:col-span-1">
              <h3 className="text-lg font-medium text-gray-800 mb-4">X-Ray Image</h3>
              <div className="relative aspect-square w-full mb-4 bg-black rounded-lg overflow-hidden">
                <Image 
                  src={scan.imageUrl || "/images/xray.png"} 
                  alt="X-Ray" 
                  fill
                  className="object-contain"
                />
              </div>
              <div className="text-center">
                <span className="text-sm text-gray-900">Reference: {scan.referenceNumber}</span>
              </div>
            </div>
            
            {/* Right column: Scan details and results */}
            <div className="lg:col-span-2 space-y-6">
              {/* Patient information */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-50 p-6">
                <h3 className="text-lg font-medium text-gray-800 mb-4">Patient Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-900 mb-1">Name</p>
                    <p className="font-medium text-gray-900">{scan.patientName}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-900 mb-1">Scan Date</p>
                    <p className="font-medium text-gray-900">{new Date(scan.date).toLocaleDateString()}</p>
                  </div>
                </div>
              </div>
  
              {/* Analysis results */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-50 p-6">
                <h3 className="text-lg font-medium text-gray-800 mb-4">Analysis Results</h3>
                
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <div>
                      <p className="text-sm text-gray-900 mb-1">Diagnosis</p>
                      <div className="flex items-center">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          scan.result === 'Pneumonia' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                        }`}>
                          {scan.result}
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-900 mb-1">Confidence</p>
                      <p className="font-medium text-gray-900">{scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : scan.confidence}%</p>
                    </div>
                  </div>
                  
                  {/* Confidence bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${scan.result === 'Pneumonia' ? 'bg-red-600' : 'bg-green-600'}`} 
                      style={{ width: `${scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : scan.confidence}%` }}
                    ></div>
                  </div>
                </div>
  
                {scan.result === 'Pneumonia' && (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                      {scan.pneumoniaType && (
                        <div>
                          <p className="text-sm text-gray-900 mb-1">Type</p>
                          <p className="font-medium text-gray-900">{scan.pneumoniaType}</p>
                        </div>
                      )}
                      {scan.severity && (
                        <div>
                          <p className="text-sm text-gray-900 mb-1">Severity</p>
                          <p className={`font-medium ${getSeverityColor(scan.severity)}`}>
                            {scan.severity}
                          </p>
                        </div>
                      )}
                    </div>
                    
                    {/* Severity indicator */}
                    {scan.severity && (
                      <div className="mb-6">
                        <p className="text-sm text-gray-900 mb-2">Severity Level</p>
                        <div className="relative pt-1">
                          <div className="flex mb-2 items-center justify-between">
                            <div className="text-xs font-semibold text-green-600">Mild</div>
                            <div className="text-xs font-semibold text-yellow-600">Moderate</div>
                            <div className="text-xs font-semibold text-red-600">Severe</div>
                          </div>
                          <div className="flex h-2 overflow-hidden rounded bg-gray-200">
                            <div 
                              className="bg-green-500" 
                              style={{ width: '33.3%' }}
                            ></div>
                            <div 
                              className="bg-yellow-500" 
                              style={{ width: '33.3%' }}
                            ></div>
                            <div 
                              className="bg-red-500" 
                              style={{ width: '33.3%' }}
                            ></div>
                          </div>
                          <div 
                            className={`absolute w-3 h-3 rounded-full -mt-0.5 transform -translate-x-1/2 ${getSeverityBgColor(scan.severity)}`}
                            style={{ 
                              left: scan.severity === 'Mild' ? '16.65%' : 
                                   scan.severity === 'Moderate' ? '50%' : '83.35%' 
                            }}
                          ></div>
                        </div>
                      </div>
                    )}
                  </>
                )}
  
                {scan.recommendedAction && (
                  <div>
                    <p className="text-sm text-gray-900 mb-1">Recommended Action</p>
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                      <p className="text-sm text-gray-900">{scan.recommendedAction}</p>
                    </div>
                  </div>
                )}
              </div>
  
              {/* Buttons */}
              <div className="flex space-x-4">
                <button
                  onClick={() => router.push('/dashboard/doctor/upload-xray')}
                  className="flex-1 inline-flex justify-center items-center px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
                >
                  Upload New X-Ray
                </button>
                <button
                  onClick={() => router.back()}
                  className="flex-1 inline-flex justify-center items-center px-4 py-2 bg-white text-indigo-600 font-medium rounded-md border border-indigo-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
                >
                  Back to List
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
            <svg className="w-12 h-12 text-yellow-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <h3 className="text-lg font-medium text-yellow-800 mb-2">Scan Not Found</h3>
            <p className="text-yellow-600">The requested scan could not be found.</p>
            <button 
              onClick={() => router.push('/dashboard/doctor/all-scans')}
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
            >
              View All Scans
            </button>
          </div>
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
                <span className="text-gray-500 text-sm">© {new Date().getFullYear()} MedRecord Hub. All rights reserved.</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
} 