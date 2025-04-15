"use client";

import { useState, useRef, FormEvent, useEffect } from "react";
import Image from "next/image";
import Link from "next/link";
import { uploadXray } from "./actions";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from 'next/navigation';
import { v4 as uuidv4 } from 'uuid';

// Define the analysis result type to fix TypeScript errors
interface AnalysisResult {
  diagnosis: string;
  confidence: number;
  pneumoniaType: string | null;
  severity: string | null;
  severityDescription: string | null;
  recommendedAction: string;
  imageUrl: string;
  referenceNumber: string;
  timestamp: string;
  usingMock?: boolean; // Flag to indicate if using mock predictions
  error?: string; // Add error property for error handling
  dbSaved?: boolean; // Flag to indicate if the data was saved to the database
  scanId?: string; // Database ID of the saved scan
}

// Define doctor info interface
type DoctorInfo = {
  name: string;
  title: string;
  department: string;
  hospital: string;
};

export default function UploadXrayPage() {
  const styleRef = useRef<HTMLStyleElement | null>(null);
  
  // Add print styles
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    // Only run this on the client-side
    if (!styleRef.current) {
      // Add print stylesheet
      styleRef.current = document.createElement('style');
      styleRef.current.innerHTML = `
        @media print {
          header, footer, .no-print, button, h2.text-2xl, .bg-indigo-50 {
            display: none !important;
          }
          body, html {
            margin: 0;
            padding: 0;
            font-size: 12pt;
            background: white !important;
          }
          .page-break {
            page-break-after: always;
            clear: both;
            display: block;
            height: 0;
          }
          .print-only {
            display: block !important;
          }
          @page {
            size: letter;
            margin: 0.75in;
          }
          .mx-auto {
            float: none !important;
            margin-left: auto !important;
            margin-right: auto !important;
          }
          h1, h2, h3, p {
            orphans: 3;
            widows: 3;
          }
          h1, h2, h3 {
            page-break-after: avoid;
          }
        }
      `;
      document.head.appendChild(styleRef.current);
    }
    
    return () => {
      if (styleRef.current && document.head.contains(styleRef.current)) {
        document.head.removeChild(styleRef.current);
        styleRef.current = null;
      }
    };
  }, []);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [modelInfo, setModelInfo] = useState<string>("EfficientNet Model");
  
  // Step navigation for results display
  const [currentStep, setCurrentStep] = useState(0);
  const totalSteps = 4; // Diagnosis, Pneumonia Type, Severity, Recommendations
  
  // Patient information
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientGender, setPatientGender] = useState("male");
  const [patientNotes, setPatientNotes] = useState("");
  const [referenceNumber, setReferenceNumber] = useState<string>("");
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Update the doctor info to fetch from the database context
  const [doctorInfo, setDoctorInfo] = useState<DoctorInfo>({
    name: "Dr. Jonathan Smith",
    title: "Radiologist, MD",
    department: "Department of Radiology",
    hospital: "Central Medical Center"
  });

  // Fetch doctor information from database
  useEffect(() => {
    const fetchDoctorInfo = async () => {
      try {
        // First try to get info from localStorage for immediate display
        let doctorName = "Dr. John Doe"; // Default fallback
        let found = false;

        // Try localStorage first for instant results
        try {
          const storedUserInfo = localStorage.getItem('userInfo');
          if (storedUserInfo) {
            const user = JSON.parse(storedUserInfo);
            if (user && user.name) {
              doctorName = user.name;
              found = true;
              
              // Immediately update state for faster display
              setDoctorInfo(prev => ({
                ...prev,
                name: doctorName
              }));
            }
          }
        } catch (e) {
          console.error("Error accessing localStorage:", e);
        }

        // Only make API call if we couldn't find in localStorage 
        // to minimize load time but ensure we get updated data
        if (!found) {
          // Simple fetch with timeout for quicker response
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 1500); // 1.5s timeout
          
          try {
            const response = await fetch('/api/auth/user', {
              signal: controller.signal
            });
            
            if (response.ok) {
              const data = await response.json();
              if (data && data.user && data.user.name) {
                doctorName = data.user.name;
                
                // Also store in localStorage for future fast access
                localStorage.setItem('userInfo', JSON.stringify({
                  name: doctorName,
                  // other user fields as needed
                }));
              }
            }
          } catch (err) {
            console.log("Could not fetch user data, using default");
          } finally {
            clearTimeout(timeoutId);
          }
        }
        
        // Update state with final result
        setDoctorInfo(prev => ({
          ...prev,
          name: doctorName
        }));
      } catch (error) {
        console.error("Error in doctor info loading:", error);
        // Set a default name as fallback
        setDoctorInfo(prev => ({
          ...prev,
          name: "Dr. John Doe"
        }));
      }
    };

    // Run immediately on component mount
    fetchDoctorInfo();
  }, []);

  // Initialize reference number and check server status only on client side
  useEffect(() => {
    // Generate reference number only on the client side
    if (typeof window !== 'undefined') {
      setReferenceNumber(generateReferenceNumber());
    }

    // Set up beforeunload warning if upload is in progress
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isUploading) {
        e.preventDefault();
        e.returnValue = '';
        return '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isUploading]); // Only re-run when isUploading changes

  const generateReferenceNumber = () => {
    // During server-side rendering, return a placeholder
    if (typeof window === 'undefined') {
      return 'XR-XXXXXX-XXXX';
    }
    
    const now = new Date();
    const dateStr = `${now.getFullYear().toString().slice(-2)}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}`;
    const randomPart = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
    return `XR-${dateStr}-${randomPart}`;
  };

  const checkServerStatus = async () => {
    setServerStatus('checking');
    try {
      const response = await fetch('/api/model-status');
      if (response.ok) {
        const data = await response.json();
        setServerStatus('online');
        if (data.model) {
          setModelInfo(data.model);
        }
      } else {
        setServerStatus('offline');
      }
    } catch (err) {
      console.error('Error checking server status:', err);
      setServerStatus('offline');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    setError(null);
    setAnalysisResult(null);

    // Create a preview URL
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setError(null);
      setAnalysisResult(null);
      
      // Create a preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleUpload = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile) {
      setError("Please select an X-ray image to upload");
      return;
    }

    if (!patientName) {
      setError("Please enter patient name");
      return;
    }
    
    setIsUploading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append("xrayFile", selectedFile);
      formData.append("patientName", patientName);
      formData.append("patientAge", patientAge);
      formData.append("patientGender", patientGender);
      formData.append("referenceNumber", referenceNumber);
      formData.append("patientNotes", patientNotes);
      
      const result = await uploadXray(formData) as AnalysisResult;
      
      if (result.error) {
        setError(result.error);
      } else {
        setAnalysisResult(result);
        setCurrentStep(0); // Start with the first step
        // Log the successful analysis
        console.log("Analysis complete with EfficientNet model:", result);
      }
    } catch (err) {
      console.error("Error uploading X-ray:", err);
      setError("An error occurred while uploading and analyzing the X-ray. Please try again or contact support if the issue persists.");
    } finally {
      setIsUploading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'mild': return 'text-green-600';
      case 'moderate': return 'text-yellow-600';
      case 'severe': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  // Function to handle step navigation
  const nextStep = () => {
    if (currentStep < totalSteps - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  // Reset everything including the step navigation when analyzing a new X-ray
  const resetForm = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setError(null);
    setPatientName("");
    setPatientAge("");
    setPatientGender("male");
    setPatientNotes("");
    setCurrentStep(0);
    setReferenceNumber(generateReferenceNumber());
    if (fileInputRef.current) fileInputRef.current.value = "";
    
    // Refresh the dashboard data to show the new scan
    router.refresh();
  };

  const router = useRouter();

  return (
    <main className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="mr-3 flex items-center justify-center w-8 h-8">
                <Image src="/icons/logo.png" alt="Logo" width={20} height={20} />
              </div>
              <h1 className="font-semibold text-gray-800 text-lg">MedRecord Hub</h1>
            </div>
            <Link 
              href="/dashboard/doctor" 
              className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center transition-colors"
            >
              <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Dashboard
            </Link>
          </div>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-grow max-w-7xl mx-auto w-full px-6 py-10">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload X-Ray for Analysis</h2>
        <div className="bg-indigo-50 border border-indigo-100 rounded-md p-3 mb-6">
          <div className="flex items-center">
            <div className="mr-3 text-indigo-500">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            </div>
            <div>
              <p className="text-sm text-indigo-800 font-medium">Using EfficientNet model for highly accurate pneumonia detection</p>
              <p className="text-xs text-indigo-600 mt-0.5">Our EfficientNet model achieves up to 96% accuracy on pneumonia classification from chest X-rays.</p>
            </div>
          </div>
        </div>
        
        {/* X-ray upload and analysis section */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-50 p-8">
          {!analysisResult ? (
            <form onSubmit={handleUpload} suppressHydrationWarning>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div>
                  <h3 className="text-lg font-medium text-gray-800 mb-4">Patient Information</h3>
                  <div className="space-y-4">
                    <div>
                      <label htmlFor="patientName" className="block text-sm font-medium text-gray-700 mb-1">
                        Patient Name
                      </label>
                      <input
                        id="patientName"
                        type="text"
                        value={patientName}
                        onChange={(e) => setPatientName(e.target.value)}
                        className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        required
                        key="patient-name-input"
                      />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label htmlFor="patientAge" className="block text-sm font-medium text-gray-700 mb-1">
                          Age
                        </label>
                        <input
                          id="patientAge"
                          type="number"
                          value={patientAge}
                          onChange={(e) => setPatientAge(e.target.value)}
                          className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                          key="patient-age-input"
                        />
                      </div>
                      <div>
                        <label htmlFor="patientGender" className="block text-sm font-medium text-gray-700 mb-1">
                          Gender
                        </label>
                        <select
                          id="patientGender"
                          value={patientGender}
                          onChange={(e) => setPatientGender(e.target.value)}
                          className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                          key="patient-gender-select"
                        >
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                          <option value="other">Other</option>
                        </select>
                      </div>
                    </div>
                    
                    <div>
                      <label htmlFor="referenceNumber" className="block text-sm font-medium text-gray-700 mb-1">
                        Reference Number
                      </label>
                      <input
                        id="referenceNumber"
                        type="text"
                        value={referenceNumber}
                        readOnly
                        className="w-full px-3 py-2 bg-gray-50 border text-gray-600 border-gray-300 rounded-md"
                        key="reference-number-input"
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="patientNotes" className="block text-sm font-medium text-gray-700 mb-1">
                        Clinical Notes
                      </label>
                      <textarea
                        id="patientNotes"
                        value={patientNotes}
                        onChange={(e) => setPatientNotes(e.target.value)}
                        rows={3}
                        className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        placeholder="Any symptoms or relevant medical history..."
                        key="patient-notes-textarea"
                      />
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium text-gray-800 mb-4">X-Ray Upload</h3>
                  <div 
                    className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center hover:border-indigo-500 transition-colors cursor-pointer"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    {previewUrl ? (
                      <div className="space-y-4">
                        <div className="relative h-48 w-full">
                          <Image 
                            src={previewUrl} 
                            alt="X-ray preview" 
                            fill
                            style={{ objectFit: 'contain' }} 
                          />
                        </div>
                        <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                        <button 
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedFile(null);
                            setPreviewUrl(null);
                            if (fileInputRef.current) fileInputRef.current.value = "";
                          }}
                          className="text-indigo-600 text-sm hover:text-indigo-800"
                          key="change-file-button"
                        >
                          Change file
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        <div className="flex text-sm text-gray-600 justify-center">
                          <label 
                            htmlFor="xray-upload" 
                            className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500"
                          >
                            <span>Upload an X-ray</span>
                            <input 
                              id="xray-upload" 
                              name="xray-upload" 
                              type="file" 
                              className="sr-only" 
                              accept="image/*"
                              onChange={handleFileChange}
                              ref={fileInputRef}
                              key="xray-file-input"
                            />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs text-gray-500">PNG, JPG, JPEG up to 10MB</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {error && (
                <div className="mb-6 p-3 bg-red-50 text-red-700 rounded-md">
                  {error}
                </div>
              )}
              
              <div className="flex justify-between items-center">
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full ${
                    serverStatus === 'online' ? 'bg-green-500' : 
                    serverStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
                  } mr-2 ${serverStatus === 'checking' ? 'animate-pulse' : ''}`}></div>
                  <span className="text-xs text-gray-500">
                    {modelInfo} - {
                      serverStatus === 'online' ? 'Online' : 
                      serverStatus === 'offline' ? 'Offline' : 'Checking...'
                    }
                  </span>
                  <button 
                    className="ml-2 text-xs text-indigo-600 cursor-pointer hover:text-indigo-800 hover:underline"
                    onClick={(e) => {
                      e.preventDefault();
                      checkServerStatus();
                    }}
                  >
                    Check Status
                  </button>
                  {serverStatus === 'offline' && (
                    <span className="ml-2 text-xs text-red-500">
                      Using fallback predictions
                    </span>
                  )}
                </div>
                <button
                  type="submit"
                  disabled={isUploading || !selectedFile || !patientName}
                  className={`py-2.5 px-6 rounded-md text-white font-medium 
                    ${isUploading || !selectedFile || !patientName 
                      ? 'bg-gray-400 cursor-not-allowed' 
                      : 'bg-indigo-600 hover:bg-indigo-700'}`}
                >
                  {isUploading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing...
                    </span>
                  ) : (
                    'Analyze X-Ray'
                  )}
                </button>
              </div>
            </form>
          ) : (
            <div>
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-xl font-bold text-gray-800 mb-3">X-Ray Analysis Results</h3>
                  <div className="text-sm text-gray-500 mb-4">
                    Ref: {analysisResult.referenceNumber} | Timestamp: {new Date(analysisResult.timestamp).toLocaleString()}
                  </div>
                  
                  {/* Database save status message */}
                  <div className="p-2 rounded-md mb-4 bg-gray-50 border border-gray-100 no-print">
                    {analysisResult.dbSaved ? (
                      <div className="flex items-center text-green-700 text-sm">
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        {analysisResult.scanId && `Scan ID: ${analysisResult.scanId.substring(0, 8)}...`}
                      </div>
                    ) : (
                      <div className="flex items-center text-yellow-700 text-sm">
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        Scan results displayed but not saved to database
                      </div>
                    )}
                  </div>
                </div>
                
                <div>
                <button
                    type="button"
                  onClick={resetForm}
                    className="text-indigo-600 font-medium"
                >
                    Analyze New X-Ray
                </button>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <div className="mb-4 bg-gray-50 rounded-lg p-2">
                    <div className="relative h-72 w-full">
                      <Image 
                        src={previewUrl!} 
                        alt="X-ray" 
                        fill
                        style={{ objectFit: 'contain' }} 
                      />
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-sm font-medium text-gray-500 mb-2">Patient Information</h4>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="grid grid-cols-2 gap-4 mb-2">
                        <div>
                          <p className="text-xs text-gray-500">Name</p>
                          <p className="font-medium text-gray-800">{patientName}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Reference</p>
                          <p className="font-medium text-gray-800">{referenceNumber}</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs text-gray-500">Age</p>
                          <p className="font-medium text-gray-800">{patientAge}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Gender</p>
                          <p className="font-medium text-gray-800">{patientGender}</p>
                        </div>
                      </div>
                      {patientNotes && (
                        <div className="mt-3">
                          <p className="text-xs text-gray-500">Clinical Notes</p>
                          <p className="text-sm text-gray-700 mt-1">{patientNotes}</p>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-sm font-medium text-gray-500 mb-2">About EfficientNet Model</h4>
                    <div className="bg-indigo-50 rounded-lg p-4 text-xs text-indigo-800">
                      <p className="mb-2">The EfficientNet model used in this analysis is trained on thousands of labeled chest X-ray images for accurate pneumonia detection.</p>
                      <p>Our latest model achieved 96% accuracy on validation data, with improved detection of both bacterial and viral pneumonia cases.</p>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-6">
                  {/* Step indicator */}
                  <div className="flex justify-between items-center mb-4">
                    {Array.from({length: totalSteps}).map((_, index) => (
                      <div 
                        key={index}
                        className={`h-2 rounded-full flex-grow mx-1 transition-colors duration-300 ${
                          index === currentStep ? 
                            (analysisResult.diagnosis === 'Pneumonia' ? 'bg-red-500' : 'bg-green-500') : 
                            index < currentStep ? 'bg-gray-400' : 'bg-gray-200'
                        }`}
                      />
                    ))}
                  </div>
                  
                  {/* Step content with animations */}
                  <div className="bg-white border border-gray-100 rounded-lg p-5 shadow-sm min-h-[300px]">
                    <AnimatePresence mode="wait">
                      {currentStep === 0 && (
                        <motion.div 
                          key="diagnosis"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-lg font-medium text-gray-800">Diagnosis</h4>
                      <span 
                        className={`px-3 py-1 rounded-full text-xs font-medium 
                          ${analysisResult.diagnosis === 'Pneumonia' 
                            ? 'bg-red-100 text-red-800' 
                            : 'bg-green-100 text-green-800'}`}
                      >
                        {analysisResult.diagnosis}
                      </span>
                    </div>
                          
                          <motion.div 
                            className="h-2 w-full bg-gray-200 rounded-full mb-1"
                            initial={{ width: "0%" }}
                            animate={{ width: "100%" }}
                            transition={{ duration: 0.5, delay: 0.2 }}
                          >
                            <motion.div 
                        className={`h-2 rounded-full ${
                          analysisResult.diagnosis === 'Pneumonia' 
                            ? 'bg-red-500' 
                            : 'bg-green-500'}`}
                              initial={{ width: "0%" }}
                              animate={{ width: `${analysisResult.confidence}%` }}
                              transition={{ duration: 0.8, delay: 0.3 }}
                            />
                          </motion.div>
                          
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Confidence: {analysisResult.confidence}%</span>
                            <span>{analysisResult.diagnosis === 'Pneumonia' ? 'Abnormal' : 'Normal'}</span>
                          </div>
                          
                          <div className="mt-6 text-sm">
                            <p className="text-gray-700 mb-3">
                              {analysisResult.diagnosis === 'Pneumonia' 
                                ? 'Analysis indicates patterns consistent with pneumonia in this X-ray image. Further evaluation is recommended.'
                                : 'Analysis indicates a normal chest X-ray with no significant findings suggestive of pneumonia or other abnormalities.'}
                            </p>
                            
                            <p className="text-gray-600">
                              {analysisResult.diagnosis === 'Pneumonia'
                                ? 'Pneumonia is characterized by inflammation of the air sacs in one or both lungs, which may fill with fluid.'
                                : 'A normal chest X-ray shows clear lung fields with no signs of infection, fluid accumulation, or other significant abnormalities.'}
                            </p>
                    </div>
                    
                    {analysisResult.usingMock && (
                            <div className="mt-4 p-2 bg-yellow-50 border border-yellow-100 rounded-md">
                        <div className="flex items-center text-yellow-800">
                          <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                          </svg>
                          <p className="text-xs">Using simulated predictions - model not fully trained</p>
                        </div>
                      </div>
                    )}
                        </motion.div>
                      )}
                      
                      {currentStep === 1 && analysisResult.diagnosis === 'Pneumonia' && (
                        <motion.div 
                          key="pneumonia-type"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          <div className="mb-4">
                        <h4 className="text-lg font-medium text-gray-800 mb-3">Pneumonia Type</h4>
                        <div className="flex items-center">
                              <div className={`w-3 h-3 rounded-full ${analysisResult.pneumoniaType === 'Bacterial' ? 'bg-blue-500' : 'bg-purple-500'} mr-2`}></div>
                              <span className="text-gray-800 font-medium">{analysisResult.pneumoniaType}</span>
                        </div>
                          </div>
                          
                          <div className="p-4 rounded-lg bg-gray-50 mb-4">
                            <h5 className="font-medium text-sm mb-2 text-indigo-700">Characteristics</h5>
                            <p className="text-sm text-gray-700 mb-3">
                          {analysisResult.pneumoniaType === 'Bacterial' 
                                ? 'Bacterial pneumonia typically presents as a dense, lobar consolidation with distinct borders. It often affects a specific segment or lobe of the lung.' 
                                : 'Viral pneumonia typically appears as a diffuse interstitial pattern or ground glass opacities. It may present with a more patchy and bilateral distribution.'}
                            </p>
                            <div className="grid grid-cols-2 gap-4 mt-3">
                              <div>
                                <h6 className="text-xs font-medium mb-1 text-indigo-600">Common Pathogens</h6>
                                <p className="text-xs text-gray-600">
                                  {analysisResult.pneumoniaType === 'Bacterial' 
                                    ? 'Streptococcus pneumoniae, Haemophilus influenzae, Staphylococcus aureus'
                                    : 'Influenza virus, RSV, COVID-19, adenovirus'}
                                </p>
                              </div>
                              <div>
                                <h6 className="text-xs font-medium mb-1 text-indigo-600">Onset</h6>
                                <p className="text-xs text-gray-600">
                                  {analysisResult.pneumoniaType === 'Bacterial' 
                                    ? 'Typically sudden onset with high fever'
                                    : 'Usually gradual onset with mild to moderate symptoms'}
                                </p>
                              </div>
                            </div>
                      </div>
                    
                          <div className="text-sm">
                            <h5 className="font-medium mb-2 text-indigo-700">Treatment Considerations</h5>
                            <p className="text-gray-700">
                          {analysisResult.pneumoniaType === 'Bacterial' 
                                ? 'Bacterial pneumonia typically responds well to appropriate antibiotic therapy. Treatment should be started promptly, especially in high-risk patients.' 
                                : 'Viral pneumonia often resolves with supportive care. Antiviral medications may be beneficial in specific cases if started early.'}
                        </p>
                      </div>
                        </motion.div>
                      )}
                      
                      {currentStep === 2 && analysisResult.diagnosis === 'Pneumonia' && (
                        <motion.div 
                          key="severity"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          <h4 className="text-lg font-medium text-gray-800 mb-3">Severity Assessment</h4>
                          
                          <div className="flex items-center mb-4">
                          <div className={`w-3 h-3 rounded-full mr-2 ${
                            getSeverityColor(analysisResult.severity || '').replace('text-', 'bg-')
                          }`}></div>
                            <span className={`${getSeverityColor(analysisResult.severity || '')} font-medium`}>
                              {analysisResult.severity} Pneumonia
                          </span>
                        </div>
                          
                          <motion.div 
                            className="w-full bg-gray-200 h-2 rounded-full mb-4"
                            initial={{ width: "0%" }}
                            animate={{ width: "100%" }}
                            transition={{ duration: 0.5 }}
                          >
                            <motion.div 
                              className={`h-2 rounded-full ${
                                analysisResult.severity === 'Mild' ? 'bg-green-500' :
                                analysisResult.severity === 'Moderate' ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              initial={{ width: "0%" }}
                              animate={{ width: `${
                                analysisResult.severity === 'Mild' ? '33%' :
                                analysisResult.severity === 'Moderate' ? '66%' : '100%'
                              }` }}
                              transition={{ duration: 0.8, delay: 0.2 }}
                            />
                          </motion.div>
                          
                          <div className="text-xs text-gray-500 flex justify-between mb-4">
                            <span>Mild</span>
                            <span>Moderate</span>
                            <span>Severe</span>
                          </div>
                          
                          <div className="p-4 bg-gray-50 rounded-lg mb-4">
                            <h5 className="font-medium text-sm mb-2 text-indigo-700">Clinical Assessment</h5>
                            <p className="text-sm text-gray-700">
                          {analysisResult.severityDescription}
                        </p>
                      </div>
                      
                          <div>
                            <h5 className="font-medium text-sm mb-2 text-indigo-700">CURB-65 Risk Assessment</h5>
                            <p className="text-xs text-gray-600 mb-2">The CURB-65 score helps assess the severity of pneumonia and the need for hospital admission:</p>
                            <ul className="text-xs text-gray-600 list-disc pl-5 space-y-1">
                              <li>Confusion</li>
                              <li>Blood Urea nitrogen &gt; 19 mg/dL</li>
                              <li>Respiratory rate ≥ 30 breaths/min</li>
                              <li>Blood pressure (systolic &lt; 90 mmHg or diastolic ≤ 60 mmHg)</li>
                              <li>Age ≥ 65 years</li>
                            </ul>
                            <p className="text-xs text-gray-600 mt-2">
                              Score 0-1: Consider outpatient treatment<br />
                              Score 2: Consider short hospital stay or supervised outpatient treatment<br />
                              Score 3-5: Hospitalize, consider ICU for score 4-5
                        </p>
                      </div>
                        </motion.div>
                      )}
                      
                      {currentStep === 3 && (
                        <motion.div 
                          key="recommendations"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          <h4 className="text-lg font-medium text-gray-800 mb-4">Clinical Recommendations</h4>
                          
                          {analysisResult.diagnosis === 'Pneumonia' ? (
                            <>
                              <div className="border-l-4 border-indigo-500 pl-3 mb-5">
                                <p className="text-gray-800 font-medium">Primary Recommendation</p>
                                <p className="text-sm text-gray-700 mt-1">{analysisResult.recommendedAction}</p>
                              </div>
                              
                              <div className="space-y-4 mt-6">
                                <div>
                                  <h5 className="font-medium text-sm mb-2 text-indigo-700">Additional Diagnostic Tests</h5>
                                  <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
                                    <li>Complete blood count with differential</li>
                                    <li>Blood cultures (before antibiotic administration)</li>
                                    <li>Sputum Gram stain and culture</li>
                                    <li>Pulse oximetry or arterial blood gas</li>
                                    {analysisResult.severity === 'Severe' && <li>Consider CT scan of the chest</li>}
                                  </ul>
                                </div>
                                
                                <div>
                                  <h5 className="font-medium text-sm mb-2 text-indigo-700">Treatment Options</h5>
                                  <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
                                    {analysisResult.pneumoniaType === 'Bacterial' ? (
                                      <>
                                        <li>Empiric antibiotic therapy based on local resistance patterns</li>
                                        <li>{analysisResult.severity === 'Mild' ? 'Outpatient oral antibiotics' : 'Consider IV antibiotics'}</li>
                                        <li>Reassess after 48-72 hours for clinical response</li>
                                      </>
                                    ) : (
                                      <>
                                        <li>Supportive care (rest, hydration, antipyretics)</li>
                                        <li>Consider antiviral therapy if presenting early</li>
                                        <li>Monitor for secondary bacterial infection</li>
                    </>
                  )}
                                    <li>Supplemental oxygen if SpO2 &lt; 92%</li>
                                  </ul>
                                </div>
                                
                                <div>
                                  <h5 className="font-medium text-sm mb-2 text-indigo-700">Follow-up Recommendations</h5>
                                  <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
                                    <li>Follow-up chest X-ray in 4-6 weeks to confirm resolution</li>
                                    <li>Clinical assessment within 48-72 hours for high-risk patients</li>
                                    <li>Patient education on preventing recurrence</li>
                                    <li>Consider pneumococcal and influenza vaccination</li>
                                  </ul>
                                </div>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="border-l-4 border-green-500 pl-3 mb-5">
                                <p className="text-gray-800 font-medium">Primary Recommendation</p>
                                <p className="text-sm text-gray-700 mt-1">{analysisResult.recommendedAction}</p>
                              </div>
                              
                              <div className="p-4 bg-gray-50 rounded-lg mb-4">
                                <h5 className="font-medium text-sm mb-2 text-indigo-700">Clinical Assessment</h5>
                                <p className="text-sm text-gray-700">
                                  The X-ray appears normal with no significant findings suggestive of pneumonia or other pulmonary pathology. The lungs are clear with no signs of consolidation, effusion, or infiltrates.
                        </p>
                      </div>
                              
                              <div className="space-y-4 mt-6">
                                <div>
                                  <h5 className="font-medium text-sm mb-2 text-indigo-700">Consider If Symptoms Persist</h5>
                                  <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
                                    <li>If respiratory symptoms persist, consider additional diagnostic testing</li>
                                    <li>Pulmonary function tests may be indicated for persistent cough or dyspnea</li>
                                    <li>Consider CT scan for better visualization if clinical suspicion remains high</li>
                                  </ul>
                                </div>
                                
                                <div>
                                  <h5 className="font-medium text-sm mb-2 text-indigo-700">Preventive Recommendations</h5>
                                  <ul className="text-sm text-gray-600 list-disc pl-5 space-y-1">
                                    <li>Maintain regular health check-ups</li>
                                    <li>Consider annual influenza vaccination</li>
                                    <li>Pneumococcal vaccination as per age and risk factor guidelines</li>
                                    <li>Smoking cessation if applicable</li>
                                  </ul>
                                </div>
                              </div>
                            </>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>
                    
                    {/* Navigation controls */}
                    <div className="flex justify-between mt-8">
                    <button 
                        type="button"
                        onClick={prevStep}
                        disabled={currentStep === 0}
                        className={`flex items-center text-sm font-medium ${currentStep === 0 ? 'text-gray-300 cursor-not-allowed' : 'text-indigo-600 hover:text-indigo-800'}`}
                      >
                        <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                        Previous
                      </button>
                      
                      {(currentStep < totalSteps - 1) && (
                        <button
                          type="button"
                          onClick={nextStep}
                          disabled={analysisResult.diagnosis !== 'Pneumonia' && currentStep >= 1}
                          className={`flex items-center text-sm font-medium ${
                            analysisResult.diagnosis !== 'Pneumonia' && currentStep >= 1 
                              ? 'text-gray-300 cursor-not-allowed' 
                              : 'text-indigo-600 hover:text-indigo-800'
                          }`}
                        >
                          Next
                          <svg className="w-5 h-5 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </button>
                      )}
                      
                      {currentStep === totalSteps - 1 && (
                    <button 
                          className="flex items-center bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-1.5 px-3 rounded-md"
                      onClick={() => window.print()}
                    >
                          <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
                          </svg>
                      Print Report
                    </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Print-only formal letter format */}
      {analysisResult && (
        <div className="print-only hidden">
          {/* First page - X-Ray Image and Basic Info */}
          <div className="mx-auto max-w-4xl">
            

            {/* Letter header */}
            <div className="flex justify-between items-center mb-8">
              <div>
                {/* Details moved here */}
              </div>
              <div className="text-right">
                <p>Date: {new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</p>
                <p>Reference: {referenceNumber}</p>
              </div>
            </div>
            
            {/* Patient information */}
            <div className="mb-6">
              <p><strong>Patient Name:</strong> {patientName}</p>
              <p><strong>Age:</strong> {patientAge} years</p>
              <p><strong>Gender:</strong> {patientGender}</p>
              {patientNotes && (
                <div className="mt-2">
                  <p><strong>Clinical Notes:</strong> {patientNotes}</p>
                </div>
              )}
            </div>
            
            {/* X-Ray Image */}
            <div className="mb-6 text-center">
              <h3 className="font-bold mb-2">Chest X-Ray Image:</h3>
              <div className="border border-gray-300 p-2 inline-block">
                <img 
                  src={previewUrl || ''} 
                  alt="Chest X-Ray" 
                  className="max-h-[400px] max-w-full"
                />
              </div>
            </div>
            
            {/* Signature section for this page */}
            <div className="mt-16">
              <div className="w-56 border-b border-black" style={{ minHeight: '30px' }}></div>
              <p className="text-sm text-gray-500 mb-1">Doctor's Signature:</p>
              <p className="font-bold mt-1">{doctorInfo.name}</p>
              <p>{doctorInfo.title}</p>
            </div>
            
            {/* Footer for first page */}
            <div className="text-center text-sm mt-12 pt-4">
              <p>{doctorInfo.hospital} | Page 1 of 3</p>
            </div>
          </div>
          
          {/* Page break after first page */}
          <div className="page-break"></div>
          
          {/* Second page - X-Ray Analysis Results */}
          <div className="mx-auto max-w-4xl">
            {/* Header for consistency */}
            <div className="text-center mb-6">
              <h1 className="text-2xl font-bold">{doctorInfo.hospital}</h1>
              <p>{doctorInfo.department}</p>
            </div>

            {/* Patient identifier */}
            <div className="mb-4">
              <p><strong>Patient:</strong> {patientName}</p>
              <p><strong>Reference:</strong> {referenceNumber}</p>
            </div>
            
            {/* Report title */}
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold underline">X-Ray Analysis Results</h2>
            </div>
            
            {/* Diagnosis */}
            <div className="mb-4">
              <h3 className="font-bold">Diagnosis:</h3>
              <p>{analysisResult.diagnosis} {analysisResult.confidence && `(${analysisResult.confidence}% confidence)`}</p>
              
              {analysisResult.diagnosis === 'Pneumonia' && (
                <>
                  <p><strong>Type:</strong> {analysisResult.pneumoniaType}</p>
                  <p><strong>Severity:</strong> {analysisResult.severity}</p>
                </>
              )}
            </div>
            
            {/* Findings */}
            <div className="mb-4">
              <h3 className="font-bold">Clinical Assessment:</h3>
              <p>
                {analysisResult.diagnosis === 'Pneumonia' 
                  ? analysisResult.severityDescription 
                  : 'The X-ray appears normal with no significant findings suggestive of pneumonia or other pulmonary pathology. The lungs are clear with no signs of consolidation, effusion, or infiltrates.'}
              </p>
            </div>
            
            {/* Signature section for this page */}
            <div className="mt-16">
              <div className="w-56 border-b border-black" style={{ minHeight: '30px' }}></div>
              <p className="text-sm text-gray-500 mb-1">Doctor's Signature:</p>
              <p className="font-bold mt-1">{doctorInfo.name}</p>
              <p>{doctorInfo.title}</p>
            </div>
            
            {/* Footer for second page */}
            <div className="text-center text-sm mt-12 pt-4">
              <p>{doctorInfo.hospital} | Page 2 of 3</p>
            </div>
          </div>
          
          {/* Page break after second page */}
          <div className="page-break"></div>
          
          {/* Third page - Clinical Recommendations */}
          <div className="mx-auto max-w-4xl">
            {/* Header for consistency */}
            <div className="text-center mb-6">
              <h1 className="text-2xl font-bold">{doctorInfo.hospital}</h1>
              <p>{doctorInfo.department}</p>
            </div>

            {/* Patient identifier */}
            <div className="mb-4">
              <p><strong>Patient:</strong> {patientName}</p>
              <p><strong>Reference:</strong> {referenceNumber}</p>
            </div>
            
            {/* Report title */}
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold underline">Clinical Recommendations</h2>
            </div>
            
            {/* Recommendations */}
            <div className="mb-8">
              <h3 className="font-bold">Recommendations:</h3>
              <p>{analysisResult.recommendedAction}</p>
              
              {analysisResult.diagnosis === 'Pneumonia' && (
                <div className="mt-2">
                  <p><strong>Additional Requirements:</strong></p>
                  <ul className="list-disc ml-8">
                    <li>Complete blood count with differential</li>
                    <li>Blood cultures (before antibiotic administration)</li>
                    <li>Sputum Gram stain and culture</li>
                    <li>Follow-up chest X-ray in 4-6 weeks to confirm resolution</li>
                    {analysisResult.severity === 'Severe' && <li>Consider CT scan of the chest</li>}
                  </ul>
                </div>
              )}
            </div>
            
            {/* Treatment plan details */}
            {analysisResult.diagnosis === 'Pneumonia' && (
              <div className="mb-8">
                <h3 className="font-bold">Treatment Plan:</h3>
                <div className="mt-2">
                  <p><strong>Medication:</strong></p>
                  <ul className="list-disc ml-8">
                    {analysisResult.pneumoniaType === 'Bacterial' ? (
                      <>
                        <li>Empiric antibiotic therapy based on local resistance patterns</li>
                        <li>{analysisResult.severity === 'Mild' ? 'Outpatient oral antibiotics' : 'Consider IV antibiotics'}</li>
                      </>
                    ) : (
                      <>
                        <li>Supportive care (rest, hydration, antipyretics)</li>
                        <li>Consider antiviral therapy if presenting early</li>
                      </>
                    )}
                    <li>Supplemental oxygen if SpO2 &lt; 92%</li>
                  </ul>
                </div>
              </div>
            )}
            
            {/* Signature section for this page */}
            <div className="mt-16">
              <div className="w-56 border-b border-black" style={{ minHeight: '30px' }}></div>
              <p className="text-sm text-gray-500 mb-1">Doctor's Signature:</p>
              <p className="font-bold mt-1">{doctorInfo.name}</p>
              <p>{doctorInfo.title}</p>
            </div>
            
            {/* Footer */}
            <div className="text-center text-sm mt-12 pt-4 border-t border-gray-300">
              <p>{doctorInfo.hospital} | {doctorInfo.department}</p>
              <p>© 2025 MedRecord Hub. All rights reserved. | Page 3 of 3</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Footer */}
      <footer className="bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex justify-center md:justify-start">
            <div className="flex items-center">
              <div className="mr-2 flex items-center justify-center w-7 h-7">
                <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
              </div>
              <span className="text-gray-500 text-sm">© 2025 MedRecord Hub. All rights reserved.</span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
} 