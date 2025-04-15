"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

interface PneumoniaRecord {
  id: string;
  reference_number: string;
  image_url?: string;
  analysis_result: string;
  pneumonia_type?: string;
  severity?: string;
  confidence_score: number;
  recommended_action?: string;
  createdAt: string;
  patient: {
    id: string;
    name: string;
    referenceNumber: string;
  };
  doctor: {
    id: string;
    name: string;
    email: string;
  };
}

interface PneumoniaFormData {
  id: string;
  reference_number: string;
  pneumonia_type: string;
  severity: string;
  confidence_score: number;
  recommended_action: string;
  patientId: string;
  doctorId: string;
}

export default function PneumoniaManagement() {
  const [records, setRecords] = useState<PneumoniaRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentRecord, setCurrentRecord] = useState<PneumoniaFormData | null>(null);

  // Fetch pneumonia records
  useEffect(() => {
    const fetchRecords = async () => {
      try {
        setIsLoading(true);
        setError(null); // Clear previous errors
        
        const response = await fetch("/api/admin/pneumonia");
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.details || "Failed to fetch pneumonia records");
        }
        
        const data = await response.json();
        setRecords(data);
      } catch (err) {
        console.error("Error fetching pneumonia records:", err);
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchRecords();
  }, []);

  const handleEditRecord = (record: PneumoniaRecord) => {
    setCurrentRecord({
      id: record.id,
      reference_number: record.reference_number,
      pneumonia_type: record.pneumonia_type || "",
      severity: record.severity || "",
      confidence_score: record.confidence_score,
      recommended_action: record.recommended_action || "",
      patientId: record.patient.id,
      doctorId: record.doctor.id,
    });
    setIsModalOpen(true);
  };

  const handleDeleteRecord = async (recordId: string) => {
    if (!confirm("Are you sure you want to delete this record?")) {
      return;
    }
    
    try {
      setError(null); // Clear previous errors
      
      const response = await fetch(`/api/admin/pneumonia?id=${recordId}`, {
        method: "DELETE",
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.details || "Failed to delete record");
      }
      
      // Remove from state
      setRecords(records.filter(record => record.id !== recordId));
    } catch (err) {
      console.error("Error deleting record:", err);
      setError(err instanceof Error ? err.message : "Failed to delete record");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentRecord) return;
    
    try {
      setError(null); // Clear previous errors
      
      const response = await fetch("/api/admin/pneumonia", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(currentRecord),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.details || "Failed to update record");
      }
      
      const updatedRecord = await response.json();
      
      // Update in state
      setRecords(records.map(record => 
        record.id === updatedRecord.id 
          ? { ...record, ...updatedRecord } 
          : record
      ));
      
      setIsModalOpen(false);
    } catch (err) {
      console.error("Error updating record:", err);
      setError(err instanceof Error ? err.message : "An error occurred");
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-50 p-6 mb-8">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-semibold text-gray-800">Pneumonia Data Management</h3>
      </div>

      {error && (
        <div className="bg-red-50 text-red-600 p-3 rounded-md mb-4">
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead>
              <tr>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Reference Number
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Patient
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Result
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-6 py-3 bg-gray-50 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {records.map(record => (
                <tr key={record.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800">
                    {record.reference_number}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {record.patient?.name || "N/A"}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      record.analysis_result === 'PNEUMONIA' 
                        ? 'bg-red-100 text-red-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {record.analysis_result}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {record.pneumonia_type || "N/A"}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {record.confidence_score ? `${(record.confidence_score).toFixed(1)}%` : "N/A"}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {new Date(record.createdAt).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button
                      onClick={() => handleEditRecord(record)}
                      className="text-indigo-600 hover:text-indigo-900 mr-4"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => handleDeleteRecord(record.id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {records.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-6 py-4 text-center text-sm text-gray-500">
                    No records found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Edit Record Modal */}
      {isModalOpen && currentRecord && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen p-4">
            <div 
              className="fixed inset-0 bg-black/30 backdrop-blur-sm transition-opacity"
              onClick={() => setIsModalOpen(false)}
            ></div>
            
            <div className="relative z-10 bg-white rounded-lg shadow-xl max-w-md w-full mx-4 overflow-hidden">
              <div className="p-6">
                <h3 className="text-xl font-bold text-gray-800 mb-4">
                  Edit Pneumonia Record
                </h3>
                
                <form onSubmit={handleSubmit}>
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Reference Number
                    </label>
                    <input
                      type="text"
                      value={currentRecord.reference_number}
                      onChange={(e) => setCurrentRecord({ ...currentRecord, reference_number: e.target.value })}
                      className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      required
                    />
                  </div>
                  
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Pneumonia Type
                    </label>
                    <select
                      value={currentRecord.pneumonia_type}
                      onChange={(e) => setCurrentRecord({ ...currentRecord, pneumonia_type: e.target.value })}
                      className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="">None</option>
                      <option value="BACTERIAL">Bacterial</option>
                      <option value="VIRAL">Viral</option>
                      <option value="FUNGAL">Fungal</option>
                    </select>
                  </div>
                  
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Severity
                    </label>
                    <select
                      value={currentRecord.severity}
                      onChange={(e) => setCurrentRecord({ ...currentRecord, severity: e.target.value })}
                      className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="">None</option>
                      <option value="MILD">Mild</option>
                      <option value="MODERATE">Moderate</option>
                      <option value="SEVERE">Severe</option>
                    </select>
                  </div>
                  
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Confidence Score (%)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="0.1"
                      value={currentRecord.confidence_score * 100}
                      onChange={(e) => setCurrentRecord({ 
                        ...currentRecord, 
                        confidence_score: parseFloat(e.target.value) / 100 
                      })}
                      className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      required
                    />
                  </div>
                  
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Recommended Action
                    </label>
                    <textarea
                      value={currentRecord.recommended_action}
                      onChange={(e) => setCurrentRecord({ ...currentRecord, recommended_action: e.target.value })}
                      className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      rows={3}
                    />
                  </div>
                  
                  <div className="flex justify-end space-x-3 mt-6">
                    <button
                      type="button"
                      onClick={() => setIsModalOpen(false)}
                      className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 bg-white hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                    >
                      Save Changes
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 