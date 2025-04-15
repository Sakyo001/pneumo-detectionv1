"use client";

import React, { useState, useEffect } from 'react';
import { 
  Document, 
  Page, 
  Text, 
  View, 
  StyleSheet, 
  PDFDownloadLink 
} from '@react-pdf/renderer';

// Define the ScanData interface
interface ScanData {
  id: string;
  patientName: string;
  date: string;
  result: string;
  confidence: number;
  [key: string]: any;
}

interface PDFExportProps {
  scans: ScanData[];
  dashboardData: {
    totalScans: number;
    pneumoniaCases: number;
    normalCases: number;
    todayScans: number;
    recentScans: any[];
  };
}

// Create styles for PDF
const styles = StyleSheet.create({
  page: {
    flexDirection: 'column',
    backgroundColor: '#FFFFFF',
    padding: 30,
  },
  header: {
    fontSize: 24,
    marginBottom: 20,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  subheader: {
    fontSize: 16,
    marginBottom: 10,
  },
  table: {
    display: 'flex',
    flexDirection: 'column',
    width: 'auto',
    borderStyle: 'solid',
    borderWidth: 1,
    borderColor: '#bfbfbf',
    marginBottom: 20,
  },
  tableRow: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#bfbfbf',
    borderBottomStyle: 'solid',
    alignItems: 'center',
    minHeight: 30,
  },
  tableHeader: {
    backgroundColor: '#f0f0f0',
    fontWeight: 'bold',
  },
  tableCell: {
    padding: 6,
    fontSize: 10,
    textAlign: 'left',
  },
  patientName: {
    width: '25%',
  },
  date: {
    width: '25%',
  },
  result: {
    width: '25%',
  },
  confidence: {
    width: '25%',
  },
  footer: {
    marginTop: 20,
    fontSize: 10,
    textAlign: 'center',
  },
});

// PDF Document Component
const ScansPDF = ({ scans, dashboardData }: PDFExportProps) => (
  <Document>
    <Page size="A4" style={styles.page}>
      <Text style={styles.header}>X-Ray Scan Reports</Text>
      <Text style={styles.subheader}>Generated on {new Date().toLocaleDateString()}</Text>
      
      <View style={{ marginBottom: 20 }}>
        <Text style={{ fontSize: 12, marginBottom: 5 }}>Total Scans: {dashboardData.totalScans}</Text>
        <Text style={{ fontSize: 12, marginBottom: 5 }}>Pneumonia Cases: {dashboardData.pneumoniaCases}</Text>
        <Text style={{ fontSize: 12, marginBottom: 5 }}>Normal Cases: {dashboardData.normalCases}</Text>
      </View>
      
      <Text style={styles.subheader}>Scan Details</Text>
      <View style={styles.table}>
        <View style={[styles.tableRow, styles.tableHeader]}>
          <Text style={[styles.tableCell, styles.patientName]}>Patient Name</Text>
          <Text style={[styles.tableCell, styles.date]}>Date</Text>
          <Text style={[styles.tableCell, styles.result]}>Result</Text>
          <Text style={[styles.tableCell, styles.confidence]}>Confidence</Text>
        </View>
        {scans.map((scan, index) => (
          <View key={index} style={styles.tableRow}>
            <Text style={[styles.tableCell, styles.patientName]}>{scan.patientName || ''}</Text>
            <Text style={[styles.tableCell, styles.date]}>{scan.date || ''}</Text>
            <Text style={[styles.tableCell, styles.result]}>{scan.result || ''}</Text>
            <Text style={[styles.tableCell, styles.confidence]}>{scan.confidence > 100 ? (scan.confidence / 100).toFixed(2) : (scan.confidence || 0)}%</Text>
          </View>
        ))}
      </View>
      
      <Text style={styles.footer}>
        © 2023 MedRecord Hub. All rights reserved.
      </Text>
    </Page>
  </Document>
);

// Main PDF Export Component
const PDFExport: React.FC<PDFExportProps> = ({ scans, dashboardData }) => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <button className="inline-flex items-center px-4 py-2 bg-gray-400 text-white font-medium rounded-md">
        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        Loading PDF Export...
      </button>
    );
  }
  
  try {
    return (
      <PDFDownloadLink 
        document={<ScansPDF scans={scans} dashboardData={dashboardData} />} 
        fileName="xray-scans-report.pdf"
        className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
      >
        {({ loading, error }) => {
          if (error) {
            console.error("PDF generation error:", error);
            return (
              <>
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                PDF Error
              </>
            );
          }
          
          return (
            <>
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              {loading ? 'Generating PDF...' : 'Export to PDF'}
            </>
          );
        }}
      </PDFDownloadLink>
    );
  } catch (err) {
    console.error("PDF render error:", err);
    return (
      <button className="inline-flex items-center px-4 py-2 bg-red-600 text-white font-medium rounded-md">
        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        PDF Error
      </button>
    );
  }
};

export default PDFExport; 