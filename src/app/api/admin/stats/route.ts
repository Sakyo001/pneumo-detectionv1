import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/lib/auth";
import prisma from "@/lib/prisma";
import { serializeForJson } from "@/lib/utils";
import { executeRawQuery, getSystemStats } from "@/lib/db-utils";

// Maximum number of retries for database operations
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second
const TRANSACTION_TIMEOUT = 10000; // 10 seconds

// Helper function to wait between retries
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Define types for our raw query results
interface ScanByMonth {
  month: Date;
  scans: bigint;
  pneumonia: bigint;
}

interface PneumoniaDistribution {
  name: string;
  value: bigint;
}

interface DoctorActivity {
  name: string;
  scans: bigint;
  accuracy: number | null;
}

// Helper function to execute a database operation with retries
async function executeWithRetry<T>(operation: () => Promise<T>): Promise<T> {
  let lastError;
  
  for (let i = 0; i < MAX_RETRIES; i++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      console.error(`Attempt ${i + 1} failed:`, error);
      
      if (i < MAX_RETRIES - 1) {
        await wait(RETRY_DELAY);
      }
    }
  }
  
  throw lastError;
}

export async function GET(request: NextRequest) {
  try {
    const user = await requireAuth('ADMIN');
    
    // If not admin, return unauthorized
    if (!user || user.role !== 'ADMIN') {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    // Execute queries in parallel without transaction for better performance
    const [basicStats, scansByMonth, pneumoniaDistribution, doctorActivity] = await Promise.all([
      // Get basic stats
      executeWithRetry(() => getSystemStats()),
      
      // Get scans over time (last 6 months)
      (async () => {
        const sixMonthsAgo = new Date();
        sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);
        
        return executeWithRetry(() => 
          executeRawQuery<ScanByMonth>(`
            SELECT 
              DATE_TRUNC('month', "createdAt") as month,
              COUNT(*) as scans,
              COUNT(CASE WHEN UPPER(result) = 'PNEUMONIA' THEN 1 END) as pneumonia
            FROM "XrayScan"
            WHERE "createdAt" >= $1
            GROUP BY DATE_TRUNC('month', "createdAt")
            ORDER BY month ASC
          `, sixMonthsAgo)
        );
      })(),
      
      // Get pneumonia distribution
      executeWithRetry(() =>
        executeRawQuery<PneumoniaDistribution>(`
          SELECT 
            CASE 
              WHEN UPPER(result) = 'PNEUMONIA' THEN 'Pneumonia'
              ELSE 'Normal'
            END as name,
            COUNT(*) as value
          FROM "XrayScan"
          GROUP BY 
            CASE 
              WHEN UPPER(result) = 'PNEUMONIA' THEN 'Pneumonia'
              ELSE 'Normal'
            END
        `)
      ),
      
      // Get doctor activity
      executeWithRetry(() =>
        executeRawQuery<DoctorActivity>(`
          SELECT 
            u.name,
            COUNT(x.id) as scans,
            LEAST(AVG(m.confidence), 100) as accuracy
          FROM "User" u
          LEFT JOIN "XrayScan" x ON u.id = x."doctorId"
          LEFT JOIN "ScanMetadata" m ON x.id = m."scanId"
          WHERE u.role = 'DOCTOR'
          GROUP BY u.id, u.name
          ORDER BY scans DESC
          LIMIT 5
        `)
      )
    ]);
    
    // Calculate average confidence and cap at 100%
    const avgConfidence = basicStats.avgConfidence ? 
      Math.min(Number(basicStats.avgConfidence), 100) : 0;

    // Format the response
    const responseData = {
      stats: {
        totalScans: Number(basicStats.totalScans) || 0,
        pneumoniaCases: Number(basicStats.pneumoniaCases) || 0,
        activeDoctors: Number(basicStats.activeDoctors) || 0,
        avgConfidence: avgConfidence
      },
      scanData: scansByMonth,
      pneumoniaData: pneumoniaDistribution,
      doctorData: doctorActivity,
    };
    
    // Log the response data for debugging
    console.log('API Response Data:', JSON.stringify({
      stats: {
        totalScans: Number(basicStats.totalScans) || 0,
        pneumoniaCases: Number(basicStats.pneumoniaCases) || 0,
        activeDoctors: Number(basicStats.activeDoctors) || 0,
        avgConfidence: avgConfidence
      },
      scanDataLength: scansByMonth.length,
      pneumoniaDataLength: pneumoniaDistribution.length,
      doctorDataLength: doctorActivity.length
    }, null, 2));
    
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching statistics:", error);
    
    // Handle transaction timeout specifically
    if (error instanceof Error && error.message.includes('Transaction already closed')) {
      return NextResponse.json(
        { 
          error: "Database operation timed out", 
          details: "The operation took too long to complete. Please try again.",
          timestamp: new Date().toISOString() 
        },
        { status: 504 } // Gateway Timeout
      );
    }
    
    return NextResponse.json(
      { 
        error: "Failed to fetch statistics", 
        details: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString() 
      },
      { status: 500 }
    );
  }
} 