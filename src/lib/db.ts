import { PrismaClient } from '@prisma/client';
import prisma from './prisma';
import { safeNumberConversion, serializeForJson } from './utils';

// Define a type for global variables
declare global {
  var prisma: PrismaClient | undefined;
}

// Define the appropriate log level type
type PrismaLogLevel = 'query' | 'info' | 'warn' | 'error';

// Create a singleton to avoid multiple connections in dev
const prismaSingleton = global.prisma || new PrismaClient();

// Add connection retry logic
const connectWithRetry = async (retries = 5, delay = 1000) => {
  let currentAttempt = 0;
  
  while (currentAttempt < retries) {
    try {
      await prismaSingleton.$connect();
      console.log('Successfully connected to the database');
      return;
    } catch (error) {
      currentAttempt++;
      console.error(`Failed to connect to the database (attempt ${currentAttempt}/${retries}):`, error);
      
      if (currentAttempt >= retries) {
        console.error('Maximum connection attempts reached. Please check your database configuration.');
      } else {
        console.log(`Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        // Exponential backoff
        delay *= 2;
      }
    }
  }
};

// Initialize connection in development
if (process.env.NODE_ENV !== 'production') {
  // Keep a global reference in development to prevent multiple connections
  global.prisma = prismaSingleton;
  
  // Connect to the database
  connectWithRetry();
} else {
  // In production, connect immediately but don't worry about global reference
  connectWithRetry();
}

// Graceful shutdown
process.on('beforeExit', async () => {
  await prismaSingleton.$disconnect();
});

// Database utility functions for common queries and operations
export {
  prisma,
  serializeForJson,
  safeNumberConversion
};

/**
 * Get basic statistics about the system
 * Returns counts of scans, pneumonia cases, and doctors
 */
export async function getSystemStats() {
  try {
    // Get total scans
    const totalScans = await prisma.xrayScan.count();
    
    // Get pneumonia cases
    const pneumoniaCases = await prisma.xrayScan.count({
      where: {
        result: "PNEUMONIA",
      },
    });
    
    // Get active doctors
    const activeDoctors = await prisma.user.count({
      where: {
        role: "DOCTOR",
      },
    });
    
    // Get average confidence score
    const avgConfidenceResult = await prisma.scanMetadata.aggregate({
      _avg: {
        confidence: true,
      },
    });
    
    return {
      totalScans,
      pneumoniaCases,
      activeDoctors,
      avgConfidence: avgConfidenceResult._avg?.confidence || 0
    };
  } catch (error) {
    console.error("Error fetching system stats:", error);
    throw error;
  }
}

/**
 * Get a doctor's recent scans with patient information
 */
export async function getDoctorScans(doctorId: string, limit = 10) {
  try {
    const scans = await prisma.xrayScan.findMany({
      where: {
        doctorId,
      },
      include: {
        patient: true,
        metadata: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
      take: limit,
    });
    
    return serializeForJson(scans);
  } catch (error) {
    console.error(`Error fetching scans for doctor ${doctorId}:`, error);
    throw error;
  }
}

/**
 * Safely execute a raw SQL query and handle BigInt serialization
 */
export async function executeRawQuery<T>(query: string, ...parameters: any[]): Promise<T[]> {
  try {
    const results = await prisma.$queryRawUnsafe<T[]>(query, ...parameters);
    return serializeForJson(results);
  } catch (error) {
    console.error("Error executing raw query:", error);
    throw error;
  }
} 