import prisma from './prisma';
import { safeNumberConversion, serializeForJson } from './utils';

/**
 * Database utility functions for common queries and operations
 * with proper type handling for BigInt and serialization
 */
export {
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
    console.log('Total scans:', totalScans);
    
    // Check what values exist in the result column
    const resultValues = await prisma.$queryRaw`
      SELECT DISTINCT result FROM "XrayScan"
    `;
    console.log('Possible result values in database:', resultValues);
    
    // Get pneumonia cases - checking for different case variations
    const pneumoniaCases = await prisma.xrayScan.count({
      where: {
        OR: [
          { result: "PNEUMONIA" },
          { result: "Pneumonia" },
          { result: "pneumonia" }
        ]
      },
    });
    console.log('Pneumonia cases (case insensitive):', pneumoniaCases);
    
    // Log the raw query result for verification with case insensitivity
    const rawPneumoniaCases = await prisma.$queryRaw`
      SELECT COUNT(*) as count 
      FROM "XrayScan" 
      WHERE UPPER(result) = 'PNEUMONIA'
    `;
    console.log('Raw pneumonia query result:', rawPneumoniaCases);
    
    // Get active doctors
    const activeDoctors = await prisma.user.count({
      where: {
        role: "DOCTOR",
      },
    });
    console.log('Active doctors:', activeDoctors);
    
    // Get average confidence score
    const avgConfidenceResult = await prisma.scanMetadata.aggregate({
      _avg: {
        confidence: true,
      },
    });
    
    // Cap confidence at 100% and ensure it's a number
    const avgConfidence = Math.min(
      avgConfidenceResult._avg?.confidence || 0,
      100
    );
    
    const stats = {
      totalScans,
      pneumoniaCases,
      activeDoctors,
      avgConfidence
    };
    
    console.log('Final stats:', stats);
    return stats;
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
  }
  catch (error) {
    console.error("Error executing raw query:", error);
    throw error;
  }
} 