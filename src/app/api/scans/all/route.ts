import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { cookies } from "next/headers";

export async function GET(req: NextRequest) {
  try {
    // Get doctor ID from cookie
    const cookieStore = await cookies();
    const userId = cookieStore.get("userId")?.value;
    
    if (!userId) {
      return NextResponse.json(
        { 
          success: false, 
          message: "Not authenticated" 
        }, 
        { status: 401 }
      );
    }

    // Get query parameters for filtering and pagination
    const { searchParams } = new URL(req.url);
    const searchTerm = searchParams.get('search') || '';
    const page = parseInt(searchParams.get('page') || '1');
    const limit = parseInt(searchParams.get('limit') || '10');
    const sortBy = searchParams.get('sortBy') || 'createdAt';
    const sortOrder = searchParams.get('sortOrder') || 'desc';
    
    // Compute pagination values
    const skip = (page - 1) * limit;
    
    // Prepare filter conditions
    const whereCondition: any = {
      patient: {
        doctorId: userId
      }
    };
    
    if (searchTerm) {
      whereCondition.OR = [
        {
          patient: {
            name: {
              contains: searchTerm,
              mode: 'insensitive'
            }
          }
        },
        {
          result: {
            contains: searchTerm,
            mode: 'insensitive'
          }
        },
        {
          referenceNumber: {
            contains: searchTerm,
            mode: 'insensitive'
          }
        }
      ];
    }
    
    // Get total matching records for pagination
    const totalScans = await prisma.xrayScan.count({
      where: whereCondition
    });
    
    // Get paginated and sorted records
    const scans = await prisma.xrayScan.findMany({
      where: whereCondition,
      select: {
        id: true,
        referenceNumber: true,
        result: true,
        createdAt: true,
        imageUrl: true,
        status: true,
        patient: {
          select: {
            name: true
          }
        },
        metadata: {
          select: {
            confidence: true,
            pneumoniaType: true,
            severity: true,
            recommendedAction: true
          }
        }
      },
      orderBy: {
        [sortBy]: sortOrder === 'asc' ? 'asc' : 'desc'
      },
      skip,
      take: limit
    });
    
    // Get stats for dashboard overview
    const pneumoniaCases = await prisma.xrayScan.count({
      where: {
        patient: {
          doctorId: userId
        },
        result: "Pneumonia"
      }
    });
    
    const totalScanCount = await prisma.xrayScan.count({
      where: {
        patient: {
          doctorId: userId
        }
      }
    });
    
    const normalCases = totalScanCount - pneumoniaCases;
    
    // Get today's scans
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const todayScans = await prisma.xrayScan.count({
      where: {
        patient: {
          doctorId: userId
        },
        createdAt: {
          gte: today
        }
      }
    });
    
    // Format scans for response
    const formattedScans = scans.map(scan => ({
      id: scan.id,
      patientName: scan.patient.name,
      date: scan.createdAt.toISOString(),
      result: scan.result || 'Unknown',
      confidence: scan.metadata?.confidence ? Math.round(scan.metadata.confidence * 100) : 0,
      pneumoniaType: scan.metadata?.pneumoniaType || null,
      severity: scan.metadata?.severity || null,
      referenceNumber: scan.referenceNumber,
      imageUrl: scan.imageUrl,
      status: scan.status
    }));
    
    return NextResponse.json({
      success: true,
      data: {
        scans: formattedScans,
        totalScans: totalScanCount,
        pneumoniaCases,
        normalCases,
        todayScans,
        pagination: {
          total: totalScans,
          page,
          limit,
          totalPages: Math.ceil(totalScans / limit)
        }
      }
    });
  } catch (error) {
    console.error("Error fetching scans:", error);
    return NextResponse.json(
      { 
        success: false, 
        message: "Failed to fetch scans" 
      }, 
      { status: 500 }
    );
  }
} 