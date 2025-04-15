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

    // Get total scans count
    const totalScans = await prisma.xrayScan.count({
      where: {
        patient: {
          doctorId: userId
        }
      }
    });

    // Get pneumonia cases count
    const pneumoniaCases = await prisma.xrayScan.count({
      where: {
        patient: {
          doctorId: userId
        },
        OR: [
          { result: "PNEUMONIA" },
          { result: "Pneumonia" },
          { result: "pneumonia" }
        ]
      }
    });

    // Calculate normal cases
    const normalCases = totalScans - pneumoniaCases;

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

    // Get recent scans with patient info
    const recentScans = await prisma.xrayScan.findMany({
      where: {
        patient: {
          doctorId: userId
        }
      },
      select: {
        id: true,
        referenceNumber: true,
        result: true,
        createdAt: true,
        patient: {
          select: {
            name: true
          }
        },
        metadata: {
          select: {
            confidence: true
          }
        }
      },
      orderBy: {
        createdAt: 'desc'
      },
      take: 5
    });

    // Format recent scans for the frontend
    const formattedRecentScans = recentScans.map(scan => ({
      id: scan.id,
      patientName: scan.patient.name,
      date: scan.createdAt.toISOString(),
      result: scan.result || 'Unknown',
      confidence: scan.metadata?.confidence ? Math.round(scan.metadata.confidence * 100) : 0
    }));

    return NextResponse.json({
      success: true,
      data: {
        totalScans,
        pneumoniaCases,
        normalCases,
        todayScans,
        recentScans: formattedRecentScans
      }
    });
  } catch (error) {
    console.error("Error fetching statistics:", error);
    return NextResponse.json(
      { 
        success: false, 
        message: "Failed to fetch statistics" 
      }, 
      { status: 500 }
    );
  }
} 