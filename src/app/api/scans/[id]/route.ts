import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { cookies } from "next/headers";

export async function GET(req: NextRequest, { params }: { params: { id: string } }) {
  try {
    // Ensure params is properly awaited
    const { id: scanId } = params;
    
    if (!scanId) {
      return NextResponse.json(
        { 
          success: false, 
          message: "Scan ID is required" 
        }, 
        { status: 400 }
      );
    }
    
    // Get doctor ID from cookie for authorization
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

    // Fetch scan with related data
    const scan = await prisma.xrayScan.findUnique({
      where: {
        id: scanId,
      },
      include: {
        patient: {
          select: {
            name: true,
            doctorId: true
          }
        },
        metadata: true
      }
    });
    
    // Check if scan exists
    if (!scan) {
      return NextResponse.json(
        { 
          success: false, 
          message: "Scan not found" 
        }, 
        { status: 404 }
      );
    }
    
    // Ensure the doctor has access to this scan
    if (scan.patient.doctorId !== userId) {
      return NextResponse.json(
        { 
          success: false, 
          message: "You don't have permission to view this scan" 
        }, 
        { status: 403 }
      );
    }
    
    // Format the scan data for the response
    const formattedScan = {
      id: scan.id,
      patientName: scan.patient.name,
      date: scan.createdAt.toISOString(),
      result: scan.result,
      confidence: scan.metadata?.confidence ? Math.round(scan.metadata.confidence * 100) : null,
      pneumoniaType: scan.metadata?.pneumoniaType || null,
      severity: scan.metadata?.severity || null,
      recommendedAction: scan.metadata?.recommendedAction || null,
      referenceNumber: scan.referenceNumber,
      imageUrl: scan.imageUrl,
      status: scan.status
    };
    
    return NextResponse.json({
      success: true,
      data: formattedScan
    });
  } catch (error) {
    console.error("Error fetching scan details:", error);
    return NextResponse.json(
      { 
        success: false, 
        message: "Failed to fetch scan details" 
      }, 
      { status: 500 }
    );
  }
} 