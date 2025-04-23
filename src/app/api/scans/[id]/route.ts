import { cookies } from "next/headers";
import { prisma } from "@/lib/db";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    // Await the params to resolve the promise
    const resolvedParams = await params;
    const { id: scanId } = resolvedParams;
    
    if (!scanId) {
      return Response.json(
        { 
          error: "Scan ID is required"
        },
        { status: 400 }
      );
    }
    
    // In Next.js 15, cookies() returns a Promise that needs to be awaited
    const cookiesList = await cookies();
    const userId = cookiesList.get("userId")?.value;
    
    if (!userId) {
      return Response.json(
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
      return Response.json(
        { 
          success: false, 
          message: "Scan not found" 
        }, 
        { status: 404 }
      );
    }
    
    // Ensure the doctor has access to this scan
    if (scan.patient.doctorId !== userId) {
      return Response.json(
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
    
    return Response.json({
      success: true,
      data: formattedScan
    });
  } catch (error) {
    console.error("Error fetching scan details:", error);
    return Response.json(
      { 
        success: false, 
        message: "Failed to fetch scan details" 
      }, 
      { status: 500 }
    );
  }
} 