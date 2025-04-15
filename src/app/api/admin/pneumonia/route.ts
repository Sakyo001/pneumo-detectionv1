import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/lib/auth";
import prisma from "@/lib/prisma";
import { serializeForJson } from "@/lib/utils";
import { User, ScanMetadata, Patient, XrayScan } from "@prisma/client";

// Define the combined XrayScan with its relations for proper typing
interface XrayScanWithRelations extends XrayScan {
  patient: Patient;
  metadata: ScanMetadata | null;
}

// Response data interface
interface PneumoniaResponseData {
  id: string;
  reference_number: string;
  image_url: string;
  analysis_result: string;
  pneumonia_type: string | null;
  severity: string | null;
  confidence_score: number;
  recommended_action: string | null;
  createdAt: Date;
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
    
    // Get all doctors for lookup
    const doctors = await prisma.user.findMany({
      where: {
        role: "DOCTOR"
      },
          select: {
            id: true,
            name: true,
            email: true,
      }
    });
    
    // Create a map of doctors by ID for easy lookup
    const doctorsMap: Record<string, { id: string; name: string; email: string }> = {};
    doctors.forEach(doctor => {
      doctorsMap[doctor.id] = doctor;
    });
    
    // Get pneumonia data with patient and metadata information
    const scans = await prisma.xrayScan.findMany({
      include: {
        patient: true,
        metadata: true
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
    
    // Format the response to match expected structure
    const pneumoniaData: PneumoniaResponseData[] = scans.map(scan => {
      // Get the doctor from our map or use a default
      const doctor = scan.doctorId && doctorsMap[scan.doctorId] 
        ? doctorsMap[scan.doctorId] 
        : {
            id: scan.doctorId || "unknown",
            name: 'Unknown Doctor',
            email: 'unknown@example.com'
          };
      
      return {
        id: scan.id,
        reference_number: scan.referenceNumber,
        image_url: scan.imageUrl,
        analysis_result: scan.result || "UNKNOWN",
        pneumonia_type: scan.metadata?.pneumoniaType || null,
        severity: scan.metadata?.severity || null,
        confidence_score: scan.metadata?.confidence || 0,
        recommended_action: scan.metadata?.recommendedAction || null,
        createdAt: scan.createdAt,
        patient: {
          id: scan.patient.id,
          name: scan.patient.name,
          referenceNumber: scan.patient.referenceNumber,
        },
        doctor: doctor,
      };
    });
    
    return NextResponse.json(serializeForJson(pneumoniaData));
  } catch (error) {
    console.error("Error fetching pneumonia data:", error);
    return NextResponse.json(
      { error: "Failed to fetch pneumonia data", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

// Interface for incoming update data
interface PneumoniaUpdateData {
  id: string;
  reference_number: string;
  analysis_result: string;
  pneumonia_type?: string | null;
  severity?: string | null;
  confidence_score?: number;
  recommended_action?: string | null;
  patientId?: string;
  doctorId?: string;
}

export async function PUT(request: NextRequest) {
  try {
    const user = await requireAuth('ADMIN');
    
    // If not admin, return unauthorized
    if (!user || user.role !== 'ADMIN') {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    const data: PneumoniaUpdateData = await request.json();
    
    // Update pneumonia record and metadata in a transaction
    const result = await prisma.$transaction(async (tx) => {
      // Update the scan record
      const updateData: any = {
        referenceNumber: data.reference_number,
        result: data.analysis_result
      };
      
      // Only include these fields if they are provided and valid
      if (data.patientId) {
        updateData.patient = {
          connect: { id: data.patientId }
        };
      }
      
      if (data.doctorId) {
        updateData.doctor = {
          connect: { id: data.doctorId }
        };
      }
      
      const updatedScan = await tx.xrayScan.update({
        where: { id: data.id },
        data: updateData,
        include: {
          patient: true,
          metadata: true
        }
      });
      
      // Get doctor information
      const doctor = await tx.user.findUnique({
        where: { id: updatedScan.doctorId },
        select: {
          id: true,
          name: true,
          email: true,
        }
      });
      
      // Update or create metadata
      let updatedMetadata;
      if (updatedScan.metadata) {
        // Update existing metadata
        updatedMetadata = await tx.scanMetadata.update({
          where: { scanId: updatedScan.id },
          data: {
            pneumoniaType: data.pneumonia_type,
            severity: data.severity,
            confidence: data.confidence_score || 0,
            recommendedAction: data.recommended_action
          }
        });
      } else {
        // Create new metadata
        updatedMetadata = await tx.scanMetadata.create({
          data: {
            scanId: updatedScan.id,
            pneumoniaType: data.pneumonia_type,
            severity: data.severity,
            confidence: data.confidence_score || 0,
            recommendedAction: data.recommended_action
          }
        });
      }
      
      return { 
        scan: updatedScan, 
        metadata: updatedMetadata,
        doctor
      };
    });
    
    // Format the response
    const formattedRecord: PneumoniaResponseData = {
      id: result.scan.id,
      reference_number: result.scan.referenceNumber,
      image_url: result.scan.imageUrl,
      analysis_result: result.scan.result || "UNKNOWN",
      pneumonia_type: result.metadata?.pneumoniaType || null,
      severity: result.metadata?.severity || null,
      confidence_score: result.metadata?.confidence || 0,
      recommended_action: result.metadata?.recommendedAction || null,
      createdAt: result.scan.createdAt,
      patient: {
        id: result.scan.patient.id,
        name: result.scan.patient.name,
        referenceNumber: result.scan.patient.referenceNumber,
      },
      doctor: result.doctor || {
        id: result.scan.doctorId || "unknown",
        name: 'Unknown Doctor',
        email: 'unknown@example.com'
      },
    };
    
    return NextResponse.json(serializeForJson(formattedRecord));
  } catch (error) {
    console.error("Error updating pneumonia record:", error);
    return NextResponse.json(
      { error: "Failed to update pneumonia record", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const user = await requireAuth('ADMIN');
    
    // If not admin, return unauthorized
    if (!user || user.role !== 'ADMIN') {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    const url = new URL(request.url);
    const id = url.searchParams.get('id');
    
    if (!id) {
      return NextResponse.json(
        { error: "Record ID is required" },
        { status: 400 }
      );
    }
    
    // Delete in a transaction to ensure atomicity
    await prisma.$transaction(async (tx) => {
      // Delete scan metadata first (if it exists)
      await tx.scanMetadata.deleteMany({
        where: { scanId: id }
      });
      
      // Then delete the scan record
      await tx.xrayScan.delete({
        where: { id }
      });
    });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting pneumonia record:", error);
    return NextResponse.json(
      { error: "Failed to delete pneumonia record", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
} 