import { NextResponse } from "next/server";
import prisma from "@/lib/prisma";
import { getUserFromCookie } from "@/lib/auth";

export async function GET() {
  try {
    // Get the current doctor from the session
    const user = await getUserFromCookie();
    
    if (!user || user.role !== 'DOCTOR') {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    // Fetch patients assigned to this doctor
    const patients = await prisma.patient.findMany({
      where: {
        doctorId: user.id
      },
      select: {
        id: true,
        name: true,
        referenceNumber: true,
        dateOfBirth: true
      },
      orderBy: {
        name: 'asc'
      }
    });
    
    return NextResponse.json({ patients });
  } catch (error) {
    console.error("Error fetching patients:", error);
    return NextResponse.json(
      { error: "Failed to fetch patients" },
      { status: 500 }
    );
  }
} 