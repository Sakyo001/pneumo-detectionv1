import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export async function GET() {
  try {
    // Test database connection
    await prisma.$queryRaw`SELECT 1 as test`;
    
    // List users in database
    const users = await prisma.user.findMany({
      select: {
        id: true,
        email: true,
        doctorId: true,
        role: true,
        name: true,
        // Don't include passwords in the response
      }
    });
    
    return NextResponse.json({ 
      success: true, 
      message: "Database connection successful",
      users,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json({ 
      success: false, 
      message: "Database error",
      error: error.message
    }, { status: 500 });
  }
} 