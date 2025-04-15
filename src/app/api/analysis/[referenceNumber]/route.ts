import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { Prisma } from '@prisma/client';

export async function GET(
  request: NextRequest,
  { params }: { params: { referenceNumber: string } }
) {
  try {
    console.log('Received request for reference number:', params.referenceNumber);
    
    const referenceNumber = params.referenceNumber;
    
    if (!referenceNumber) {
      console.log('No reference number provided');
      return NextResponse.json(
        { error: 'Reference number is required' },
        { status: 400 }
      );
    }

    // Test database connection
    try {
      await prisma.$connect();
      console.log('Database connection successful');
    } catch (dbError) {
      console.error('Database connection failed:', dbError);
      return NextResponse.json(
        { error: 'Database connection failed' },
        { status: 500 }
      );
    }

    // Fetch the X-ray scan data from the database
    console.log('Fetching X-ray scan data from database...');
    let xrayScan;
    try {
      xrayScan = await prisma.xrayScan.findFirst({
        where: {
          referenceNumber: referenceNumber
        },
        include: {
          metadata: true,
          patient: {
            include: {
              doctor: true
            }
          }
        }
      });
    } catch (queryError) {
      console.error('Database query failed:', queryError);
      if (queryError instanceof Prisma.PrismaClientKnownRequestError) {
        return NextResponse.json(
          { error: `Database query failed: ${queryError.message}` },
          { status: 500 }
        );
      }
      return NextResponse.json(
        { error: 'Failed to fetch data from database' },
        { status: 500 }
      );
    }

    console.log('Database query result:', xrayScan);

    if (!xrayScan) {
      console.log('No X-ray scan found for reference number:', referenceNumber);
      return NextResponse.json(
        { error: 'No analysis found for this reference number' },
        { status: 404 }
      );
    }

    // Format the data for the client
    const result = {
      id: xrayScan.id,
      reference_number: xrayScan.referenceNumber,
      image_url: xrayScan.imageUrl,
      analysis_result: xrayScan.result,
      confidence_score: xrayScan.metadata?.confidence || 0,
      created_at: xrayScan.createdAt.toISOString(),
      doctor_name: xrayScan.patient?.doctor?.name || 'Unknown Doctor',
      pneumonia_type: xrayScan.metadata?.pneumoniaType || null,
      severity: xrayScan.metadata?.severity || null,
      recommended_action: xrayScan.metadata?.recommendedAction || null,
      patient_name: xrayScan.patient?.name || 'Unknown Patient'
    };

    console.log('Sending response:', result);
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error in GET /api/analysis/[referenceNumber]:', error);
    return NextResponse.json(
      { error: 'Failed to fetch analysis results', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  } finally {
    try {
      await prisma.$disconnect();
    } catch (disconnectError) {
      console.error('Error disconnecting from database:', disconnectError);
    }
  }
} 