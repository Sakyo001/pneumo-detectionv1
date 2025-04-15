import { NextRequest, NextResponse } from 'next/server';

// Simple endpoint to check if PDF generation is available
export async function HEAD(request: NextRequest) {
  try {
    // Check if SWC helpers module is available with the right exports
    if (typeof window === 'undefined') { // Only on server-side
      try {
        // Try to dynamically import without actually loading the PDF generator
        await import('@swc/helpers');
        return new NextResponse(null, { status: 200 });
      } catch (err) {
        console.error('SWC helpers check failed:', err);
        return new NextResponse(null, { status: 503 }); // Service Unavailable
      }
    }
    
    // Return success for client-side
    return new NextResponse(null, { status: 200 });
  } catch (error) {
    console.error('Error in PDF availability check:', error);
    return new NextResponse(null, { status: 503 }); // Service Unavailable
  }
}

// For GET requests - provide more detailed info
export async function GET(request: NextRequest) {
  try {
    // Check if PDF generation is available
    let pdfAvailable = true;
    let errorMessage = null;
    
    // Check SWC helpers availability
    if (typeof window === 'undefined') { // Only on server-side
      try {
        // Check if SWC helpers are available
        const swcHelpers = require('@swc/helpers');
        
        // Check if the module has the required export or can be patched
        if (!swcHelpers.applyDecoratedDescriptor && !swcHelpers._apply_decorated_descriptor) {
          pdfAvailable = false;
          errorMessage = 'Missing required SWC helper exports';
        }
      } catch (err) {
        pdfAvailable = false;
        errorMessage = err instanceof Error ? err.message : 'Unknown SWC helpers error';
      }
    }
    
    return NextResponse.json({
      pdfAvailable,
      errorMessage
    }, { status: pdfAvailable ? 200 : 503 });
  } catch (error) {
    return NextResponse.json({
      pdfAvailable: false,
      errorMessage: error instanceof Error ? error.message : 'Unknown error checking PDF availability'
    }, { status: 503 });
  }
} 