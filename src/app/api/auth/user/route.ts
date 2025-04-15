import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";

/**
 * Fast, lightweight endpoint to get current user info
 * Used by the upload-xray page to quickly display doctor name
 */
export async function GET(req: NextRequest) {
  try {
    // Get user ID from cookie
    const cookieStore = cookies();
    const userId = cookieStore.get("userId");
    const userRole = cookieStore.get("userRole");
    const userName = cookieStore.get("userName");

    if (!userId || !userRole) {
      return NextResponse.json(
        { 
          success: false, 
          message: "User not authenticated" 
        }, 
        { status: 401 }
      );
    }

    // Return basic user info from cookies
    // This is optimized to be very lightweight compared to a database query
    return NextResponse.json({
      success: true,
      user: {
        id: userId.value,
        role: userRole.value,
        name: userName?.value || "Doctor" // Default if not in cookie
      }
    });
  } catch (error) {
    console.error("Error retrieving user data:", error);
    return NextResponse.json(
      { 
        success: false, 
        message: "Failed to retrieve user data" 
      }, 
      { status: 500 }
    );
  }
} 