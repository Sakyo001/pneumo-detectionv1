import { NextResponse } from "next/server";
import { cookies } from "next/headers";

export async function POST() {
  try {
    // Get the cookie store
    const cookieStore = await cookies();
    
    // Clear the authentication cookies
    cookieStore.delete("userId");
    cookieStore.delete("userRole");
    
    return NextResponse.json({ 
      success: true, 
      message: "Logged out successfully" 
    });
  } catch (error) {
    console.error("Logout error:", error);
    return NextResponse.json(
      { success: false, message: "Error during logout" },
      { status: 500 }
    );
  }
} 