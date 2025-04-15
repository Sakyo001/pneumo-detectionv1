import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import prisma from "@/lib/prisma";
import { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  try {
    // Ensure we can read the body properly
    const contentType = request.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      return NextResponse.json(
        { success: false, message: "Content-Type must be application/json" },
        { status: 400 }
      );
    }

    const body = await request.json();
    const { doctorId, password, role } = body;
    
    console.log("Login attempt:", { doctorId, role });
    
    let user;
    
    if (role === "DOCTOR") {
      // Find doctor by doctorId
      user = await prisma.user.findFirst({
        where: { doctorId }
      });
    } else {
      // Find admin by email
      user = await prisma.user.findFirst({
        where: { 
          email: body.email,
          role
        }
      });
    }
    
    if (!user) {
      console.log("User not found");
      return NextResponse.json({ success: false, message: "Invalid credentials" }, { status: 401 });
    }
    
    // Check password
    if (user.password !== password) {
      console.log("Invalid password");
      return NextResponse.json({ success: false, message: "Invalid credentials" }, { status: 401 });
    }
    
    // Set session cookies
    const cookieStore = await cookies();
    
    cookieStore.set("userId", user.id, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      maxAge: 60 * 60 * 24 * 7, // 1 week
      path: "/"
    });
    
    cookieStore.set("userRole", user.role, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      maxAge: 60 * 60 * 24 * 7, // 1 week
      path: "/"
    });
    
    // Return success
    return NextResponse.json({ 
      success: true, 
      user: {
        id: user.id,
        name: user.name,
        role: user.role
      }
    });
  } catch (error) {
    console.error("Login error:", error);
    return NextResponse.json({ success: false, message: "An error occurred during login" }, { status: 500 });
  }
} 