import { NextRequest, NextResponse } from "next/server";
import { requireAuth } from "@/lib/auth";
import prisma from "@/lib/prisma";

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
    
    // Get all users
    const users = await prisma.user.findMany({
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        doctorId: true,
        createdAt: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
    
    return NextResponse.json(users);
  } catch (error) {
    console.error("Error fetching users:", error);
    return NextResponse.json(
      { error: "Failed to fetch users", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const user = await requireAuth('ADMIN');
    
    // If not admin, return unauthorized
    if (!user || user.role !== 'ADMIN') {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    const data = await request.json();
    
    // Create new user
    const newUser = await prisma.user.create({
      data: {
        name: data.name,
        email: data.email,
        password: data.password, // In production, you should hash the password
        role: data.role,
        doctorId: data.doctorId,
      },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        doctorId: true,
        createdAt: true
      }
    });
    
    return NextResponse.json(newUser);
  } catch (error) {
    console.error("Error creating user:", error);
    return NextResponse.json(
      { error: "Failed to create user", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
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
    
    const data = await request.json();
    
    // Update user
    const updatedUser = await prisma.user.update({
      where: {
        id: data.id,
      },
      data: {
        name: data.name,
        email: data.email,
        role: data.role,
        doctorId: data.doctorId,
        // Only update password if provided
        ...(data.password ? { password: data.password } : {}),
      },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        doctorId: true,
        createdAt: true
      }
    });
    
    return NextResponse.json(updatedUser);
  } catch (error) {
    console.error("Error updating user:", error);
    return NextResponse.json(
      { error: "Failed to update user", details: error instanceof Error ? error.message : String(error) },
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
        { error: "User ID is required" },
        { status: 400 }
      );
    }
    
    // Delete user
    await prisma.user.delete({
      where: {
        id,
      },
    });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting user:", error);
    return NextResponse.json(
      { error: "Failed to delete user", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
} 