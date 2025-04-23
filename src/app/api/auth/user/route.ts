import { cookies } from 'next/headers';

/**
 * Fast, lightweight endpoint to get current user info
 * Used by the upload-xray page to quickly display doctor name
 */
export async function GET() {
  try {
    // Get user ID from cookie - await the cookies Promise
    const cookieStore = await cookies();
    const userId = cookieStore.get("userId");
    const userRole = cookieStore.get("userRole");
    const userName = cookieStore.get("userName");

    if (!userId || !userRole) {
      return Response.json(
        { success: false, message: "Not authenticated" },
        { status: 401 }
      );
    }

    // Return user info
    return Response.json({
      success: true,
      user: {
        id: userId.value,
        role: userRole.value,
        name: userName?.value || "User"
      }
    });
  } catch (error) {
    console.error("Error getting user info:", error);
    return Response.json(
      { success: false, message: "Server error" },
      { status: 500 }
    );
  }
} 