import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import DoctorDashboardClient from "./client";

export default async function DoctorDashboard() {
  // Check if user is logged in
  const cookieStore = await cookies();
  const userId = cookieStore.get("userId");
  const userRole = cookieStore.get("userRole");
  const userName = cookieStore.get("userName");
  
  if (!userId || !userRole) {
    redirect("/auth/login");
  }
  
  // Check if user has the doctor role
  if (userRole.value !== "DOCTOR") {
    redirect("/auth/login");
  }

  // We need to fetch the user data to pass to the client component
  // Since we don't have direct access to the session data
  try {
    // Use the userName cookie if available, otherwise use a default value
    // The client component will attempt to get the real name from localStorage or API
    const user = {
      id: userId.value,
      role: userRole.value,
      name: userName?.value || "Doctor"
    };
    
    return <DoctorDashboardClient user={user} />;
  } catch (err) {
    console.error("Error setting up dashboard:", err);
    redirect("/auth/login");
  }
}