import { redirect } from "next/navigation";
import { requireAuth } from "@/lib/auth";
import AdminDashboard from "./dashboard";

export default async function AdminPage() {
  const user = await requireAuth('ADMIN');

  if (!user) {
    redirect("/auth/login");
  }

  return <AdminDashboard user={user} />;
}