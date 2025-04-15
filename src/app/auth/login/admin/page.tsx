import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";
import { login, seedUsers } from "@/lib/auth";

// Run the seed function to ensure we have sample users
// In production, you'd do this differently
export async function generateMetadata() {
  try {
    await seedUsers();
  } catch (error) {
    console.error("Error seeding users:", error);
  }
  
  return {
    title: "Admin Login - MedRecord Hub",
  };
}

async function handleAdminLogin(formData: FormData) {
  "use server";
  
  const email = formData.get("email") as string;
  const password = formData.get("password") as string;
  
  const result = await login(email, password, "ADMIN");
  
  if (result.success) {
    redirect("/dashboard/admin");
  }
  
  // In a real app, you'd want to display the error message
  // For simplicity, we'll just redirect back to the login page
  return redirect("/auth/login/admin?error=Invalid credentials");
}

export default function AdminLogin() {
  return (
    <main className="flex flex-col min-h-screen bg-white">
      {/* Navigation Bar */}
      <nav className="flex justify-between items-center px-6 py-3 bg-white">
        <div className="flex items-center">
          <div className="mr-2 flex items-center justify-center w-7 h-7">
            <Image src="/icons/logo.png" alt="Logo" width={16} height={16} />
          </div>
          <h1 className="font-semibold text-gray-800">MedRecord Hub</h1>
        </div>
        <div>
          <Link href="/auth/login" className="text-indigo-600 text-sm hover:text-indigo-800">
            Back to Login
          </Link>
        </div>
      </nav>

      {/* Admin Login Content */}
      <div className="flex flex-1 justify-center items-center px-4 py-12">
        <div className="w-full max-w-md">
          <div className="bg-white shadow-md rounded-xl p-8">
            <div className="text-center mb-6">
              <div className="flex justify-center mb-4">
                <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center">
                  <Image src="/images/admin.png" alt="Admin" width={40} height={40} />
                </div>
              </div>
              <h2 className="text-2xl font-bold text-gray-800 mb-1">Admin Login</h2>
              <p className="text-gray-600 text-sm">Access the administrative dashboard</p>
            </div>

            {/* Login Form */}
            <form action={handleAdminLogin} className="mt-8 space-y-5">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <input 
                  type="email" 
                  id="email" 
                  name="email" 
                  placeholder="admin@medrecord.com" 
                  className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  required
                />
              </div>
              
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                  Password
                </label>
                <input 
                  type="password" 
                  id="password" 
                  name="password" 
                  placeholder="Enter your password" 
                  className="w-full px-3 py-2 border text-gray-600 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  required
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
                    Remember me
                  </label>
                </div>
                
                <div className="text-sm">
                  <a href="#" className="text-indigo-600 hover:text-indigo-800">
                    Forgot password?
                  </a>
                </div>
              </div>
              
              <div>
                <button
                  type="submit"
                  className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Sign in to Admin Dashboard
                </button>
              </div>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                This area is restricted to authorized administrators only
              </p>
              {/* Sample login info for demo purposes */}
              <div className="mt-4 p-3 bg-gray-50 rounded-md text-xs text-gray-500 text-left">
                <p className="font-medium mb-1">Demo Credentials:</p>
                <p>Email: admin@medrecord.com</p>
                <p>Password: admin123</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Simple Footer */}
      <footer className="bg-white py-4 border-t text-center text-sm text-gray-600">
        <div className="container mx-auto">
          <p>© 2023 MedRecord Hub • <a href="#" className="text-indigo-600">Privacy</a> • <a href="#" className="text-indigo-600">Terms</a></p>
        </div>
      </footer>
    </main>
  );
}
