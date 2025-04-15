import Image from "next/image";
import Link from "next/link";

export default function Login() {
  // All your existing Login component code
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
          <Link href="/" className="text-indigo-600 text-sm hover:text-indigo-800">
            Back to Home
          </Link>
        </div>
      </nav>

      {/* Login Content */}
      <div className="flex flex-1 justify-center items-center px-4 py-12">
        <div className="w-full max-w-md">
          <div className="bg-white shadow-md rounded-xl p-8">
            <div className="text-center mb-8">
              <div className="flex justify-center mb-4">
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center">
                  <Image src="/icons/logo.png" alt="Logo" width={24} height={24} />
                </div>
              </div>
              <h2 className="text-2xl font-bold text-gray-800 mb-1">Welcome Back</h2>
              <p className="text-gray-600 text-sm">Sign in to continue to MedRecord Hub</p>
            </div>

            {/* Login Options */}
            <div className="space-y-4">
              {/* Admin Login - Now clickable with Link */}
              <Link href="/auth/login/admin">
                <div className="border border-gray-200 mb-2 rounded-lg p-4 hover:border-indigo-300 hover:bg-indigo-50/30 transition-all cursor-pointer">
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-indigo-50 rounded-full flex items-center justify-center mr-4">
                      <Image src="/images/admin.png" alt="Admin" width={24} height={24} />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-800">Sign in as Admin</h3>
                      <p className="text-xs text-gray-500">Full system control and management</p>
                    </div>
                    <div className="ml-auto">
                      <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7"></path>
                      </svg>
                    </div>
                  </div>
                </div>
              </Link>

              {/* Doctor Login - Now clickable with Link */}
              <Link href="/auth/login/doctor">
                <div className="border border-gray-200 rounded-lg p-4 hover:border-indigo-300 hover:bg-indigo-50/30 transition-all cursor-pointer">
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-indigo-50 rounded-full flex items-center justify-center mr-4">
                      <Image src="/images/doctor.png" alt="Doctor" width={24} height={24} />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-800">Sign in as Doctor</h3>
                      <p className="text-xs text-gray-500">Access patient records and appointments</p>
                    </div>
                    <div className="ml-auto">
                      <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7"></path>
                      </svg>
                    </div>
                  </div>
                </div>
              </Link>

              {/* Divider */}
              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-200"></div>
                </div>
                <div className="relative flex justify-center">
                  <span className="bg-white px-4 text-xs text-gray-500">or</span>
                </div>
              </div>

              {/* Patient Access */}
              <div className="text-center">
                <Link 
                  href="/dashboard/patient" 
                  className="text-indigo-600 hover:text-indigo-800 text-sm font-medium flex items-center justify-center"
                >
                  <span className="mr-2">
                    <Image src="/images/patient.png" alt="Patient" width={20} height={20} />
                  </span>
                  Continue as Patient (Public View)
                </Link>
                <p className="text-xs text-gray-500 mt-1">
                  No login required for basic information
                </p>
              </div>
            </div>

            {/* Help Section */}
            <div className="mt-8 pt-6 border-t border-gray-100 text-center">
              <p className="text-sm text-gray-600">
                Need help? <a href="#" className="text-indigo-600 hover:text-indigo-800">Contact Support</a>
              </p>
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