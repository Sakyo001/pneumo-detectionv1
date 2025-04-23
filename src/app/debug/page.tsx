"use server";

import prisma from "@/lib/prisma";

export default async function DebugPage() {
  let dbTestResult = "Not tested";
  let createUserResult = "Not attempted";
  let errorMessage = null;
  
  try {
    // Test basic connection
    try {
      const result = await prisma.$queryRaw`SELECT 1 as test`;
      dbTestResult = `Connection successful: ${JSON.stringify(result)}`;
    } catch (err) {
      // Add type guard for the error
      const errorMessage = err instanceof Error ? err.message : String(err);
      dbTestResult = `Connection error: ${errorMessage}`;
      throw err;
    }
    
    // Try to create a test user
    try {
      const testUser = await prisma.user.create({
        data: {
          email: 'test@example.com',
          password: 'test123',
          role: 'DOCTOR',
          name: 'Test User',
          doctorId: 'TEST123'
        }
      });
      
      createUserResult = `User created successfully with ID: ${testUser.id}`;
      
      // List all users
      const users = await prisma.user.findMany();
      createUserResult += `\nTotal users: ${users.length}\nUsers: ${JSON.stringify(users.map(u => ({ id: u.id, email: u.email, name: u.name })))}`;
    } catch (err) {
      // Add type guard for the error
      const errorMessage = err instanceof Error ? err.message : String(err);
      createUserResult = `User creation error: ${errorMessage}`;
      throw err;
    }
  } catch (error) {
    // Add type guard for the error
    errorMessage = error instanceof Error 
      ? `${error.name}: ${error.message}\n${error.stack}` 
      : String(error);
  }
  
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Database Debug Page</h1>
      
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-2">Database Connection Test</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-40">{dbTestResult}</pre>
      </div>
      
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-2">User Creation Test</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-60">{createUserResult}</pre>
      </div>
      
      {errorMessage && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-2 text-red-600">Error Details</h2>
          <pre className="bg-red-50 text-red-800 p-4 rounded overflow-auto max-h-96">{errorMessage}</pre>
        </div>
      )}
      
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-2">Environment</h2>
        <pre className="bg-gray-100 p-4 rounded overflow-auto max-h-40">
          NODE_ENV: {process.env.NODE_ENV}
          Database URL: {process.env.DATABASE_URL ? '(Set, but hidden for security)' : '(Not set)'}
        </pre>
      </div>
    </div>
  );
} 