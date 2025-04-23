"use server";

import prisma from '@/lib/prisma';
import { redirect } from 'next/navigation';
import Link from 'next/link';

async function seedDatabase() {
  try {
    console.log("Checking if users exist...");
    
    // Check if users already exist
    const adminCount = await prisma.user.count({
      where: { role: 'ADMIN' }
    });

    const doctorCount = await prisma.user.count({
      where: { role: 'DOCTOR' }
    });

    console.log(`Found ${adminCount} admin users and ${doctorCount} doctor users`);

    let adminCreated = false;
    let doctorCreated = false;

    // Create admin user if it doesn't exist
    if (adminCount === 0) {
      console.log("Creating admin user...");
      await prisma.user.create({
        data: {
          email: 'admin@medrecord.com',
          password: 'admin123', // In production, hash this!
          role: 'ADMIN',
          name: 'Admin User',
        },
      });
      adminCreated = true;
      console.log("Admin user created successfully");
    }

    // Create doctor user if it doesn't exist
    if (doctorCount === 0) {
      console.log("Creating doctor user...");
      await prisma.user.create({
        data: {
          email: 'doctor@medrecord.com',
          doctorId: 'DOC12345',
          password: 'doctor123', // In production, hash this!
          role: 'DOCTOR',
          name: 'Dr. Smith',
        },
      });
      doctorCreated = true;
      console.log("Doctor user created successfully");
    }

    return {
      adminCreated,
      doctorCreated,
      adminCount: adminCount + (adminCreated ? 1 : 0),
      doctorCount: doctorCount + (doctorCreated ? 1 : 0)
    };
  } catch (error) {
    console.error("Error seeding database:", error);
    throw error;
  }
}

export default async function SetupPage() {
  // Initialize result with default values to prevent undefined errors
  let result = {
    adminCreated: false,
    doctorCreated: false,
    adminCount: 0,
    doctorCount: 0
  };
  let error = null;
  
  try {
    // Test database connection
    await prisma.$queryRaw`SELECT 1 as test`;
    console.log("Database connection successful");
    
    // Seed database
    result = await seedDatabase();
  } catch (err) {
    console.error("Setup error:", err);
    // Add type guard to handle unknown error type
    error = err instanceof Error ? err.message : String(err);
  }
  
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4">
      <div className="bg-white rounded-lg shadow-md p-8 max-w-md w-full">
        <h1 className="text-2xl font-bold mb-6 text-center">Database Setup</h1>
        
        {error ? (
          <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-md">
            <h2 className="font-bold mb-2">Error:</h2>
            <p className="mb-2">{error}</p>
            <p>Check your database connection string in .env file:</p>
            <div className="bg-gray-100 p-3 rounded mt-2 text-sm font-mono break-all">
              DATABASE_URL="postgres://username:password@endpoint/database?sslmode=require"
            </div>
          </div>
        ) : (
          <div className="mb-6">
            <div className="p-4 bg-green-50 text-green-700 rounded-md mb-4">
              <p className="font-bold">✅ Database connection successful!</p>
            </div>
            
            <h2 className="font-bold mb-2">User Setup Results:</h2>
            <ul className="list-disc pl-5 mb-4">
              <li className="mb-1">
                Admin Users: {result.adminCount} 
                {result.adminCreated && <span className="text-green-600 ml-2">(New user created)</span>}
              </li>
              <li className="mb-1">
                Doctor Users: {result.doctorCount}
                {result.doctorCreated && <span className="text-green-600 ml-2">(New user created)</span>}
              </li>
            </ul>
            
            <div className="bg-gray-50 p-4 rounded-md">
              <h3 className="font-bold mb-2">Login Credentials:</h3>
              
              <div className="mb-3">
                <p className="font-medium">Admin:</p>
                <ul className="text-sm">
                  <li>Email: admin@medrecord.com</li>
                  <li>Password: admin123</li>
                </ul>
              </div>
              
              <div>
                <p className="font-medium">Doctor:</p>
                <ul className="text-sm">
                  <li>Email: doctor@medrecord.com</li>
                  <li>Doctor ID: DOC12345</li>
                  <li>Password: doctor123</li>
                </ul>
              </div>
            </div>
          </div>
        )}
        
        <div className="flex justify-center space-x-4">
          <Link 
            href="/auth/login/doctor" 
            className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
          >
            Go to Doctor Login
          </Link>
          
          <Link 
            href="/auth/login/admin" 
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Go to Admin Login
          </Link>
        </div>
      </div>
    </div>
  );
} 