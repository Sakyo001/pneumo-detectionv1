"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import * as fs from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { analyzeXrayImage } from '@/lib/analysis';
import { prisma } from '@/lib/db';
import { cookies } from 'next/headers';

// Helper function to test database connection with retries
async function testDatabaseConnection(maxRetries = 3, delayMs = 1000): Promise<boolean> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Database connection attempt ${attempt}/${maxRetries}`);
      await prisma.$queryRaw`SELECT 1`;
      console.log("Database connection successful");
      return true;
    } catch (error) {
      console.error(`Database connection attempt ${attempt} failed:`, error);
      if (attempt < maxRetries) {
        console.log(`Retrying in ${delayMs}ms...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
        // Increase delay for next attempt (exponential backoff)
        delayMs *= 2;
      }
    }
  }
  return false;
}

// Helper function to generate a unique reference number
async function generateUniqueReferenceNumber(): Promise<string> {
  // Try up to 5 times to generate a unique number
  for (let attempt = 0; attempt < 5; attempt++) {
    // Generate a reference number format: XR-YYMMDD-RANDOM
    const now = new Date();
    const dateStr = `${now.getFullYear().toString().slice(-2)}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}`;
    const randomPart = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
    const refNumber = `XR-${dateStr}-${randomPart}`;
    
    // Check if this reference number already exists
    const existing = await prisma.xrayScan.findUnique({
      where: { referenceNumber: refNumber }
    });
    
    // If no existing scan with this reference, return it
    if (!existing) {
      return refNumber;
    }
  }
  
  // If all attempts failed, use UUID-based approach as fallback
  return `XR-${new Date().getTime().toString().slice(-6)}-${uuidv4().slice(0, 6)}`;
}

// Use the analyze API route to connect to the EfficientNet model
export async function uploadXray(formData: FormData) {
  try {
    console.log("Starting X-ray upload process");
    
    // Test database connection with retries
    const isConnected = await testDatabaseConnection();
    if (!isConnected) {
      console.error("Failed to establish database connection after multiple attempts");
      return { 
        error: "Database connection failed. Please try again later.",
        dbSaved: false
      };
    }
    
    // Extract the file from formData
    const file = formData.get('xrayFile') as File | null;
    if (!file) {
      return { error: "No X-ray image provided" };
    }
    
    // Extract patient information
    const patientName = formData.get('patientName') as string;
    const patientAge = formData.get('patientAge') as string;
    const patientGender = formData.get('patientGender') as string;
    const referenceNumber = formData.get('referenceNumber') as string;
    const patientNotes = formData.get('patientNotes') as string;
    
    // Save the image to the uploads directory
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    
    // Create a unique filename
    const fileName = `${referenceNumber}-${uuidv4()}.${file.name.split('.').pop()}`;
    const uploadDir = path.join(process.cwd(), 'public', 'uploads');
    
    // Ensure upload directory exists
    await fs.promises.mkdir(uploadDir, { recursive: true });
    
    const filePath = path.join(uploadDir, fileName);
    await fs.promises.writeFile(filePath, buffer);
    
    // File path for the client
    const imageUrl = `/uploads/${fileName}`;
    
    // Analyze the X-ray using the EfficientNet model directly
    const analysisResult = await analyzeXrayImage(file, {
      name: patientName,
      age: patientAge,
      gender: patientGender,
      referenceNumber: referenceNumber,
    }, {
      // Enable this if you want to use mock predictions instead of real model inference
      // This is useful when the model or Python environment is not set up correctly
      useMock: false
    });
    
    if (!analysisResult) {
      throw new Error("Analysis returned empty result");
    }

    console.log("Analysis result pneumoniaType:", analysisResult.pneumoniaType);
    console.log("Full analysis result:", JSON.stringify(analysisResult, null, 2));

    // Get current doctor ID from cookie
    const cookieStore = await cookies();
    const doctorId = cookieStore.get("userId")?.value;
    
    if (!doctorId) {
      console.error("No doctor ID found in cookies");
      return {
        ...analysisResult,
        imageUrl,
        referenceNumber,
        timestamp: new Date().toISOString(),
        patientNotes,
        dbSaved: false
      };
    }

    // Validate required data before database operations
    if (!referenceNumber) {
      console.error("Missing required reference number");
      return {
        ...analysisResult,
        imageUrl,
        referenceNumber: "ERROR-" + Date.now(),
        timestamp: new Date().toISOString(),
        patientNotes,
        dbSaved: false,
        error: "Failed to save: Missing reference number"
      };
    }

    if (!doctorId) {
      console.error("Missing required doctor ID");
      return {
        ...analysisResult,
        imageUrl,
        referenceNumber,
        timestamp: new Date().toISOString(),
        patientNotes,
        dbSaved: false,
        error: "Failed to save: Missing doctor ID"
      };
    }

    if (!analysisResult.diagnosis) {
      console.error("Missing required diagnosis result");
      return {
        ...analysisResult,
        diagnosis: "Unknown", // Provide a fallback value
        imageUrl,
        referenceNumber,
        timestamp: new Date().toISOString(),
        patientNotes,
        dbSaved: false,
        error: "Failed to save: Missing diagnosis result"
      };
    }

    // First, check if this patient already exists by name with this doctor
    let patient = await prisma.patient.findFirst({
      where: {
        name: patientName,
        doctorId: doctorId
      }
    });

    // If patient doesn't exist, create new patient record
    if (!patient) {
      // Generate a unique patient reference number
      const patientRefNumber = `P-${Date.now().toString().slice(-6)}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`;
      
      patient = await prisma.patient.create({
        data: {
          name: patientName,
          referenceNumber: patientRefNumber, // Use unique reference number instead of name
          dateOfBirth: patientAge ? new Date(new Date().getFullYear() - parseInt(patientAge), 0, 1) : new Date(),
          doctor: {
            connect: {
              id: doctorId
            }
          }
        }
      });
    }

    // Save X-ray scan data to database using the correct nested relation syntax
    let retryCount = 0;
    const maxRetries = 3;
    
    // Keep trying database save with exponential backoff
    while (retryCount <= maxRetries) {
      try {
        // Test connection before each attempt
        const isConnected = await testDatabaseConnection(1, 500);
        if (!isConnected) {
          throw new Error("Database connection lost");
        }
        
        const xrayScan = await prisma.xrayScan.create({
          data: {
            referenceNumber: referenceNumber,
            patientId: patient.id,
            doctorId: doctorId,
            imageUrl: imageUrl,
            result: analysisResult.diagnosis,
            pneumoniaType: analysisResult.pneumoniaType,
            severity: analysisResult.severity,
            recommendedAction: analysisResult.recommendedAction,
            status: "COMPLETED",
            metadata: {
              create: {
                confidence: analysisResult.confidence,
                pneumoniaType: analysisResult.pneumoniaType,
                severity: analysisResult.severity,
                recommendedAction: analysisResult.recommendedAction
              }
            }
          }
        });
        
        // Return the combined result
        return {
          ...analysisResult,
          imageUrl,
          referenceNumber,
          timestamp: new Date().toISOString(),
          patientNotes,
          dbSaved: true,
          scanId: xrayScan.id
        };
      } catch (dbError) {
        retryCount++;
        console.error(`Database error during X-ray scan creation (Attempt ${retryCount}/${maxRetries}):`, dbError);
        
        // Check for connection errors specifically
        const errorMsg = dbError instanceof Error ? dbError.message : String(dbError);
        const isConnectionError = 
          errorMsg.includes('Closed') || 
          errorMsg.includes('connection') || 
          errorMsg.includes('timeout') ||
          errorMsg.includes('ECONNREFUSED');
        
        // Log detailed error information for debugging
        console.error("Error details:", JSON.stringify({
          error: errorMsg,
          isConnectionError,
          patient: { id: patient.id, name: patientName },
          doctorId,
          referenceNumber
        }, null, 2));
        
        // For connection errors, try to reconnect
        if (isConnectionError) {
          console.log("Connection error detected, attempting to reconnect...");
          const reconnected = await testDatabaseConnection(2, 1000);
          if (!reconnected) {
            console.error("Failed to reconnect to database");
            if (retryCount >= maxRetries) {
              return {
                ...analysisResult,
                imageUrl,
                referenceNumber,
                timestamp: new Date().toISOString(),
                patientNotes,
                dbSaved: false,
                error: "Database connection error. Please try again later."
              };
            }
          }
        }
        
        // Check for unique constraint violation on reference number
        const isUniqueViolation = errorMsg.includes('Unique constraint failed on the fields: (`referenceNumber`)');
        
        // For unique constraint violations, return immediately with specific error
        if (isUniqueViolation) {
          // Try to generate a new unique reference number
          const newReferenceNumber = await generateUniqueReferenceNumber();
          
          // Attempt one more save with the new reference number
          try {
            // Test connection before retry
            const isConnected = await testDatabaseConnection(1, 500);
            if (!isConnected) {
              throw new Error("Database connection lost");
            }
            
            const xrayScan = await prisma.xrayScan.create({
              data: {
                referenceNumber: newReferenceNumber,
                patientId: patient.id,
                doctorId: doctorId,
                imageUrl: imageUrl,
                result: analysisResult.diagnosis,
                pneumoniaType: analysisResult.pneumoniaType,
                severity: analysisResult.severity,
                recommendedAction: analysisResult.recommendedAction,
                status: "COMPLETED",
                metadata: {
                  create: {
                    confidence: analysisResult.confidence,
                    pneumoniaType: analysisResult.pneumoniaType,
                    severity: analysisResult.severity,
                    recommendedAction: analysisResult.recommendedAction
                  }
                }
              }
            });
            
            // Return the combined result with the new reference number
            return {
              ...analysisResult,
              imageUrl,
              referenceNumber: newReferenceNumber,
              originalReference: referenceNumber,
              timestamp: new Date().toISOString(),
              patientNotes,
              dbSaved: true,
              scanId: xrayScan.id,
              message: "A new reference number was generated due to a conflict."
            };
          } catch (retryError) {
            // If the retry also fails, return error
            console.error("Failed to save scan with new reference number:", retryError);
            return {
              ...analysisResult,
              imageUrl,
              referenceNumber,
              timestamp: new Date().toISOString(),
              patientNotes,
              dbSaved: false,
              error: "A scan with this reference number already exists and automatic retry failed."
            };
          }
        }
        
        if (retryCount <= maxRetries) {
          // Wait with exponential backoff before retrying (1s, 2s, 4s)
          const delay = Math.pow(2, retryCount - 1) * 1000;
          console.log(`Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          // All retries failed, return error
          return {
            ...analysisResult,
            imageUrl,
            referenceNumber,
            timestamp: new Date().toISOString(),
            patientNotes,
            dbSaved: false,
            error: "Failed to save scan to database after multiple attempts. Please try again later."
          };
        }
      }
    }
  } catch (error) {
    console.error("Error in uploadXray:", error);
    return { 
      error: "An error occurred while processing the X-ray",
      // Provide fallback data
      usingMock: true,
      fallback: true,
      diagnosis: "Pneumonia", // Fixed value for consistency
      confidence: 75, // Fixed value for consistency
      pneumoniaType: "Bacterial",
      severity: "Moderate",
      severityDescription: "Unable to accurately determine severity due to analysis error.",
      recommendedAction: "Please consult with a medical professional as the automated analysis encountered an error.",
      dbSaved: false
    };
  }
} 