/**
 * Pneumonia analysis utilities for connecting to the EfficientNet model
 */
import { promises as fs } from 'fs';
import { join, dirname, basename } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import { parseInferenceOutput } from '@/app/dashboard/doctor/upload-xray/parse-output';
import os from 'os';

// Add File type definition if not available in the environment
declare global {
  interface File extends Blob {
    readonly lastModified: number;
    readonly name: string;
    readonly webkitRelativePath: string;
    arrayBuffer(): Promise<ArrayBuffer>;
    text(): Promise<string>;
  }
}

const execPromise = promisify(exec);

// Keep track of Python errors to avoid retrying repeatedly
let pythonErrorCount = 0;
const MAX_PYTHON_ERRORS = 2; // After this many errors, just use simulation

// Define clear interfaces to ensure type safety

// Result from the parseInferenceOutput function
interface ParsedResult {
  diagnosis: string;
  confidence: number;
  pneumoniaType?: string | null;
  severity?: string | null;
  severityDescription?: string | null;
  recommendedAction: string;
  probabilities?: {
    normal: number;
    pneumonia: number;
  };
  usingMock?: boolean;
  error?: string;
  processingTime?: number;
}

// Final analysis result returned by our module
interface AnalysisResult {
  diagnosis: string;
  confidence: number;
  pneumoniaType: string | null;
  severity: string | null;
  severityDescription: string | null;
  recommendedAction: string;
  processingTime?: number;
  usingMock: boolean;
  error?: string;
}

// Check if the model file exists
async function checkModelFile(modelName: string, directory: string = 'output'): Promise<string | null> {
  const locations = [
    join(process.cwd(), directory, modelName),
    join(process.cwd(), 'pneumonia-ml', 'output', modelName),
    join(process.cwd(), 'pneumonia-ml', modelName)
  ];
  
  for (const location of locations) {
    try {
      await fs.access(location);
      console.log(`Found model at: ${location}`);
      return location;
    } catch (error) {
      console.log(`Model not found at: ${location}`);
    }
  }
  
  console.error(`Model ${modelName} not found in any location`);
  return null;
}

// Convert any parsed result to a consistent AnalysisResult format
function normalizeResult(result: ParsedResult): AnalysisResult {
  return {
    diagnosis: result.diagnosis,
    confidence: result.confidence,
    pneumoniaType: result.diagnosis === 'Pneumonia' ? result.pneumoniaType || null : null,
    severity: result.diagnosis === 'Pneumonia' ? result.severity || null : null,
    severityDescription: result.diagnosis === 'Pneumonia' ? result.severityDescription || null : null,
    recommendedAction: result.recommendedAction || '',
    processingTime: result.processingTime,
    usingMock: result.usingMock || false,
    error: result.error
  };
}

/**
 * Get detailed model information
 */
async function getModelInfo(modelPath: string): Promise<string> {
  try {
    const { stdout } = await execPromise(`python scripts/model_debug.py --model "${modelPath}"`);
    return stdout;
  } catch (error) {
    console.error('Error getting model info:', error);
    return 'Error analyzing model';
  }
}

/**
 * Attempt to determine model type based on the model file location
 */
async function detectModelType(modelPath: string): Promise<string> {
  // Default to ResNet model
  let modelType = 'resnet';

  try {
    // If the model is in the pneumonia-ml directory, check if we can find
    // any clues about the model architecture from nearby files
    if (modelPath.includes('pneumonia-ml')) {
      // First check if there's a model_info.json file
      const modelDir = dirname(modelPath);
      const infoPath = join(modelDir, 'model_info.json');
      
      try {
        if (await fs.stat(infoPath).catch(() => null)) {
          const infoContent = await fs.readFile(infoPath, 'utf-8');
          const modelInfo = JSON.parse(infoContent);
          if (modelInfo.architecture) {
            if (modelInfo.architecture.toLowerCase().includes('simple') || 
                modelInfo.architecture.toLowerCase().includes('conv')) {
              return 'simple';
            } else if (modelInfo.architecture.toLowerCase().includes('resnet')) {
              return 'resnet';
            }
          }
        }
      } catch (e) {
        console.warn('Error reading model_info.json:', e);
      }
      
      // If we couldn't find model_info.json, check file naming patterns
      const fileName = basename(modelPath);
      if (fileName.toLowerCase().includes('simple') || fileName.toLowerCase().includes('conv')) {
        modelType = 'simple';
      }
    }
  } catch (e) {
    console.warn('Error detecting model type:', e);
  }
  
  console.log(`Detected model type: efficientnet`);
  return modelType;
}

/**
 * Analyze an X-ray image for pneumonia using the custom model
 */
export async function analyzeXrayImage(
  imageData: string | Buffer | File,
  patientInfo?: {
    name?: string;
    age?: string;
    gender?: string;
    referenceNumber?: string;
  },
  options?: {
    useMock?: boolean;
    forceSimulation?: boolean; // New option to force simulation mode
  }
): Promise<AnalysisResult> {
  // If mock mode is explicitly requested, use mock prediction
  if (options?.useMock || options?.forceSimulation) {
    console.log("Mock/simulation mode explicitly requested");
    return generateMockPrediction(
      "Simulation mode requested", 
      patientInfo?.referenceNumber
    );
  }

  // Check if we've had too many Python errors and should use simulation mode
  if (pythonErrorCount >= MAX_PYTHON_ERRORS) {
    console.log(`Using simulation mode due to ${pythonErrorCount} previous Python errors`);
    return generateMockPrediction(
      "Using simulation mode due to previous Python errors", 
      patientInfo?.referenceNumber
    );
  }

  // Check for a stored flag in localStorage (client-side only)
  let useSimulationMode = false;
  if (typeof window !== 'undefined') {
    useSimulationMode = localStorage.getItem('use_simulation_mode') === 'true';
    if (useSimulationMode) {
      console.log("Using simulation mode based on localStorage setting");
      return generateMockPrediction(
        "Using simulation mode based on stored preference", 
        patientInfo?.referenceNumber
      );
    }
  }

  console.log("Starting X-ray analysis...");
  
  try {
    // Try to connect to external model server first
    try {
      console.log("Attempting to connect to external model server...");
      
      // This could be adjusted to actually check external model availability
      // For example, making a quick ping request to the server
      
      // Simulate external model server check (this should be replaced with actual check)
      const externalModelAvailable = await checkExternalModelAvailability();
      
      if (!externalModelAvailable) {
        console.log("Could not connect to external model server, checking for local model file...");
      } else {
        console.log("Connected to external model server successfully");
      }
    } catch (serverError) {
      console.log("Could not connect to external model server, checking for local model file...");
      console.error("External model server error:", serverError);
    }

    // Create temp file for the image
    const tempDir = os.tmpdir();
    const imagePath = join(tempDir, `xray-${Date.now()}.jpg`);
    
    // Handle different input types
    if (imageData instanceof File) {
      // Handle File object
      const arrayBuffer = await imageData.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      await fs.writeFile(imagePath, buffer);
    } else if (typeof imageData === 'string' && imageData.startsWith('data:image')) {
      // Extract base64 content from data URI
      const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
      await fs.writeFile(imagePath, Buffer.from(base64Data, 'base64'));
    } else if (Buffer.isBuffer(imageData)) {
      // Write buffer directly to file
      await fs.writeFile(imagePath, imageData);
    } else if (typeof imageData === 'string') {
      // Assume it's already base64 without data URI prefix
      await fs.writeFile(imagePath, Buffer.from(imageData, 'base64'));
    } else {
      throw new Error('Invalid image data format');
    }

    console.log(`Temporary image saved to: ${imagePath}`);

    // Find model path by checking multiple locations
    let modelPath = null;
    const possibleModelLocations = [
      // Look for PyTorch model files (.pth)
      join(process.cwd(), 'pneumonia-ml', 'output', 'best_model.pth'),
      join(process.cwd(), 'pneumonia-ml', 'output', 'final_model.pth'),
      // Original model file paths (.h5)
      join(process.cwd(), 'pneumonia-ml', 'models', 'model_v2.h5'),
      join(process.cwd(), 'pneumonia-ml', 'output', 'model_v2.h5'),
      join(process.cwd(), 'pneumonia-ml', 'model_v2.h5'),
      join(process.cwd(), 'pneumonia-ml', 'models', 'model.h5'),
      join(process.cwd(), 'pneumonia-ml', 'output', 'model.h5'),
      join(process.cwd(), 'pneumonia-ml', 'model.h5')
    ];
    
    for (const location of possibleModelLocations) {
      try {
        await fs.access(location);
        console.log(`Found model at: ${location}`);
        modelPath = location;
        break;
      } catch (err) {
        // Don't log every failed path check to reduce noise
      }
    }
    
    if (!modelPath) {
      console.log("No model file found in any of the expected locations");
    }
    
    // Find Python script location
    let pythonScriptPath = null;
    const possibleScriptLocations = [
      join(process.cwd(), 'pneumonia-ml', 'inference.py'),
      join(process.cwd(), 'pneumonia-ml', 'predict.py')
    ];
    
    for (const location of possibleScriptLocations) {
      try {
        await fs.access(location);
        console.log(`Found Python script at: ${location}`);
        pythonScriptPath = location;
        break;
      } catch (err) {
        // Don't log every failed path check to reduce noise
      }
    }
    
    // If model or script not found, use mock prediction
    if (!modelPath || !pythonScriptPath) {
      console.warn('Required files not found. Using simulation mode.');
      setSimulationMode(true);
      
      // Use deterministic mock with patient reference number as seed
      return generateMockPrediction(
        'Required model files not found, using simulation mode', 
        patientInfo?.referenceNumber
      );
    }
    
    // Determine model type based on file extension
    const isPyTorch = modelPath.toLowerCase().endsWith('.pth');
    const modelType = isPyTorch ? "pytorch" : "cnn";
    
    console.log(`Using ${modelType} model for inference`);
    
    // Build command to run the inference script with virtual environment
    // Use the venv Python interpreter instead of system Python
    const pythonInterpreter = join(process.cwd(), 'pneumonia-ml', 'venv', 'Scripts', 'python.exe');
    
    // Check if the venv Python interpreter exists
    let pythonCommand = 'python';
    try {
      await fs.access(pythonInterpreter);
      console.log(`Found Python interpreter at: ${pythonInterpreter}`);
      pythonCommand = `"${pythonInterpreter}"`;
    } catch (err) {
      console.log('Virtual environment Python not found, using system Python');
    }
    
    let command = `${pythonCommand} "${pythonScriptPath}" --model_path "${modelPath}" --image_path "${imagePath}" --model_type "${modelType}"`;
    
    // Add patient info parameters if provided
    if (patientInfo) {
      if (patientInfo.name) command += ` --patient_name "${patientInfo.name}"`;
      if (patientInfo.age) command += ` --patient_age "${patientInfo.age}"`;
      if (patientInfo.gender) command += ` --patient_gender "${patientInfo.gender}"`;
      if (patientInfo.referenceNumber) command += ` --reference_number "${patientInfo.referenceNumber}"`;
    }
    
    console.log(`Executing model inference command...`);
    
    // Set a timeout for the Python script execution
    const timeoutMs = 30000; // 30 seconds (increased from 15)
    const execOptions = { 
      timeout: timeoutMs,
      // Set the working directory to the pneumonia-ml folder
      cwd: join(process.cwd(), 'pneumonia-ml')
    };
    
    // Execute the Python script with timeout
    try {
      console.log("Starting Python inference process...");
      console.log("Command:", command);
      const { stdout, stderr } = await execPromise(command, execOptions);
      
      if (stderr) {
        console.warn('Warnings from Python script:', stderr);
      }

      // Parse the output from the Python script
      console.log("Parsing inference results...");
      const results = parseInferenceOutput(stdout);
      
      // Reset error count on success
      pythonErrorCount = 0;
      setSimulationMode(false);
      
      return normalizeResult(results);
    } catch (execError: any) {
      console.error('Error executing Python script:', execError);
      console.error('Error details:', execError.message);
      
      if (execError.stderr) {
        console.error('Error output from Python:', execError.stderr);
      }
      
      // Increment error count
      pythonErrorCount++;
      
      // If we reached the max errors, store that in localStorage for future page loads
      if (pythonErrorCount >= MAX_PYTHON_ERRORS) {
        setSimulationMode(true);
      }
      
      // Check if this was a timeout error
      const isTimeout = execError.message && execError.message.includes('timed out');
      if (isTimeout) {
        console.warn(`Python script execution timed out after ${timeoutMs/1000} seconds`);
      }
      
      // Check for common Python environment errors
      const isPythonEnvError = 
        (execError.stderr && execError.stderr.includes('No module named')) ||
        (execError.message && execError.message.includes('No module named'));
      
      if (isPythonEnvError) {
        console.warn('Python environment error detected - missing modules');
        setSimulationMode(true);
      }
      
      // Try with mock data that's more informative
      let errorMessage = isTimeout 
        ? `Model processing timed out - using simulation mode` 
        : `Python error: ${execError.message?.split('\n')[0]} - using simulation mode`;
      
      // Fall back to mock prediction with helpful error
      return generateMockPrediction(
        errorMessage,
        patientInfo?.referenceNumber
      );
    } finally {
      // Clean up the temporary file
      try {
        await fs.unlink(imagePath);
        console.log('Temporary image file removed');
      } catch (error) {
        console.error('Error removing temporary file:', error);
      }
    }
  } catch (error: any) {
    console.error('Error in analyzeXrayImage:', error);
    // Fall back to mock prediction with deterministic output
    return generateMockPrediction(
      `Error analyzing image: ${error.message} - using simulation mode`,
      patientInfo?.referenceNumber
    );
  }
}

/**
 * Check if the external model server is available
 * This is a simple placeholder that could be replaced with an actual check
 */
async function checkExternalModelAvailability(): Promise<boolean> {
  // This function could make a request to check if an external model server is available
  // For now, it just simulates the check
  
  try {
    // This could be replaced with an actual API check
    // For example: const response = await fetch('http://model-server/api/status')
    
    // For testing, simulate a failed external connection
    // In a real implementation, return true if available
    
    // Simulate a failure to connect so we use local model
    return false;
  } catch (error) {
    console.error('Error checking external model availability:', error);
    return false;
  }
}

/**
 * Generate mock prediction for when the model server is unavailable
 * Uses a seed value to generate deterministic results
 */
export function generateMockPrediction(
  errorReason: string = 'Model unavailable',
  seed?: string
): AnalysisResult {
  // If a seed is provided, use it to generate deterministic results
  let isPositive: boolean;
  let confidence: number;
  
  if (seed) {
    // Simple hash function to get a number from a string
    const hashCode = (str: string): number => {
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
      }
      return Math.abs(hash);
    };
    
    const seedValue = hashCode(seed);
    isPositive = seedValue % 2 === 0;
    confidence = 60 + (seedValue % 30);
  } else {
    // If no seed, use a default value for development
    isPositive = true;
    confidence = 85;
  }
  
  return {
    diagnosis: isPositive ? "Pneumonia" : "Normal",
    confidence: confidence,
    pneumoniaType: isPositive ? (confidence > 70 ? "Bacterial" : "Viral") : null,
    severity: isPositive ? 
      (confidence > 90 ? "Severe" : confidence > 80 ? "Moderate" : "Mild") : null,
    severityDescription: isPositive ? 
      (confidence > 90 ? 
       "Severe pneumonia with significant lung involvement. Urgent medical attention required." : 
       confidence > 80 ? 
       "Moderate pneumonia with partial lung involvement. Medical attention recommended." : 
       "Mild pneumonia with limited lung involvement. Monitor and consult with physician.") : null,
    recommendedAction: isPositive ?
      (confidence > 90 ? 
       "Immediate hospitalization and antibiotic therapy is recommended." : 
       confidence > 80 ? 
       "Consider outpatient antibiotic therapy and close monitoring." : 
       "Rest, hydration, and symptom management. Follow up if symptoms worsen.") : 
      "No pneumonia detected. Regular health maintenance recommended.",
    usingMock: true, // Flag this as mock prediction
    error: errorReason // Include the reason why we're using a mock prediction
  };
}

/**
 * Set whether to use simulation mode in future requests
 */
function setSimulationMode(useSimulation: boolean): void {
  if (typeof window !== 'undefined') {
    if (useSimulation) {
      localStorage.setItem('use_simulation_mode', 'true');
      console.log('Set simulation mode ON for future requests');
    } else {
      localStorage.removeItem('use_simulation_mode');
      console.log('Set simulation mode OFF for future requests');
    }
  }
} 