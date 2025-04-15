import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import { join } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

export async function GET() {
  try {
    // First check if we can connect to the Flask API server
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'}/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(2000), // 2 second timeout
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json({ 
          status: 'online',
          message: data.message || 'EfficientNet model is online',
          model: 'EfficientNet (Pneumonia Detection)',
          source: 'external'
        });
      }
    } catch (error) {
      console.log('Could not connect to external model server, checking for local model file...');
    }

    // If external API is unavailable, check if we have the model file locally
    // First try final_model.pth, then best_model.pth as fallback
    let modelFile = 'final_model.pth';
    let modelPath = join(process.cwd(), 'output', modelFile);
    
    try {
      await fs.access(modelPath);
      
      // Try to get model info
      let modelInfo = 'EfficientNet (Local)';
      let modelDetails = '';
      
      try {
        // Run a simple Python script to get model info
        const scriptPath = join(process.cwd(), 'scripts', 'model_info.py');
        
        try {
          await fs.access(scriptPath);
          const { stdout } = await execPromise(`python "${scriptPath}" --model "${modelPath}"`);
          if (stdout) {
            modelDetails = stdout.trim();
          }
        } catch (scriptError) {
          modelDetails = 'Unable to get model details';
          console.error('Error getting model details:', scriptError);
        }
      } catch (infoError) {
        console.error('Error checking model info:', infoError);
      }
      
      // Model file exists
      return NextResponse.json({ 
        status: 'online',
        message: 'Local EfficientNet model is available',
        model: modelInfo,
        details: modelDetails,
        source: 'local',
        modelFile,
        path: modelPath
      });
    } catch (finalModelError) {
      // Try best_model.pth as fallback
      modelFile = 'best_model.pth';
      modelPath = join(process.cwd(), 'output', modelFile);
      
      try {
        await fs.access(modelPath);
        // Backup model file exists
        return NextResponse.json({ 
          status: 'online',
          message: 'Local EfficientNet model is available (backup)',
          model: 'EfficientNet (Local - Backup)',
          source: 'local',
          modelFile,
          path: modelPath
        });
      } catch (bestModelError) {
        console.error('No model files found:', bestModelError);
        return NextResponse.json({ 
          status: 'offline',
          message: 'EfficientNet model is offline',
          error: 'No model files found'
        }, { status: 503 });
      }
    }
  } catch (error) {
    console.error('Error checking model status:', error);
    return NextResponse.json({ 
      status: 'offline',
      message: 'Could not connect to EfficientNet model server',
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 503 });
  }
} 