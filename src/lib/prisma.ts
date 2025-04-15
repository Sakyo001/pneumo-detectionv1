import { PrismaClient } from '@prisma/client';

// PrismaClient is attached to the `global` object in development to prevent
// exhausting your database connection limit.
// Learn more: https://pris.ly/d/help/next-js-best-practices

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

const RETRY_COUNT = 5;
const INITIAL_RETRY_DELAY = 1000;

const prismaClientSingleton = () => {
  return new PrismaClient({
    log: ['error', 'warn'],
    datasources: {
      db: {
        url: process.env.DATABASE_URL,
      },
    },
  });
};

// Initialize Prisma Client
const prisma = globalForPrisma.prisma ?? prismaClientSingleton();

// Add connection retry logic with exponential backoff
const connectWithRetry = async (retries = RETRY_COUNT, delay = INITIAL_RETRY_DELAY) => {
  let currentAttempt = 0;
  
  while (currentAttempt < retries) {
    try {
      await prisma.$connect();
      console.log('✅ Successfully connected to the database');
      return;
    } catch (error) {
      currentAttempt++;
      console.error(`❌ Failed to connect to the database (attempt ${currentAttempt}/${retries}):`, error);
      
      if (currentAttempt >= retries) {
        console.error('⛔ Maximum connection attempts reached. Please check your database configuration.');
      } else {
        console.log(`⏱️ Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        // Exponential backoff
        delay = Math.min(delay * 2, 30000); // Cap at 30 seconds
      }
    }
  }
};

// Connect with retry logic
connectWithRetry();

// Store the client in the global object in development
if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

export default prisma;