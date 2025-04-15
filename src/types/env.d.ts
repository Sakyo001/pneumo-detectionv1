declare global {
  namespace NodeJS {
    interface ProcessEnv {
      DATABASE_URL: string;
      DIRECT_URL: string;
      NODE_ENV: 'development' | 'production' | 'test';
      // ... other env variables
    }
  }
}

export {}; 