/**
 * Utility functions for the application
 */

/**
 * Serialize data to JSON, converting any BigInt values to numbers
 * This is necessary because BigInt cannot be serialized to JSON directly
 */
export function serializeForJson<T>(data: T): T {
  return JSON.parse(
    JSON.stringify(data, (key, value) => 
      typeof value === 'bigint' 
        ? Number(value) 
        : value
    )
  );
}

/**
 * Safely convert any value to a number
 * Handles BigInt, number, and string conversions with fallback to 0
 */
export function safeNumberConversion(value: any): number {
  if (typeof value === 'bigint') {
    return Number(value);
  }
  
  if (typeof value === 'number') {
    return value;
  }
  
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    return isNaN(parsed) ? 0 : parsed;
  }
  
  return 0;
} 