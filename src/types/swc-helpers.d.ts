/**
 * Type declarations for @swc/helpers
 */
declare module '@swc/helpers' {
  export function applyDecoratedDescriptor(target: any, property: string, decorators: any[], descriptor: PropertyDescriptor, context?: any): PropertyDescriptor;
  export function _apply_decorated_descriptor(target: any, property: string, decorators: any[], descriptor: PropertyDescriptor, context?: any): PropertyDescriptor;
  
  // Add any other exports that might be needed
  export function _extends(...args: any[]): any;
  export function _object_spread(...args: any[]): any;
  export function _object_spread_props(...args: any[]): any;
} 