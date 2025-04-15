import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function middleware(request: NextRequest) {
  const userId = request.cookies.get('userId')?.value;

  // Protect dashboard routes except patient dashboard
  if (request.nextUrl.pathname.startsWith('/dashboard') && !request.nextUrl.pathname.startsWith('/dashboard/patient')) {
    if (!userId) {
      return NextResponse.redirect(new URL('/auth/login', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/auth/:path*',
  ],
} 