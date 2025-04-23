'use server';

import prisma from '@/lib/prisma';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export async function login(email: string, password: string, role: 'ADMIN' | 'DOCTOR') {
  try {
    const user = await prisma.user.findFirst({
      where: {
        email,
        role,
      },
    });

    if (!user) {
      return { success: false, message: 'User not found' };
    }

    // In production, use proper password hashing
    const isMatch = password === user.password;

    if (!isMatch) {
      return { success: false, message: 'Invalid credentials' };
    }

    // Await the cookies() function before calling set()
    const cookieStore = await cookies();
    cookieStore.set('userId', user.id, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 7, // 1 week
    });

    return { success: true, user };
  } catch (error) {
    console.error('Login error:', error);
    return { success: false, message: 'An error occurred during login' };
  }
}

export async function logout() {
  // Await the cookies() function before calling delete()
  const cookieStore = await cookies();
  cookieStore.delete('userId');
  redirect('/');
}

export async function getUserFromCookie() {
  // Await the cookies() function before calling get()
  const cookieStore = await cookies();
  const userId = cookieStore.get('userId')?.value;
  
  if (!userId) {
    return null;
  }

  try {
    return await prisma.user.findUnique({
      where: { id: userId },
    });
  } catch (error) {
    console.error('Error getting user:', error);
    return null;
  }
} 