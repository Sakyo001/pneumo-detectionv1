import { PrismaClient } from '@prisma/client';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

const prisma = new PrismaClient();

export async function login(email: string, password: string, role: 'ADMIN' | 'DOCTOR') {
  try {
    console.log(`Attempting login for: ${email} with role ${role}`);
    
    // Find the user by email
    const user = await prisma.user.findFirst({
      where: {
        email,
      },
    });

    console.log('Found user:', user ? 'Yes' : 'No');
    
    if (!user) {
      return { success: false, message: 'User not found' };
    }
    
    // Check if user has the correct role
    if (user.role !== role) {
      console.log(`Role mismatch: expected ${role}, got ${user.role}`);
      return { success: false, message: 'Invalid credentials' };
    }

    // Check if password matches (in production, use bcrypt.compare)
    const isMatch = password === user.password;
    console.log('Password match:', isMatch);

    if (!isMatch) {
      return { success: false, message: 'Invalid credentials' };
    }

    // Set cookies
    const cookieStore = await cookies();
    
    cookieStore.set('userId', user.id, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      maxAge: 60 * 60 * 24 * 7, // 1 week
      path: '/',
    });

    cookieStore.set('userRole', user.role, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      maxAge: 60 * 60 * 24 * 7, // 1 week
      path: '/',
    });

    return { success: true, user };
  } catch (error) {
    console.error('Login error:', error);
    return { success: false, message: 'An error occurred during login' };
  }
}

export async function seedUsers() {
  try {
    // Check if users already exist
    const adminCount = await prisma.user.count({
      where: { role: 'ADMIN' }
    });

    const doctorCount = await prisma.user.count({
      where: { role: 'DOCTOR' }
    });

    const patientCount = await prisma.user.count({
      where: { role: 'PATIENT' }
    });

    if (adminCount === 0) {
      await prisma.user.create({
        data: {
          email: 'admin@medrecord.com',
          password: 'admin123', // Remember to hash in production
          role: 'ADMIN',
          name: 'Admin User',
        },
      });
      console.log('Admin user created');
    }

    if (doctorCount === 0) {
      // Create first doctor
      const doctor1 = await prisma.user.create({
        data: {
          email: 'doctor@medrecord.com',
          doctorId: 'DOC12345',
          password: 'doctor123', // Remember to hash in production
          role: 'DOCTOR',
          name: 'Dr. Smith',
        },
      });
      
      // Create second doctor
      const doctor2 = await prisma.user.create({
        data: {
          email: 'drsantos@medrecord.com',
          doctorId: 'DOC67890',
          password: 'doctor123', // Remember to hash in production
          role: 'DOCTOR',
          name: 'Dr. Santos',
        },
      });
      
      console.log('Doctor users created');
    }
    
    if (patientCount === 0) {
      // Create a patient
      const patient = await prisma.user.create({
        data: {
          email: 'patient@example.com',
          password: 'patient123',
          role: 'PATIENT',
          name: 'John Doe',
        },
      });
      
      // Create patient record linked to Dr. Smith
      await prisma.patient.create({
        data: {
          name: 'John Doe',
          referenceNumber: 'PAT-001',
          dateOfBirth: new Date('1985-05-15'),
          doctorId: (await prisma.user.findFirst({ 
            where: { doctorId: 'DOC12345' } 
          }))?.id || '',
        }
      });
      
      // Create another patient record linked to Dr. Santos
      await prisma.patient.create({
        data: {
          name: 'Maria Garcia',
          referenceNumber: 'PAT-002',
          dateOfBirth: new Date('1992-08-21'),
          doctorId: (await prisma.user.findFirst({ 
            where: { doctorId: 'DOC67890' } 
          }))?.id || '',
        }
      });
      
      console.log('Patient records created');
    }
  } catch (error) {
    console.error('Error seeding users:', error);
  }
}

export async function getUserFromCookie() {
  const cookieStore = await cookies();
  const userId = cookieStore.get('userId')?.value;
  
  if (!userId) {
    return null;
  }

  const user = await prisma.user.findUnique({
    where: { id: userId },
  });

  return user;
}

export async function requireAuth(role?: 'ADMIN' | 'DOCTOR') {
  const user = await getUserFromCookie();
  
  if (!user) {
    redirect('/auth/login');
  }

  if (role && user.role !== role) {
    // Redirect to the appropriate dashboard based on role
    if (user.role === 'ADMIN') {
      redirect('/dashboard/admin');
    } else if (user.role === 'DOCTOR') {
      redirect('/dashboard/doctor');
    } else {
      redirect('/dashboard/patient');
    }
  }

  return user;
}

export async function logout() {
  const cookieStore = await cookies();
  cookieStore.delete('userId');
  cookieStore.delete('userRole');
  redirect('/');
} 