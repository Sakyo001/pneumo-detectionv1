const { PrismaClient } = require('./generated/prisma');
const prisma = new PrismaClient();

async function main() {
  try {
    console.log('Seeding database...');
    
    // Count existing records
    const adminCount = await prisma.user.count({
      where: { role: 'ADMIN' }
    });

    const doctorCount = await prisma.user.count({
      where: { role: 'DOCTOR' }
    });

    const patientCount = await prisma.user.count({
      where: { role: 'PATIENT' }
    });
    
    console.log(`Existing records: ${adminCount} admins, ${doctorCount} doctors, ${patientCount} patients`);

    // Create admin if none exists
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

    // Create doctors if none exist
    if (doctorCount < 2) {
      // Create or find first doctor
      let doctor1 = await prisma.user.findFirst({
        where: { doctorId: 'DOC12345' }
      });
      
      if (!doctor1) {
        doctor1 = await prisma.user.create({
          data: {
            email: 'doctor@medrecord.com',
            doctorId: 'DOC12345',
            password: 'doctor123', // Remember to hash in production
            role: 'DOCTOR',
            name: 'Dr. Smith',
          },
        });
        console.log('Doctor 1 created');
      }
      
      // Create second doctor
      let doctor2 = await prisma.user.findFirst({
        where: { doctorId: 'DOC67890' }
      });
      
      if (!doctor2) {
        doctor2 = await prisma.user.create({
          data: {
            email: 'drsantos@medrecord.com',
            doctorId: 'DOC67890',
            password: 'doctor123', // Remember to hash in production
            role: 'DOCTOR',
            name: 'Dr. Santos',
          },
        });
        console.log('Doctor 2 created');
      }
    }
    
    // Create patients
    if (patientCount === 0) {
      // Create a patient user
      await prisma.user.create({
        data: {
          email: 'patient@example.com',
          password: 'patient123',
          role: 'PATIENT',
          name: 'John Doe',
        },
      });
      console.log('Patient user created');
    }
    
    // Get doctor IDs
    const doctor1 = await prisma.user.findFirst({ 
      where: { doctorId: 'DOC12345' } 
    });
    
    const doctor2 = await prisma.user.findFirst({ 
      where: { doctorId: 'DOC67890' } 
    });
    
    // Check if patients exist
    const patient1Count = await prisma.patient.count({
      where: { referenceNumber: 'PAT-001' }
    });
    
    const patient2Count = await prisma.patient.count({
      where: { referenceNumber: 'PAT-002' }
    });
    
    // Create patient records if they don't exist
    if (patient1Count === 0 && doctor1) {
      await prisma.patient.create({
        data: {
          name: 'John Doe',
          referenceNumber: 'PAT-001',
          dateOfBirth: new Date('1985-05-15'),
          doctorId: doctor1.id,
        }
      });
      console.log('Patient 1 record created');
    }
    
    if (patient2Count === 0 && doctor2) {
      await prisma.patient.create({
        data: {
          name: 'Maria Garcia',
          referenceNumber: 'PAT-002',
          dateOfBirth: new Date('1992-08-21'),
          doctorId: doctor2.id,
        }
      });
      console.log('Patient 2 record created');
    }
    
    console.log('Database seeding completed');
  } catch (error) {
    console.error('Error seeding database:', error);
  } finally {
    await prisma.$disconnect();
  }
}

main(); 