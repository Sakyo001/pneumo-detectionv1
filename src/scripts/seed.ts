import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1000);

  // Create a doctor with unique email
  const doctor = await prisma.user.create({
    data: {
      email: `doctor${timestamp}${random}@example.com`,
      password: 'hashed_password', // In production, this should be properly hashed
      role: 'DOCTOR',
      name: 'Dr. John Smith',
    },
  });

  // Create a patient with unique reference number
  const patient = await prisma.patient.create({
    data: {
      name: 'Jane Doe',
      referenceNumber: `PT-${timestamp}-${random}`,
      dateOfBirth: new Date('1990-01-01'),
      doctorId: doctor.id,
    },
  });

  // Create an X-ray scan with a unique reference number
  const xrayScan = await prisma.xrayScan.create({
    data: {
      referenceNumber: `XR-${timestamp}-${random}`,
      patientId: patient.id,
      doctorId: doctor.id,
      imageUrl: '/uploads/xray-sample.jpg',
      result: 'Pneumonia',
      status: 'COMPLETED',
    },
  });

  // Create scan metadata
  await prisma.scanMetadata.create({
    data: {
      scanId: xrayScan.id,
      confidence: 0.95,
      pneumoniaType: 'Bacterial',
      severity: 'Moderate',
      recommendedAction: 'Prescribe antibiotics and schedule follow-up in 2 weeks',
    },
  });

  console.log('Test data created successfully');
  console.log('Doctor Email:', doctor.email);
  console.log('Patient Reference:', patient.referenceNumber);
  console.log('X-Ray Reference:', xrayScan.referenceNumber);
}

main()
  .catch((e) => {
    console.error('Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  }); 