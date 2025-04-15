/*
  Warnings:

  - We need to add a doctorId to existing XrayScan records, so we'll first add it as nullable, then update records, then make it required
*/
-- First, add the columns as nullable
ALTER TABLE "XrayScan" 
ADD COLUMN "analysis_result" TEXT,
ADD COLUMN "confidence_score" DOUBLE PRECISION,
ADD COLUMN "doctorId" TEXT,
ADD COLUMN "pneumonia_type" TEXT,
ADD COLUMN "recommended_action" TEXT,
ADD COLUMN "severity" TEXT;

-- Now update existing records to use the first DOCTOR user id
UPDATE "XrayScan"
SET "doctorId" = (SELECT id FROM "User" WHERE role = 'DOCTOR' LIMIT 1);

-- If no doctor exists, create a dummy one and update records
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM "XrayScan" WHERE "doctorId" IS NOT NULL LIMIT 1) THEN
        -- Insert a dummy doctor if needed
        INSERT INTO "User" (id, email, password, role, name, "createdAt", "updatedAt") 
        VALUES (
            '00000000-0000-0000-0000-000000000000', 
            'system@pneumodetect.com', 
            'temporary_password', 
            'DOCTOR',
            'System Doctor',
            NOW(),
            NOW()
        )
        ON CONFLICT DO NOTHING;
        
        -- Update records to use this dummy doctor
        UPDATE "XrayScan"
        SET "doctorId" = '00000000-0000-0000-0000-000000000000'
        WHERE "doctorId" IS NULL;
    END IF;
END $$;

-- Now make doctorId required
ALTER TABLE "XrayScan" 
ALTER COLUMN "doctorId" SET NOT NULL;

-- Add the foreign key constraint
ALTER TABLE "XrayScan" ADD CONSTRAINT "XrayScan_doctorId_fkey" FOREIGN KEY ("doctorId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
