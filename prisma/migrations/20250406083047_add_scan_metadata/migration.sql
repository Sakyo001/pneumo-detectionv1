-- CreateTable
CREATE TABLE "ScanMetadata" (
    "id" TEXT NOT NULL,
    "scanId" TEXT NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,
    "pneumoniaType" TEXT,
    "severity" TEXT,
    "recommendedAction" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ScanMetadata_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ScanMetadata_scanId_key" ON "ScanMetadata"("scanId");

-- AddForeignKey
ALTER TABLE "ScanMetadata" ADD CONSTRAINT "ScanMetadata_scanId_fkey" FOREIGN KEY ("scanId") REFERENCES "XrayScan"("id") ON DELETE CASCADE ON UPDATE CASCADE;
