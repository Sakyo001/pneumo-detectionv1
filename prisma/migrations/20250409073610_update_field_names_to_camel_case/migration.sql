/*
  Warnings:

  - You are about to drop the column `analysis_result` on the `XrayScan` table. All the data in the column will be lost.
  - You are about to drop the column `confidence_score` on the `XrayScan` table. All the data in the column will be lost.
  - You are about to drop the column `pneumonia_type` on the `XrayScan` table. All the data in the column will be lost.
  - You are about to drop the column `recommended_action` on the `XrayScan` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "XrayScan" DROP COLUMN "analysis_result",
DROP COLUMN "confidence_score",
DROP COLUMN "pneumonia_type",
DROP COLUMN "recommended_action",
ADD COLUMN     "analysisResult" TEXT,
ADD COLUMN     "confidenceScore" DOUBLE PRECISION,
ADD COLUMN     "pneumoniaType" TEXT,
ADD COLUMN     "recommendedAction" TEXT;
