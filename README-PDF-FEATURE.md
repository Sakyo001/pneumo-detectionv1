# PDF Report Generation Feature

This feature allows patients to download PDF reports of their X-ray analysis results.

## How It Works

1. When a patient enters a reference number on the patient dashboard, the system fetches the analysis results from the database.
2. The patient can then click the "Download PDF Report" button to generate and download a PDF report.
3. The PDF report includes:
   - Patient information
   - X-ray image
   - Diagnosis details
   - Recommended actions
   - Doctor information

## Technical Implementation

### API Endpoints

- `GET /api/analysis/[referenceNumber]` - Fetches analysis results for a given reference number
- `POST /api/analysis/[referenceNumber]` - Generates and returns a PDF report for a given reference number

### PDF Generation

The PDF generation is handled by the `generatePDFReport` function in `src/lib/pdf-generator.ts`. This function:

1. Creates a new PDF document using PDFKit
2. Adds patient information, diagnosis details, and the X-ray image
3. Returns the PDF as a buffer

### Frontend Integration

The patient dashboard (`src/app/dashboard/patient/page.tsx`) includes:

1. A "Download PDF Report" button that appears when analysis results are displayed
2. A function to handle the PDF download process
3. Enhanced UI to display additional diagnosis details

## Dependencies

- `pdfkit` - For generating PDF documents
- `node-fetch` - For fetching images to include in the PDF

## Future Improvements

- Add the ability to email PDF reports directly to patients
- Include more detailed medical information in the PDF
- Add digital signatures for doctors
- Implement a PDF preview before download 