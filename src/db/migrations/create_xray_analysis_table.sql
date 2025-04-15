-- Create xray_analysis table
CREATE TABLE IF NOT EXISTS xray_analysis (
    id SERIAL PRIMARY KEY,
    reference_number VARCHAR(50) UNIQUE NOT NULL,
    image_url TEXT NOT NULL,
    analysis_result VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    doctor_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_xray_analysis_reference ON xray_analysis(reference_number); 