const { Pool } = require('pg');
const fs = require('fs/promises');
const path = require('path');

const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'medrecordhub',
  password: 'R123456@',
  port: 5432,
});

async function runMigration() {
  try {
    // Read the SQL file
    const sqlFile = path.join(process.cwd(), 'src', 'db', 'migrations', 'create_xray_analysis_table.sql');
    const sql = await fs.readFile(sqlFile, 'utf8');

    // Execute the SQL
    await pool.query(sql);
    console.log('Migration completed successfully');
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

runMigration(); 