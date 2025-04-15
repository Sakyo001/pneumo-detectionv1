const { Client } = require('pg');

// Create a prompt to get user input for password
const readline = require('readline').createInterface({
  input: process.stdin,
  output: process.stdout
});

readline.question('Enter PostgreSQL password: ', (password) => {
  const client = new Client({
    user: 'postgres',
    host: 'localhost',
    database: 'postgres', // Using the default postgres database to test connection
    password: password,
    port: 5432,
  });

  client.connect()
    .then(() => {
      console.log('Successfully connected to PostgreSQL!');
      // List databases
      return client.query('SELECT datname FROM pg_database;');
    })
    .then(res => {
      console.log('Available databases:');
      res.rows.forEach(row => {
        console.log(`- ${row.datname}`);
      });
      client.end();
      readline.close();
    })
    .catch(err => {
      console.error('Connection error:', err);
      client.end();
      readline.close();
    });
}); 