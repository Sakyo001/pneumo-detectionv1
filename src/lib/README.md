# Prisma Database Client Setup

This directory contains utility files for database interactions using Prisma.

## Standardized Imports

To ensure consistency across the codebase, please use the following import statements for Prisma:

```typescript
// When you need the Prisma client instance
import { prisma } from '@/lib/db';

// When you need Prisma types
import { YourType } from '@prisma/client';

// Example:
import { prisma } from '@/lib/db';
import { User, Patient } from '@prisma/client';
```

## Managing BigInt Values

When working with BigInt values (like IDs, counts, etc.) and sending data to the client, use the `serializeForJson` utility function from `@/lib/utils`:

```typescript
import { serializeForJson } from '@/lib/utils';

// Example:
return NextResponse.json(serializeForJson(data));
```

## Important Files

- `db.ts` - The main Prisma client configuration with singleton pattern
- `utils.ts` - Contains utilities for working with database results
- `auth.ts` - Authentication helpers that interact with the database

## Type Safety with Prisma

When working with relations, create interfaces that extend Prisma's types:

```typescript
import { XrayScan, Patient, ScanMetadata } from '@prisma/client';

interface XrayScanWithRelations extends XrayScan {
  patient: Patient;
  metadata: ScanMetadata | null;
}
```

This ensures type safety when working with related data. 