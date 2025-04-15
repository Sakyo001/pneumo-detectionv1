
/**
 * Client
**/

import * as runtime from './runtime/library.js';
import $Types = runtime.Types // general types
import $Public = runtime.Types.Public
import $Utils = runtime.Types.Utils
import $Extensions = runtime.Types.Extensions
import $Result = runtime.Types.Result

export type PrismaPromise<T> = $Public.PrismaPromise<T>


/**
 * Model User
 * 
 */
export type User = $Result.DefaultSelection<Prisma.$UserPayload>
/**
 * Model Patient
 * 
 */
export type Patient = $Result.DefaultSelection<Prisma.$PatientPayload>
/**
 * Model XrayScan
 * 
 */
export type XrayScan = $Result.DefaultSelection<Prisma.$XrayScanPayload>
/**
 * Model ScanMetadata
 * 
 */
export type ScanMetadata = $Result.DefaultSelection<Prisma.$ScanMetadataPayload>

/**
 * Enums
 */
export namespace $Enums {
  export const UserRole: {
  ADMIN: 'ADMIN',
  DOCTOR: 'DOCTOR',
  PATIENT: 'PATIENT'
};

export type UserRole = (typeof UserRole)[keyof typeof UserRole]

}

export type UserRole = $Enums.UserRole

export const UserRole: typeof $Enums.UserRole

/**
 * ##  Prisma Client ʲˢ
 *
 * Type-safe database client for TypeScript & Node.js
 * @example
 * ```
 * const prisma = new PrismaClient()
 * // Fetch zero or more Users
 * const users = await prisma.user.findMany()
 * ```
 *
 *
 * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
 */
export class PrismaClient<
  ClientOptions extends Prisma.PrismaClientOptions = Prisma.PrismaClientOptions,
  U = 'log' extends keyof ClientOptions ? ClientOptions['log'] extends Array<Prisma.LogLevel | Prisma.LogDefinition> ? Prisma.GetEvents<ClientOptions['log']> : never : never,
  ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs
> {
  [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['other'] }

    /**
   * ##  Prisma Client ʲˢ
   *
   * Type-safe database client for TypeScript & Node.js
   * @example
   * ```
   * const prisma = new PrismaClient()
   * // Fetch zero or more Users
   * const users = await prisma.user.findMany()
   * ```
   *
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
   */

  constructor(optionsArg ?: Prisma.Subset<ClientOptions, Prisma.PrismaClientOptions>);
  $on<V extends U>(eventType: V, callback: (event: V extends 'query' ? Prisma.QueryEvent : Prisma.LogEvent) => void): PrismaClient;

  /**
   * Connect with the database
   */
  $connect(): $Utils.JsPromise<void>;

  /**
   * Disconnect from the database
   */
  $disconnect(): $Utils.JsPromise<void>;

  /**
   * Add a middleware
   * @deprecated since 4.16.0. For new code, prefer client extensions instead.
   * @see https://pris.ly/d/extensions
   */
  $use(cb: Prisma.Middleware): void

/**
   * Executes a prepared raw query and returns the number of affected rows.
   * @example
   * ```
   * const result = await prisma.$executeRaw`UPDATE User SET cool = ${true} WHERE email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Executes a raw query and returns the number of affected rows.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$executeRawUnsafe('UPDATE User SET cool = $1 WHERE email = $2 ;', true, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Performs a prepared raw query and returns the `SELECT` data.
   * @example
   * ```
   * const result = await prisma.$queryRaw`SELECT * FROM User WHERE id = ${1} OR email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<T>;

  /**
   * Performs a raw query and returns the `SELECT` data.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$queryRawUnsafe('SELECT * FROM User WHERE id = $1 OR email = $2;', 1, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<T>;


  /**
   * Allows the running of a sequence of read/write operations that are guaranteed to either succeed or fail as a whole.
   * @example
   * ```
   * const [george, bob, alice] = await prisma.$transaction([
   *   prisma.user.create({ data: { name: 'George' } }),
   *   prisma.user.create({ data: { name: 'Bob' } }),
   *   prisma.user.create({ data: { name: 'Alice' } }),
   * ])
   * ```
   * 
   * Read more in our [docs](https://www.prisma.io/docs/concepts/components/prisma-client/transactions).
   */
  $transaction<P extends Prisma.PrismaPromise<any>[]>(arg: [...P], options?: { isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<runtime.Types.Utils.UnwrapTuple<P>>

  $transaction<R>(fn: (prisma: Omit<PrismaClient, runtime.ITXClientDenyList>) => $Utils.JsPromise<R>, options?: { maxWait?: number, timeout?: number, isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<R>


  $extends: $Extensions.ExtendsHook<"extends", Prisma.TypeMapCb<ClientOptions>, ExtArgs, $Utils.Call<Prisma.TypeMapCb<ClientOptions>, {
    extArgs: ExtArgs
  }>>

      /**
   * `prisma.user`: Exposes CRUD operations for the **User** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Users
    * const users = await prisma.user.findMany()
    * ```
    */
  get user(): Prisma.UserDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.patient`: Exposes CRUD operations for the **Patient** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Patients
    * const patients = await prisma.patient.findMany()
    * ```
    */
  get patient(): Prisma.PatientDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.xrayScan`: Exposes CRUD operations for the **XrayScan** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more XrayScans
    * const xrayScans = await prisma.xrayScan.findMany()
    * ```
    */
  get xrayScan(): Prisma.XrayScanDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.scanMetadata`: Exposes CRUD operations for the **ScanMetadata** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more ScanMetadata
    * const scanMetadata = await prisma.scanMetadata.findMany()
    * ```
    */
  get scanMetadata(): Prisma.ScanMetadataDelegate<ExtArgs, ClientOptions>;
}

export namespace Prisma {
  export import DMMF = runtime.DMMF

  export type PrismaPromise<T> = $Public.PrismaPromise<T>

  /**
   * Validator
   */
  export import validator = runtime.Public.validator

  /**
   * Prisma Errors
   */
  export import PrismaClientKnownRequestError = runtime.PrismaClientKnownRequestError
  export import PrismaClientUnknownRequestError = runtime.PrismaClientUnknownRequestError
  export import PrismaClientRustPanicError = runtime.PrismaClientRustPanicError
  export import PrismaClientInitializationError = runtime.PrismaClientInitializationError
  export import PrismaClientValidationError = runtime.PrismaClientValidationError

  /**
   * Re-export of sql-template-tag
   */
  export import sql = runtime.sqltag
  export import empty = runtime.empty
  export import join = runtime.join
  export import raw = runtime.raw
  export import Sql = runtime.Sql



  /**
   * Decimal.js
   */
  export import Decimal = runtime.Decimal

  export type DecimalJsLike = runtime.DecimalJsLike

  /**
   * Metrics
   */
  export type Metrics = runtime.Metrics
  export type Metric<T> = runtime.Metric<T>
  export type MetricHistogram = runtime.MetricHistogram
  export type MetricHistogramBucket = runtime.MetricHistogramBucket

  /**
  * Extensions
  */
  export import Extension = $Extensions.UserArgs
  export import getExtensionContext = runtime.Extensions.getExtensionContext
  export import Args = $Public.Args
  export import Payload = $Public.Payload
  export import Result = $Public.Result
  export import Exact = $Public.Exact

  /**
   * Prisma Client JS version: 6.5.0
   * Query Engine version: 173f8d54f8d52e692c7e27e72a88314ec7aeff60
   */
  export type PrismaVersion = {
    client: string
  }

  export const prismaVersion: PrismaVersion

  /**
   * Utility Types
   */


  export import JsonObject = runtime.JsonObject
  export import JsonArray = runtime.JsonArray
  export import JsonValue = runtime.JsonValue
  export import InputJsonObject = runtime.InputJsonObject
  export import InputJsonArray = runtime.InputJsonArray
  export import InputJsonValue = runtime.InputJsonValue

  /**
   * Types of the values used to represent different kinds of `null` values when working with JSON fields.
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  namespace NullTypes {
    /**
    * Type of `Prisma.DbNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.DbNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class DbNull {
      private DbNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.JsonNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.JsonNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class JsonNull {
      private JsonNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.AnyNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.AnyNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class AnyNull {
      private AnyNull: never
      private constructor()
    }
  }

  /**
   * Helper for filtering JSON entries that have `null` on the database (empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const DbNull: NullTypes.DbNull

  /**
   * Helper for filtering JSON entries that have JSON `null` values (not empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const JsonNull: NullTypes.JsonNull

  /**
   * Helper for filtering JSON entries that are `Prisma.DbNull` or `Prisma.JsonNull`
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const AnyNull: NullTypes.AnyNull

  type SelectAndInclude = {
    select: any
    include: any
  }

  type SelectAndOmit = {
    select: any
    omit: any
  }

  /**
   * Get the type of the value, that the Promise holds.
   */
  export type PromiseType<T extends PromiseLike<any>> = T extends PromiseLike<infer U> ? U : T;

  /**
   * Get the return type of a function which returns a Promise.
   */
  export type PromiseReturnType<T extends (...args: any) => $Utils.JsPromise<any>> = PromiseType<ReturnType<T>>

  /**
   * From T, pick a set of properties whose keys are in the union K
   */
  type Prisma__Pick<T, K extends keyof T> = {
      [P in K]: T[P];
  };


  export type Enumerable<T> = T | Array<T>;

  export type RequiredKeys<T> = {
    [K in keyof T]-?: {} extends Prisma__Pick<T, K> ? never : K
  }[keyof T]

  export type TruthyKeys<T> = keyof {
    [K in keyof T as T[K] extends false | undefined | null ? never : K]: K
  }

  export type TrueKeys<T> = TruthyKeys<Prisma__Pick<T, RequiredKeys<T>>>

  /**
   * Subset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection
   */
  export type Subset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never;
  };

  /**
   * SelectSubset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection.
   * Additionally, it validates, if both select and include are present. If the case, it errors.
   */
  export type SelectSubset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    (T extends SelectAndInclude
      ? 'Please either choose `select` or `include`.'
      : T extends SelectAndOmit
        ? 'Please either choose `select` or `omit`.'
        : {})

  /**
   * Subset + Intersection
   * @desc From `T` pick properties that exist in `U` and intersect `K`
   */
  export type SubsetIntersection<T, U, K> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    K

  type Without<T, U> = { [P in Exclude<keyof T, keyof U>]?: never };

  /**
   * XOR is needed to have a real mutually exclusive union type
   * https://stackoverflow.com/questions/42123407/does-typescript-support-mutually-exclusive-types
   */
  type XOR<T, U> =
    T extends object ?
    U extends object ?
      (Without<T, U> & U) | (Without<U, T> & T)
    : U : T


  /**
   * Is T a Record?
   */
  type IsObject<T extends any> = T extends Array<any>
  ? False
  : T extends Date
  ? False
  : T extends Uint8Array
  ? False
  : T extends BigInt
  ? False
  : T extends object
  ? True
  : False


  /**
   * If it's T[], return T
   */
  export type UnEnumerate<T extends unknown> = T extends Array<infer U> ? U : T

  /**
   * From ts-toolbelt
   */

  type __Either<O extends object, K extends Key> = Omit<O, K> &
    {
      // Merge all but K
      [P in K]: Prisma__Pick<O, P & keyof O> // With K possibilities
    }[K]

  type EitherStrict<O extends object, K extends Key> = Strict<__Either<O, K>>

  type EitherLoose<O extends object, K extends Key> = ComputeRaw<__Either<O, K>>

  type _Either<
    O extends object,
    K extends Key,
    strict extends Boolean
  > = {
    1: EitherStrict<O, K>
    0: EitherLoose<O, K>
  }[strict]

  type Either<
    O extends object,
    K extends Key,
    strict extends Boolean = 1
  > = O extends unknown ? _Either<O, K, strict> : never

  export type Union = any

  type PatchUndefined<O extends object, O1 extends object> = {
    [K in keyof O]: O[K] extends undefined ? At<O1, K> : O[K]
  } & {}

  /** Helper Types for "Merge" **/
  export type IntersectOf<U extends Union> = (
    U extends unknown ? (k: U) => void : never
  ) extends (k: infer I) => void
    ? I
    : never

  export type Overwrite<O extends object, O1 extends object> = {
      [K in keyof O]: K extends keyof O1 ? O1[K] : O[K];
  } & {};

  type _Merge<U extends object> = IntersectOf<Overwrite<U, {
      [K in keyof U]-?: At<U, K>;
  }>>;

  type Key = string | number | symbol;
  type AtBasic<O extends object, K extends Key> = K extends keyof O ? O[K] : never;
  type AtStrict<O extends object, K extends Key> = O[K & keyof O];
  type AtLoose<O extends object, K extends Key> = O extends unknown ? AtStrict<O, K> : never;
  export type At<O extends object, K extends Key, strict extends Boolean = 1> = {
      1: AtStrict<O, K>;
      0: AtLoose<O, K>;
  }[strict];

  export type ComputeRaw<A extends any> = A extends Function ? A : {
    [K in keyof A]: A[K];
  } & {};

  export type OptionalFlat<O> = {
    [K in keyof O]?: O[K];
  } & {};

  type _Record<K extends keyof any, T> = {
    [P in K]: T;
  };

  // cause typescript not to expand types and preserve names
  type NoExpand<T> = T extends unknown ? T : never;

  // this type assumes the passed object is entirely optional
  type AtLeast<O extends object, K extends string> = NoExpand<
    O extends unknown
    ? | (K extends keyof O ? { [P in K]: O[P] } & O : O)
      | {[P in keyof O as P extends K ? P : never]-?: O[P]} & O
    : never>;

  type _Strict<U, _U = U> = U extends unknown ? U & OptionalFlat<_Record<Exclude<Keys<_U>, keyof U>, never>> : never;

  export type Strict<U extends object> = ComputeRaw<_Strict<U>>;
  /** End Helper Types for "Merge" **/

  export type Merge<U extends object> = ComputeRaw<_Merge<Strict<U>>>;

  /**
  A [[Boolean]]
  */
  export type Boolean = True | False

  // /**
  // 1
  // */
  export type True = 1

  /**
  0
  */
  export type False = 0

  export type Not<B extends Boolean> = {
    0: 1
    1: 0
  }[B]

  export type Extends<A1 extends any, A2 extends any> = [A1] extends [never]
    ? 0 // anything `never` is false
    : A1 extends A2
    ? 1
    : 0

  export type Has<U extends Union, U1 extends Union> = Not<
    Extends<Exclude<U1, U>, U1>
  >

  export type Or<B1 extends Boolean, B2 extends Boolean> = {
    0: {
      0: 0
      1: 1
    }
    1: {
      0: 1
      1: 1
    }
  }[B1][B2]

  export type Keys<U extends Union> = U extends unknown ? keyof U : never

  type Cast<A, B> = A extends B ? A : B;

  export const type: unique symbol;



  /**
   * Used by group by
   */

  export type GetScalarType<T, O> = O extends object ? {
    [P in keyof T]: P extends keyof O
      ? O[P]
      : never
  } : never

  type FieldPaths<
    T,
    U = Omit<T, '_avg' | '_sum' | '_count' | '_min' | '_max'>
  > = IsObject<T> extends True ? U : T

  type GetHavingFields<T> = {
    [K in keyof T]: Or<
      Or<Extends<'OR', K>, Extends<'AND', K>>,
      Extends<'NOT', K>
    > extends True
      ? // infer is only needed to not hit TS limit
        // based on the brilliant idea of Pierre-Antoine Mills
        // https://github.com/microsoft/TypeScript/issues/30188#issuecomment-478938437
        T[K] extends infer TK
        ? GetHavingFields<UnEnumerate<TK> extends object ? Merge<UnEnumerate<TK>> : never>
        : never
      : {} extends FieldPaths<T[K]>
      ? never
      : K
  }[keyof T]

  /**
   * Convert tuple to union
   */
  type _TupleToUnion<T> = T extends (infer E)[] ? E : never
  type TupleToUnion<K extends readonly any[]> = _TupleToUnion<K>
  type MaybeTupleToUnion<T> = T extends any[] ? TupleToUnion<T> : T

  /**
   * Like `Pick`, but additionally can also accept an array of keys
   */
  type PickEnumerable<T, K extends Enumerable<keyof T> | keyof T> = Prisma__Pick<T, MaybeTupleToUnion<K>>

  /**
   * Exclude all keys with underscores
   */
  type ExcludeUnderscoreKeys<T extends string> = T extends `_${string}` ? never : T


  export type FieldRef<Model, FieldType> = runtime.FieldRef<Model, FieldType>

  type FieldRefInputType<Model, FieldType> = Model extends never ? never : FieldRef<Model, FieldType>


  export const ModelName: {
    User: 'User',
    Patient: 'Patient',
    XrayScan: 'XrayScan',
    ScanMetadata: 'ScanMetadata'
  };

  export type ModelName = (typeof ModelName)[keyof typeof ModelName]


  export type Datasources = {
    db?: Datasource
  }

  interface TypeMapCb<ClientOptions = {}> extends $Utils.Fn<{extArgs: $Extensions.InternalArgs }, $Utils.Record<string, any>> {
    returns: Prisma.TypeMap<this['params']['extArgs'], ClientOptions extends { omit: infer OmitOptions } ? OmitOptions : {}>
  }

  export type TypeMap<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> = {
    globalOmitOptions: {
      omit: GlobalOmitOptions
    }
    meta: {
      modelProps: "user" | "patient" | "xrayScan" | "scanMetadata"
      txIsolationLevel: Prisma.TransactionIsolationLevel
    }
    model: {
      User: {
        payload: Prisma.$UserPayload<ExtArgs>
        fields: Prisma.UserFieldRefs
        operations: {
          findUnique: {
            args: Prisma.UserFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.UserFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          findFirst: {
            args: Prisma.UserFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.UserFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          findMany: {
            args: Prisma.UserFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          create: {
            args: Prisma.UserCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          createMany: {
            args: Prisma.UserCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.UserCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          delete: {
            args: Prisma.UserDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          update: {
            args: Prisma.UserUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          deleteMany: {
            args: Prisma.UserDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.UserUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.UserUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          upsert: {
            args: Prisma.UserUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          aggregate: {
            args: Prisma.UserAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateUser>
          }
          groupBy: {
            args: Prisma.UserGroupByArgs<ExtArgs>
            result: $Utils.Optional<UserGroupByOutputType>[]
          }
          count: {
            args: Prisma.UserCountArgs<ExtArgs>
            result: $Utils.Optional<UserCountAggregateOutputType> | number
          }
        }
      }
      Patient: {
        payload: Prisma.$PatientPayload<ExtArgs>
        fields: Prisma.PatientFieldRefs
        operations: {
          findUnique: {
            args: Prisma.PatientFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.PatientFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>
          }
          findFirst: {
            args: Prisma.PatientFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.PatientFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>
          }
          findMany: {
            args: Prisma.PatientFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>[]
          }
          create: {
            args: Prisma.PatientCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>
          }
          createMany: {
            args: Prisma.PatientCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.PatientCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>[]
          }
          delete: {
            args: Prisma.PatientDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>
          }
          update: {
            args: Prisma.PatientUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>
          }
          deleteMany: {
            args: Prisma.PatientDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.PatientUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.PatientUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>[]
          }
          upsert: {
            args: Prisma.PatientUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$PatientPayload>
          }
          aggregate: {
            args: Prisma.PatientAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregatePatient>
          }
          groupBy: {
            args: Prisma.PatientGroupByArgs<ExtArgs>
            result: $Utils.Optional<PatientGroupByOutputType>[]
          }
          count: {
            args: Prisma.PatientCountArgs<ExtArgs>
            result: $Utils.Optional<PatientCountAggregateOutputType> | number
          }
        }
      }
      XrayScan: {
        payload: Prisma.$XrayScanPayload<ExtArgs>
        fields: Prisma.XrayScanFieldRefs
        operations: {
          findUnique: {
            args: Prisma.XrayScanFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.XrayScanFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>
          }
          findFirst: {
            args: Prisma.XrayScanFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.XrayScanFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>
          }
          findMany: {
            args: Prisma.XrayScanFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>[]
          }
          create: {
            args: Prisma.XrayScanCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>
          }
          createMany: {
            args: Prisma.XrayScanCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.XrayScanCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>[]
          }
          delete: {
            args: Prisma.XrayScanDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>
          }
          update: {
            args: Prisma.XrayScanUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>
          }
          deleteMany: {
            args: Prisma.XrayScanDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.XrayScanUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.XrayScanUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>[]
          }
          upsert: {
            args: Prisma.XrayScanUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$XrayScanPayload>
          }
          aggregate: {
            args: Prisma.XrayScanAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateXrayScan>
          }
          groupBy: {
            args: Prisma.XrayScanGroupByArgs<ExtArgs>
            result: $Utils.Optional<XrayScanGroupByOutputType>[]
          }
          count: {
            args: Prisma.XrayScanCountArgs<ExtArgs>
            result: $Utils.Optional<XrayScanCountAggregateOutputType> | number
          }
        }
      }
      ScanMetadata: {
        payload: Prisma.$ScanMetadataPayload<ExtArgs>
        fields: Prisma.ScanMetadataFieldRefs
        operations: {
          findUnique: {
            args: Prisma.ScanMetadataFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.ScanMetadataFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>
          }
          findFirst: {
            args: Prisma.ScanMetadataFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.ScanMetadataFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>
          }
          findMany: {
            args: Prisma.ScanMetadataFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>[]
          }
          create: {
            args: Prisma.ScanMetadataCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>
          }
          createMany: {
            args: Prisma.ScanMetadataCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.ScanMetadataCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>[]
          }
          delete: {
            args: Prisma.ScanMetadataDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>
          }
          update: {
            args: Prisma.ScanMetadataUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>
          }
          deleteMany: {
            args: Prisma.ScanMetadataDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.ScanMetadataUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.ScanMetadataUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>[]
          }
          upsert: {
            args: Prisma.ScanMetadataUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ScanMetadataPayload>
          }
          aggregate: {
            args: Prisma.ScanMetadataAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateScanMetadata>
          }
          groupBy: {
            args: Prisma.ScanMetadataGroupByArgs<ExtArgs>
            result: $Utils.Optional<ScanMetadataGroupByOutputType>[]
          }
          count: {
            args: Prisma.ScanMetadataCountArgs<ExtArgs>
            result: $Utils.Optional<ScanMetadataCountAggregateOutputType> | number
          }
        }
      }
    }
  } & {
    other: {
      payload: any
      operations: {
        $executeRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $executeRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
        $queryRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $queryRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
      }
    }
  }
  export const defineExtension: $Extensions.ExtendsHook<"define", Prisma.TypeMapCb, $Extensions.DefaultArgs>
  export type DefaultPrismaClient = PrismaClient
  export type ErrorFormat = 'pretty' | 'colorless' | 'minimal'
  export interface PrismaClientOptions {
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasources?: Datasources
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasourceUrl?: string
    /**
     * @default "colorless"
     */
    errorFormat?: ErrorFormat
    /**
     * @example
     * ```
     * // Defaults to stdout
     * log: ['query', 'info', 'warn', 'error']
     * 
     * // Emit as events
     * log: [
     *   { emit: 'stdout', level: 'query' },
     *   { emit: 'stdout', level: 'info' },
     *   { emit: 'stdout', level: 'warn' }
     *   { emit: 'stdout', level: 'error' }
     * ]
     * ```
     * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/logging#the-log-option).
     */
    log?: (LogLevel | LogDefinition)[]
    /**
     * The default values for transactionOptions
     * maxWait ?= 2000
     * timeout ?= 5000
     */
    transactionOptions?: {
      maxWait?: number
      timeout?: number
      isolationLevel?: Prisma.TransactionIsolationLevel
    }
    /**
     * Instance of a Driver Adapter, e.g., like one provided by `@prisma/adapter-planetscale`
     */
    adapter?: runtime.DriverAdapter | null
    /**
     * Global configuration for omitting model fields by default.
     * 
     * @example
     * ```
     * const prisma = new PrismaClient({
     *   omit: {
     *     user: {
     *       password: true
     *     }
     *   }
     * })
     * ```
     */
    omit?: Prisma.GlobalOmitConfig
  }
  export type GlobalOmitConfig = {
    user?: UserOmit
    patient?: PatientOmit
    xrayScan?: XrayScanOmit
    scanMetadata?: ScanMetadataOmit
  }

  /* Types for Logging */
  export type LogLevel = 'info' | 'query' | 'warn' | 'error'
  export type LogDefinition = {
    level: LogLevel
    emit: 'stdout' | 'event'
  }

  export type GetLogType<T extends LogLevel | LogDefinition> = T extends LogDefinition ? T['emit'] extends 'event' ? T['level'] : never : never
  export type GetEvents<T extends any> = T extends Array<LogLevel | LogDefinition> ?
    GetLogType<T[0]> | GetLogType<T[1]> | GetLogType<T[2]> | GetLogType<T[3]>
    : never

  export type QueryEvent = {
    timestamp: Date
    query: string
    params: string
    duration: number
    target: string
  }

  export type LogEvent = {
    timestamp: Date
    message: string
    target: string
  }
  /* End Types for Logging */


  export type PrismaAction =
    | 'findUnique'
    | 'findUniqueOrThrow'
    | 'findMany'
    | 'findFirst'
    | 'findFirstOrThrow'
    | 'create'
    | 'createMany'
    | 'createManyAndReturn'
    | 'update'
    | 'updateMany'
    | 'updateManyAndReturn'
    | 'upsert'
    | 'delete'
    | 'deleteMany'
    | 'executeRaw'
    | 'queryRaw'
    | 'aggregate'
    | 'count'
    | 'runCommandRaw'
    | 'findRaw'
    | 'groupBy'

  /**
   * These options are being passed into the middleware as "params"
   */
  export type MiddlewareParams = {
    model?: ModelName
    action: PrismaAction
    args: any
    dataPath: string[]
    runInTransaction: boolean
  }

  /**
   * The `T` type makes sure, that the `return proceed` is not forgotten in the middleware implementation
   */
  export type Middleware<T = any> = (
    params: MiddlewareParams,
    next: (params: MiddlewareParams) => $Utils.JsPromise<T>,
  ) => $Utils.JsPromise<T>

  // tested in getLogLevel.test.ts
  export function getLogLevel(log: Array<LogLevel | LogDefinition>): LogLevel | undefined;

  /**
   * `PrismaClient` proxy available in interactive transactions.
   */
  export type TransactionClient = Omit<Prisma.DefaultPrismaClient, runtime.ITXClientDenyList>

  export type Datasource = {
    url?: string
  }

  /**
   * Count Types
   */


  /**
   * Count Type UserCountOutputType
   */

  export type UserCountOutputType = {
    patients: number
  }

  export type UserCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    patients?: boolean | UserCountOutputTypeCountPatientsArgs
  }

  // Custom InputTypes
  /**
   * UserCountOutputType without action
   */
  export type UserCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserCountOutputType
     */
    select?: UserCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * UserCountOutputType without action
   */
  export type UserCountOutputTypeCountPatientsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: PatientWhereInput
  }


  /**
   * Count Type PatientCountOutputType
   */

  export type PatientCountOutputType = {
    xrayScans: number
  }

  export type PatientCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    xrayScans?: boolean | PatientCountOutputTypeCountXrayScansArgs
  }

  // Custom InputTypes
  /**
   * PatientCountOutputType without action
   */
  export type PatientCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the PatientCountOutputType
     */
    select?: PatientCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * PatientCountOutputType without action
   */
  export type PatientCountOutputTypeCountXrayScansArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: XrayScanWhereInput
  }


  /**
   * Models
   */

  /**
   * Model User
   */

  export type AggregateUser = {
    _count: UserCountAggregateOutputType | null
    _min: UserMinAggregateOutputType | null
    _max: UserMaxAggregateOutputType | null
  }

  export type UserMinAggregateOutputType = {
    id: string | null
    email: string | null
    doctorId: string | null
    password: string | null
    role: $Enums.UserRole | null
    name: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type UserMaxAggregateOutputType = {
    id: string | null
    email: string | null
    doctorId: string | null
    password: string | null
    role: $Enums.UserRole | null
    name: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type UserCountAggregateOutputType = {
    id: number
    email: number
    doctorId: number
    password: number
    role: number
    name: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type UserMinAggregateInputType = {
    id?: true
    email?: true
    doctorId?: true
    password?: true
    role?: true
    name?: true
    createdAt?: true
    updatedAt?: true
  }

  export type UserMaxAggregateInputType = {
    id?: true
    email?: true
    doctorId?: true
    password?: true
    role?: true
    name?: true
    createdAt?: true
    updatedAt?: true
  }

  export type UserCountAggregateInputType = {
    id?: true
    email?: true
    doctorId?: true
    password?: true
    role?: true
    name?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type UserAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which User to aggregate.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Users
    **/
    _count?: true | UserCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: UserMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: UserMaxAggregateInputType
  }

  export type GetUserAggregateType<T extends UserAggregateArgs> = {
        [P in keyof T & keyof AggregateUser]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateUser[P]>
      : GetScalarType<T[P], AggregateUser[P]>
  }




  export type UserGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: UserWhereInput
    orderBy?: UserOrderByWithAggregationInput | UserOrderByWithAggregationInput[]
    by: UserScalarFieldEnum[] | UserScalarFieldEnum
    having?: UserScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: UserCountAggregateInputType | true
    _min?: UserMinAggregateInputType
    _max?: UserMaxAggregateInputType
  }

  export type UserGroupByOutputType = {
    id: string
    email: string
    doctorId: string | null
    password: string
    role: $Enums.UserRole
    name: string
    createdAt: Date
    updatedAt: Date
    _count: UserCountAggregateOutputType | null
    _min: UserMinAggregateOutputType | null
    _max: UserMaxAggregateOutputType | null
  }

  type GetUserGroupByPayload<T extends UserGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<UserGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof UserGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], UserGroupByOutputType[P]>
            : GetScalarType<T[P], UserGroupByOutputType[P]>
        }
      >
    >


  export type UserSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    doctorId?: boolean
    password?: boolean
    role?: boolean
    name?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    patients?: boolean | User$patientsArgs<ExtArgs>
    _count?: boolean | UserCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["user"]>

  export type UserSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    doctorId?: boolean
    password?: boolean
    role?: boolean
    name?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["user"]>

  export type UserSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    doctorId?: boolean
    password?: boolean
    role?: boolean
    name?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["user"]>

  export type UserSelectScalar = {
    id?: boolean
    email?: boolean
    doctorId?: boolean
    password?: boolean
    role?: boolean
    name?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type UserOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "email" | "doctorId" | "password" | "role" | "name" | "createdAt" | "updatedAt", ExtArgs["result"]["user"]>
  export type UserInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    patients?: boolean | User$patientsArgs<ExtArgs>
    _count?: boolean | UserCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type UserIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}
  export type UserIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}

  export type $UserPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "User"
    objects: {
      patients: Prisma.$PatientPayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      email: string
      doctorId: string | null
      password: string
      role: $Enums.UserRole
      name: string
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["user"]>
    composites: {}
  }

  type UserGetPayload<S extends boolean | null | undefined | UserDefaultArgs> = $Result.GetResult<Prisma.$UserPayload, S>

  type UserCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<UserFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: UserCountAggregateInputType | true
    }

  export interface UserDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['User'], meta: { name: 'User' } }
    /**
     * Find zero or one User that matches the filter.
     * @param {UserFindUniqueArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends UserFindUniqueArgs>(args: SelectSubset<T, UserFindUniqueArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one User that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {UserFindUniqueOrThrowArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends UserFindUniqueOrThrowArgs>(args: SelectSubset<T, UserFindUniqueOrThrowArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first User that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindFirstArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends UserFindFirstArgs>(args?: SelectSubset<T, UserFindFirstArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first User that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindFirstOrThrowArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends UserFindFirstOrThrowArgs>(args?: SelectSubset<T, UserFindFirstOrThrowArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Users that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Users
     * const users = await prisma.user.findMany()
     * 
     * // Get first 10 Users
     * const users = await prisma.user.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const userWithIdOnly = await prisma.user.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends UserFindManyArgs>(args?: SelectSubset<T, UserFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a User.
     * @param {UserCreateArgs} args - Arguments to create a User.
     * @example
     * // Create one User
     * const User = await prisma.user.create({
     *   data: {
     *     // ... data to create a User
     *   }
     * })
     * 
     */
    create<T extends UserCreateArgs>(args: SelectSubset<T, UserCreateArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Users.
     * @param {UserCreateManyArgs} args - Arguments to create many Users.
     * @example
     * // Create many Users
     * const user = await prisma.user.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends UserCreateManyArgs>(args?: SelectSubset<T, UserCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Users and returns the data saved in the database.
     * @param {UserCreateManyAndReturnArgs} args - Arguments to create many Users.
     * @example
     * // Create many Users
     * const user = await prisma.user.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Users and only return the `id`
     * const userWithIdOnly = await prisma.user.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends UserCreateManyAndReturnArgs>(args?: SelectSubset<T, UserCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a User.
     * @param {UserDeleteArgs} args - Arguments to delete one User.
     * @example
     * // Delete one User
     * const User = await prisma.user.delete({
     *   where: {
     *     // ... filter to delete one User
     *   }
     * })
     * 
     */
    delete<T extends UserDeleteArgs>(args: SelectSubset<T, UserDeleteArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one User.
     * @param {UserUpdateArgs} args - Arguments to update one User.
     * @example
     * // Update one User
     * const user = await prisma.user.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends UserUpdateArgs>(args: SelectSubset<T, UserUpdateArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Users.
     * @param {UserDeleteManyArgs} args - Arguments to filter Users to delete.
     * @example
     * // Delete a few Users
     * const { count } = await prisma.user.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends UserDeleteManyArgs>(args?: SelectSubset<T, UserDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Users.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Users
     * const user = await prisma.user.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends UserUpdateManyArgs>(args: SelectSubset<T, UserUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Users and returns the data updated in the database.
     * @param {UserUpdateManyAndReturnArgs} args - Arguments to update many Users.
     * @example
     * // Update many Users
     * const user = await prisma.user.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more Users and only return the `id`
     * const userWithIdOnly = await prisma.user.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends UserUpdateManyAndReturnArgs>(args: SelectSubset<T, UserUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one User.
     * @param {UserUpsertArgs} args - Arguments to update or create a User.
     * @example
     * // Update or create a User
     * const user = await prisma.user.upsert({
     *   create: {
     *     // ... data to create a User
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the User we want to update
     *   }
     * })
     */
    upsert<T extends UserUpsertArgs>(args: SelectSubset<T, UserUpsertArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Users.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserCountArgs} args - Arguments to filter Users to count.
     * @example
     * // Count the number of Users
     * const count = await prisma.user.count({
     *   where: {
     *     // ... the filter for the Users we want to count
     *   }
     * })
    **/
    count<T extends UserCountArgs>(
      args?: Subset<T, UserCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], UserCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a User.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends UserAggregateArgs>(args: Subset<T, UserAggregateArgs>): Prisma.PrismaPromise<GetUserAggregateType<T>>

    /**
     * Group by User.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends UserGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: UserGroupByArgs['orderBy'] }
        : { orderBy?: UserGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, UserGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetUserGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the User model
   */
  readonly fields: UserFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for User.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__UserClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    patients<T extends User$patientsArgs<ExtArgs> = {}>(args?: Subset<T, User$patientsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the User model
   */ 
  interface UserFieldRefs {
    readonly id: FieldRef<"User", 'String'>
    readonly email: FieldRef<"User", 'String'>
    readonly doctorId: FieldRef<"User", 'String'>
    readonly password: FieldRef<"User", 'String'>
    readonly role: FieldRef<"User", 'UserRole'>
    readonly name: FieldRef<"User", 'String'>
    readonly createdAt: FieldRef<"User", 'DateTime'>
    readonly updatedAt: FieldRef<"User", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * User findUnique
   */
  export type UserFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User findUniqueOrThrow
   */
  export type UserFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User findFirst
   */
  export type UserFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Users.
     */
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User findFirstOrThrow
   */
  export type UserFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Users.
     */
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User findMany
   */
  export type UserFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter, which Users to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User create
   */
  export type UserCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * The data needed to create a User.
     */
    data: XOR<UserCreateInput, UserUncheckedCreateInput>
  }

  /**
   * User createMany
   */
  export type UserCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Users.
     */
    data: UserCreateManyInput | UserCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * User createManyAndReturn
   */
  export type UserCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data used to create many Users.
     */
    data: UserCreateManyInput | UserCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * User update
   */
  export type UserUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * The data needed to update a User.
     */
    data: XOR<UserUpdateInput, UserUncheckedUpdateInput>
    /**
     * Choose, which User to update.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User updateMany
   */
  export type UserUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Users.
     */
    data: XOR<UserUpdateManyMutationInput, UserUncheckedUpdateManyInput>
    /**
     * Filter which Users to update
     */
    where?: UserWhereInput
    /**
     * Limit how many Users to update.
     */
    limit?: number
  }

  /**
   * User updateManyAndReturn
   */
  export type UserUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data used to update Users.
     */
    data: XOR<UserUpdateManyMutationInput, UserUncheckedUpdateManyInput>
    /**
     * Filter which Users to update
     */
    where?: UserWhereInput
    /**
     * Limit how many Users to update.
     */
    limit?: number
  }

  /**
   * User upsert
   */
  export type UserUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * The filter to search for the User to update in case it exists.
     */
    where: UserWhereUniqueInput
    /**
     * In case the User found by the `where` argument doesn't exist, create a new User with this data.
     */
    create: XOR<UserCreateInput, UserUncheckedCreateInput>
    /**
     * In case the User was found with the provided `where` argument, update it with this data.
     */
    update: XOR<UserUpdateInput, UserUncheckedUpdateInput>
  }

  /**
   * User delete
   */
  export type UserDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
    /**
     * Filter which User to delete.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User deleteMany
   */
  export type UserDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Users to delete
     */
    where?: UserWhereInput
    /**
     * Limit how many Users to delete.
     */
    limit?: number
  }

  /**
   * User.patients
   */
  export type User$patientsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    where?: PatientWhereInput
    orderBy?: PatientOrderByWithRelationInput | PatientOrderByWithRelationInput[]
    cursor?: PatientWhereUniqueInput
    take?: number
    skip?: number
    distinct?: PatientScalarFieldEnum | PatientScalarFieldEnum[]
  }

  /**
   * User without action
   */
  export type UserDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: UserInclude<ExtArgs> | null
  }


  /**
   * Model Patient
   */

  export type AggregatePatient = {
    _count: PatientCountAggregateOutputType | null
    _min: PatientMinAggregateOutputType | null
    _max: PatientMaxAggregateOutputType | null
  }

  export type PatientMinAggregateOutputType = {
    id: string | null
    name: string | null
    referenceNumber: string | null
    dateOfBirth: Date | null
    doctorId: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type PatientMaxAggregateOutputType = {
    id: string | null
    name: string | null
    referenceNumber: string | null
    dateOfBirth: Date | null
    doctorId: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type PatientCountAggregateOutputType = {
    id: number
    name: number
    referenceNumber: number
    dateOfBirth: number
    doctorId: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type PatientMinAggregateInputType = {
    id?: true
    name?: true
    referenceNumber?: true
    dateOfBirth?: true
    doctorId?: true
    createdAt?: true
    updatedAt?: true
  }

  export type PatientMaxAggregateInputType = {
    id?: true
    name?: true
    referenceNumber?: true
    dateOfBirth?: true
    doctorId?: true
    createdAt?: true
    updatedAt?: true
  }

  export type PatientCountAggregateInputType = {
    id?: true
    name?: true
    referenceNumber?: true
    dateOfBirth?: true
    doctorId?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type PatientAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Patient to aggregate.
     */
    where?: PatientWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Patients to fetch.
     */
    orderBy?: PatientOrderByWithRelationInput | PatientOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: PatientWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Patients from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Patients.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Patients
    **/
    _count?: true | PatientCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: PatientMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: PatientMaxAggregateInputType
  }

  export type GetPatientAggregateType<T extends PatientAggregateArgs> = {
        [P in keyof T & keyof AggregatePatient]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregatePatient[P]>
      : GetScalarType<T[P], AggregatePatient[P]>
  }




  export type PatientGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: PatientWhereInput
    orderBy?: PatientOrderByWithAggregationInput | PatientOrderByWithAggregationInput[]
    by: PatientScalarFieldEnum[] | PatientScalarFieldEnum
    having?: PatientScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: PatientCountAggregateInputType | true
    _min?: PatientMinAggregateInputType
    _max?: PatientMaxAggregateInputType
  }

  export type PatientGroupByOutputType = {
    id: string
    name: string
    referenceNumber: string
    dateOfBirth: Date
    doctorId: string
    createdAt: Date
    updatedAt: Date
    _count: PatientCountAggregateOutputType | null
    _min: PatientMinAggregateOutputType | null
    _max: PatientMaxAggregateOutputType | null
  }

  type GetPatientGroupByPayload<T extends PatientGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<PatientGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof PatientGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], PatientGroupByOutputType[P]>
            : GetScalarType<T[P], PatientGroupByOutputType[P]>
        }
      >
    >


  export type PatientSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    referenceNumber?: boolean
    dateOfBirth?: boolean
    doctorId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    doctor?: boolean | UserDefaultArgs<ExtArgs>
    xrayScans?: boolean | Patient$xrayScansArgs<ExtArgs>
    _count?: boolean | PatientCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["patient"]>

  export type PatientSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    referenceNumber?: boolean
    dateOfBirth?: boolean
    doctorId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    doctor?: boolean | UserDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["patient"]>

  export type PatientSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    referenceNumber?: boolean
    dateOfBirth?: boolean
    doctorId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    doctor?: boolean | UserDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["patient"]>

  export type PatientSelectScalar = {
    id?: boolean
    name?: boolean
    referenceNumber?: boolean
    dateOfBirth?: boolean
    doctorId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type PatientOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "name" | "referenceNumber" | "dateOfBirth" | "doctorId" | "createdAt" | "updatedAt", ExtArgs["result"]["patient"]>
  export type PatientInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    doctor?: boolean | UserDefaultArgs<ExtArgs>
    xrayScans?: boolean | Patient$xrayScansArgs<ExtArgs>
    _count?: boolean | PatientCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type PatientIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    doctor?: boolean | UserDefaultArgs<ExtArgs>
  }
  export type PatientIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    doctor?: boolean | UserDefaultArgs<ExtArgs>
  }

  export type $PatientPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "Patient"
    objects: {
      doctor: Prisma.$UserPayload<ExtArgs>
      xrayScans: Prisma.$XrayScanPayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      name: string
      referenceNumber: string
      dateOfBirth: Date
      doctorId: string
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["patient"]>
    composites: {}
  }

  type PatientGetPayload<S extends boolean | null | undefined | PatientDefaultArgs> = $Result.GetResult<Prisma.$PatientPayload, S>

  type PatientCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<PatientFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: PatientCountAggregateInputType | true
    }

  export interface PatientDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['Patient'], meta: { name: 'Patient' } }
    /**
     * Find zero or one Patient that matches the filter.
     * @param {PatientFindUniqueArgs} args - Arguments to find a Patient
     * @example
     * // Get one Patient
     * const patient = await prisma.patient.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends PatientFindUniqueArgs>(args: SelectSubset<T, PatientFindUniqueArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one Patient that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {PatientFindUniqueOrThrowArgs} args - Arguments to find a Patient
     * @example
     * // Get one Patient
     * const patient = await prisma.patient.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends PatientFindUniqueOrThrowArgs>(args: SelectSubset<T, PatientFindUniqueOrThrowArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Patient that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientFindFirstArgs} args - Arguments to find a Patient
     * @example
     * // Get one Patient
     * const patient = await prisma.patient.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends PatientFindFirstArgs>(args?: SelectSubset<T, PatientFindFirstArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Patient that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientFindFirstOrThrowArgs} args - Arguments to find a Patient
     * @example
     * // Get one Patient
     * const patient = await prisma.patient.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends PatientFindFirstOrThrowArgs>(args?: SelectSubset<T, PatientFindFirstOrThrowArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Patients that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Patients
     * const patients = await prisma.patient.findMany()
     * 
     * // Get first 10 Patients
     * const patients = await prisma.patient.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const patientWithIdOnly = await prisma.patient.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends PatientFindManyArgs>(args?: SelectSubset<T, PatientFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a Patient.
     * @param {PatientCreateArgs} args - Arguments to create a Patient.
     * @example
     * // Create one Patient
     * const Patient = await prisma.patient.create({
     *   data: {
     *     // ... data to create a Patient
     *   }
     * })
     * 
     */
    create<T extends PatientCreateArgs>(args: SelectSubset<T, PatientCreateArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Patients.
     * @param {PatientCreateManyArgs} args - Arguments to create many Patients.
     * @example
     * // Create many Patients
     * const patient = await prisma.patient.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends PatientCreateManyArgs>(args?: SelectSubset<T, PatientCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Patients and returns the data saved in the database.
     * @param {PatientCreateManyAndReturnArgs} args - Arguments to create many Patients.
     * @example
     * // Create many Patients
     * const patient = await prisma.patient.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Patients and only return the `id`
     * const patientWithIdOnly = await prisma.patient.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends PatientCreateManyAndReturnArgs>(args?: SelectSubset<T, PatientCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a Patient.
     * @param {PatientDeleteArgs} args - Arguments to delete one Patient.
     * @example
     * // Delete one Patient
     * const Patient = await prisma.patient.delete({
     *   where: {
     *     // ... filter to delete one Patient
     *   }
     * })
     * 
     */
    delete<T extends PatientDeleteArgs>(args: SelectSubset<T, PatientDeleteArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one Patient.
     * @param {PatientUpdateArgs} args - Arguments to update one Patient.
     * @example
     * // Update one Patient
     * const patient = await prisma.patient.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends PatientUpdateArgs>(args: SelectSubset<T, PatientUpdateArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Patients.
     * @param {PatientDeleteManyArgs} args - Arguments to filter Patients to delete.
     * @example
     * // Delete a few Patients
     * const { count } = await prisma.patient.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends PatientDeleteManyArgs>(args?: SelectSubset<T, PatientDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Patients.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Patients
     * const patient = await prisma.patient.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends PatientUpdateManyArgs>(args: SelectSubset<T, PatientUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Patients and returns the data updated in the database.
     * @param {PatientUpdateManyAndReturnArgs} args - Arguments to update many Patients.
     * @example
     * // Update many Patients
     * const patient = await prisma.patient.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more Patients and only return the `id`
     * const patientWithIdOnly = await prisma.patient.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends PatientUpdateManyAndReturnArgs>(args: SelectSubset<T, PatientUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one Patient.
     * @param {PatientUpsertArgs} args - Arguments to update or create a Patient.
     * @example
     * // Update or create a Patient
     * const patient = await prisma.patient.upsert({
     *   create: {
     *     // ... data to create a Patient
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the Patient we want to update
     *   }
     * })
     */
    upsert<T extends PatientUpsertArgs>(args: SelectSubset<T, PatientUpsertArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Patients.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientCountArgs} args - Arguments to filter Patients to count.
     * @example
     * // Count the number of Patients
     * const count = await prisma.patient.count({
     *   where: {
     *     // ... the filter for the Patients we want to count
     *   }
     * })
    **/
    count<T extends PatientCountArgs>(
      args?: Subset<T, PatientCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], PatientCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a Patient.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends PatientAggregateArgs>(args: Subset<T, PatientAggregateArgs>): Prisma.PrismaPromise<GetPatientAggregateType<T>>

    /**
     * Group by Patient.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {PatientGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends PatientGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: PatientGroupByArgs['orderBy'] }
        : { orderBy?: PatientGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, PatientGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetPatientGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the Patient model
   */
  readonly fields: PatientFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for Patient.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__PatientClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    doctor<T extends UserDefaultArgs<ExtArgs> = {}>(args?: Subset<T, UserDefaultArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    xrayScans<T extends Patient$xrayScansArgs<ExtArgs> = {}>(args?: Subset<T, Patient$xrayScansArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the Patient model
   */ 
  interface PatientFieldRefs {
    readonly id: FieldRef<"Patient", 'String'>
    readonly name: FieldRef<"Patient", 'String'>
    readonly referenceNumber: FieldRef<"Patient", 'String'>
    readonly dateOfBirth: FieldRef<"Patient", 'DateTime'>
    readonly doctorId: FieldRef<"Patient", 'String'>
    readonly createdAt: FieldRef<"Patient", 'DateTime'>
    readonly updatedAt: FieldRef<"Patient", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * Patient findUnique
   */
  export type PatientFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * Filter, which Patient to fetch.
     */
    where: PatientWhereUniqueInput
  }

  /**
   * Patient findUniqueOrThrow
   */
  export type PatientFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * Filter, which Patient to fetch.
     */
    where: PatientWhereUniqueInput
  }

  /**
   * Patient findFirst
   */
  export type PatientFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * Filter, which Patient to fetch.
     */
    where?: PatientWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Patients to fetch.
     */
    orderBy?: PatientOrderByWithRelationInput | PatientOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Patients.
     */
    cursor?: PatientWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Patients from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Patients.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Patients.
     */
    distinct?: PatientScalarFieldEnum | PatientScalarFieldEnum[]
  }

  /**
   * Patient findFirstOrThrow
   */
  export type PatientFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * Filter, which Patient to fetch.
     */
    where?: PatientWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Patients to fetch.
     */
    orderBy?: PatientOrderByWithRelationInput | PatientOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Patients.
     */
    cursor?: PatientWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Patients from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Patients.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Patients.
     */
    distinct?: PatientScalarFieldEnum | PatientScalarFieldEnum[]
  }

  /**
   * Patient findMany
   */
  export type PatientFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * Filter, which Patients to fetch.
     */
    where?: PatientWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Patients to fetch.
     */
    orderBy?: PatientOrderByWithRelationInput | PatientOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Patients.
     */
    cursor?: PatientWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Patients from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Patients.
     */
    skip?: number
    distinct?: PatientScalarFieldEnum | PatientScalarFieldEnum[]
  }

  /**
   * Patient create
   */
  export type PatientCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * The data needed to create a Patient.
     */
    data: XOR<PatientCreateInput, PatientUncheckedCreateInput>
  }

  /**
   * Patient createMany
   */
  export type PatientCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Patients.
     */
    data: PatientCreateManyInput | PatientCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * Patient createManyAndReturn
   */
  export type PatientCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * The data used to create many Patients.
     */
    data: PatientCreateManyInput | PatientCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * Patient update
   */
  export type PatientUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * The data needed to update a Patient.
     */
    data: XOR<PatientUpdateInput, PatientUncheckedUpdateInput>
    /**
     * Choose, which Patient to update.
     */
    where: PatientWhereUniqueInput
  }

  /**
   * Patient updateMany
   */
  export type PatientUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Patients.
     */
    data: XOR<PatientUpdateManyMutationInput, PatientUncheckedUpdateManyInput>
    /**
     * Filter which Patients to update
     */
    where?: PatientWhereInput
    /**
     * Limit how many Patients to update.
     */
    limit?: number
  }

  /**
   * Patient updateManyAndReturn
   */
  export type PatientUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * The data used to update Patients.
     */
    data: XOR<PatientUpdateManyMutationInput, PatientUncheckedUpdateManyInput>
    /**
     * Filter which Patients to update
     */
    where?: PatientWhereInput
    /**
     * Limit how many Patients to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * Patient upsert
   */
  export type PatientUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * The filter to search for the Patient to update in case it exists.
     */
    where: PatientWhereUniqueInput
    /**
     * In case the Patient found by the `where` argument doesn't exist, create a new Patient with this data.
     */
    create: XOR<PatientCreateInput, PatientUncheckedCreateInput>
    /**
     * In case the Patient was found with the provided `where` argument, update it with this data.
     */
    update: XOR<PatientUpdateInput, PatientUncheckedUpdateInput>
  }

  /**
   * Patient delete
   */
  export type PatientDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
    /**
     * Filter which Patient to delete.
     */
    where: PatientWhereUniqueInput
  }

  /**
   * Patient deleteMany
   */
  export type PatientDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Patients to delete
     */
    where?: PatientWhereInput
    /**
     * Limit how many Patients to delete.
     */
    limit?: number
  }

  /**
   * Patient.xrayScans
   */
  export type Patient$xrayScansArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    where?: XrayScanWhereInput
    orderBy?: XrayScanOrderByWithRelationInput | XrayScanOrderByWithRelationInput[]
    cursor?: XrayScanWhereUniqueInput
    take?: number
    skip?: number
    distinct?: XrayScanScalarFieldEnum | XrayScanScalarFieldEnum[]
  }

  /**
   * Patient without action
   */
  export type PatientDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Patient
     */
    select?: PatientSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Patient
     */
    omit?: PatientOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: PatientInclude<ExtArgs> | null
  }


  /**
   * Model XrayScan
   */

  export type AggregateXrayScan = {
    _count: XrayScanCountAggregateOutputType | null
    _min: XrayScanMinAggregateOutputType | null
    _max: XrayScanMaxAggregateOutputType | null
  }

  export type XrayScanMinAggregateOutputType = {
    id: string | null
    referenceNumber: string | null
    patientId: string | null
    imageUrl: string | null
    result: string | null
    status: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type XrayScanMaxAggregateOutputType = {
    id: string | null
    referenceNumber: string | null
    patientId: string | null
    imageUrl: string | null
    result: string | null
    status: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type XrayScanCountAggregateOutputType = {
    id: number
    referenceNumber: number
    patientId: number
    imageUrl: number
    result: number
    status: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type XrayScanMinAggregateInputType = {
    id?: true
    referenceNumber?: true
    patientId?: true
    imageUrl?: true
    result?: true
    status?: true
    createdAt?: true
    updatedAt?: true
  }

  export type XrayScanMaxAggregateInputType = {
    id?: true
    referenceNumber?: true
    patientId?: true
    imageUrl?: true
    result?: true
    status?: true
    createdAt?: true
    updatedAt?: true
  }

  export type XrayScanCountAggregateInputType = {
    id?: true
    referenceNumber?: true
    patientId?: true
    imageUrl?: true
    result?: true
    status?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type XrayScanAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which XrayScan to aggregate.
     */
    where?: XrayScanWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of XrayScans to fetch.
     */
    orderBy?: XrayScanOrderByWithRelationInput | XrayScanOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: XrayScanWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` XrayScans from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` XrayScans.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned XrayScans
    **/
    _count?: true | XrayScanCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: XrayScanMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: XrayScanMaxAggregateInputType
  }

  export type GetXrayScanAggregateType<T extends XrayScanAggregateArgs> = {
        [P in keyof T & keyof AggregateXrayScan]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateXrayScan[P]>
      : GetScalarType<T[P], AggregateXrayScan[P]>
  }




  export type XrayScanGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: XrayScanWhereInput
    orderBy?: XrayScanOrderByWithAggregationInput | XrayScanOrderByWithAggregationInput[]
    by: XrayScanScalarFieldEnum[] | XrayScanScalarFieldEnum
    having?: XrayScanScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: XrayScanCountAggregateInputType | true
    _min?: XrayScanMinAggregateInputType
    _max?: XrayScanMaxAggregateInputType
  }

  export type XrayScanGroupByOutputType = {
    id: string
    referenceNumber: string
    patientId: string
    imageUrl: string
    result: string | null
    status: string
    createdAt: Date
    updatedAt: Date
    _count: XrayScanCountAggregateOutputType | null
    _min: XrayScanMinAggregateOutputType | null
    _max: XrayScanMaxAggregateOutputType | null
  }

  type GetXrayScanGroupByPayload<T extends XrayScanGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<XrayScanGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof XrayScanGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], XrayScanGroupByOutputType[P]>
            : GetScalarType<T[P], XrayScanGroupByOutputType[P]>
        }
      >
    >


  export type XrayScanSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    referenceNumber?: boolean
    patientId?: boolean
    imageUrl?: boolean
    result?: boolean
    status?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    patient?: boolean | PatientDefaultArgs<ExtArgs>
    metadata?: boolean | XrayScan$metadataArgs<ExtArgs>
  }, ExtArgs["result"]["xrayScan"]>

  export type XrayScanSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    referenceNumber?: boolean
    patientId?: boolean
    imageUrl?: boolean
    result?: boolean
    status?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    patient?: boolean | PatientDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["xrayScan"]>

  export type XrayScanSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    referenceNumber?: boolean
    patientId?: boolean
    imageUrl?: boolean
    result?: boolean
    status?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    patient?: boolean | PatientDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["xrayScan"]>

  export type XrayScanSelectScalar = {
    id?: boolean
    referenceNumber?: boolean
    patientId?: boolean
    imageUrl?: boolean
    result?: boolean
    status?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type XrayScanOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "referenceNumber" | "patientId" | "imageUrl" | "result" | "status" | "createdAt" | "updatedAt", ExtArgs["result"]["xrayScan"]>
  export type XrayScanInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    patient?: boolean | PatientDefaultArgs<ExtArgs>
    metadata?: boolean | XrayScan$metadataArgs<ExtArgs>
  }
  export type XrayScanIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    patient?: boolean | PatientDefaultArgs<ExtArgs>
  }
  export type XrayScanIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    patient?: boolean | PatientDefaultArgs<ExtArgs>
  }

  export type $XrayScanPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "XrayScan"
    objects: {
      patient: Prisma.$PatientPayload<ExtArgs>
      metadata: Prisma.$ScanMetadataPayload<ExtArgs> | null
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      referenceNumber: string
      patientId: string
      imageUrl: string
      result: string | null
      status: string
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["xrayScan"]>
    composites: {}
  }

  type XrayScanGetPayload<S extends boolean | null | undefined | XrayScanDefaultArgs> = $Result.GetResult<Prisma.$XrayScanPayload, S>

  type XrayScanCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<XrayScanFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: XrayScanCountAggregateInputType | true
    }

  export interface XrayScanDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['XrayScan'], meta: { name: 'XrayScan' } }
    /**
     * Find zero or one XrayScan that matches the filter.
     * @param {XrayScanFindUniqueArgs} args - Arguments to find a XrayScan
     * @example
     * // Get one XrayScan
     * const xrayScan = await prisma.xrayScan.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends XrayScanFindUniqueArgs>(args: SelectSubset<T, XrayScanFindUniqueArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one XrayScan that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {XrayScanFindUniqueOrThrowArgs} args - Arguments to find a XrayScan
     * @example
     * // Get one XrayScan
     * const xrayScan = await prisma.xrayScan.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends XrayScanFindUniqueOrThrowArgs>(args: SelectSubset<T, XrayScanFindUniqueOrThrowArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first XrayScan that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanFindFirstArgs} args - Arguments to find a XrayScan
     * @example
     * // Get one XrayScan
     * const xrayScan = await prisma.xrayScan.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends XrayScanFindFirstArgs>(args?: SelectSubset<T, XrayScanFindFirstArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first XrayScan that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanFindFirstOrThrowArgs} args - Arguments to find a XrayScan
     * @example
     * // Get one XrayScan
     * const xrayScan = await prisma.xrayScan.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends XrayScanFindFirstOrThrowArgs>(args?: SelectSubset<T, XrayScanFindFirstOrThrowArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more XrayScans that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all XrayScans
     * const xrayScans = await prisma.xrayScan.findMany()
     * 
     * // Get first 10 XrayScans
     * const xrayScans = await prisma.xrayScan.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const xrayScanWithIdOnly = await prisma.xrayScan.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends XrayScanFindManyArgs>(args?: SelectSubset<T, XrayScanFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a XrayScan.
     * @param {XrayScanCreateArgs} args - Arguments to create a XrayScan.
     * @example
     * // Create one XrayScan
     * const XrayScan = await prisma.xrayScan.create({
     *   data: {
     *     // ... data to create a XrayScan
     *   }
     * })
     * 
     */
    create<T extends XrayScanCreateArgs>(args: SelectSubset<T, XrayScanCreateArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many XrayScans.
     * @param {XrayScanCreateManyArgs} args - Arguments to create many XrayScans.
     * @example
     * // Create many XrayScans
     * const xrayScan = await prisma.xrayScan.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends XrayScanCreateManyArgs>(args?: SelectSubset<T, XrayScanCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many XrayScans and returns the data saved in the database.
     * @param {XrayScanCreateManyAndReturnArgs} args - Arguments to create many XrayScans.
     * @example
     * // Create many XrayScans
     * const xrayScan = await prisma.xrayScan.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many XrayScans and only return the `id`
     * const xrayScanWithIdOnly = await prisma.xrayScan.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends XrayScanCreateManyAndReturnArgs>(args?: SelectSubset<T, XrayScanCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a XrayScan.
     * @param {XrayScanDeleteArgs} args - Arguments to delete one XrayScan.
     * @example
     * // Delete one XrayScan
     * const XrayScan = await prisma.xrayScan.delete({
     *   where: {
     *     // ... filter to delete one XrayScan
     *   }
     * })
     * 
     */
    delete<T extends XrayScanDeleteArgs>(args: SelectSubset<T, XrayScanDeleteArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one XrayScan.
     * @param {XrayScanUpdateArgs} args - Arguments to update one XrayScan.
     * @example
     * // Update one XrayScan
     * const xrayScan = await prisma.xrayScan.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends XrayScanUpdateArgs>(args: SelectSubset<T, XrayScanUpdateArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more XrayScans.
     * @param {XrayScanDeleteManyArgs} args - Arguments to filter XrayScans to delete.
     * @example
     * // Delete a few XrayScans
     * const { count } = await prisma.xrayScan.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends XrayScanDeleteManyArgs>(args?: SelectSubset<T, XrayScanDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more XrayScans.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many XrayScans
     * const xrayScan = await prisma.xrayScan.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends XrayScanUpdateManyArgs>(args: SelectSubset<T, XrayScanUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more XrayScans and returns the data updated in the database.
     * @param {XrayScanUpdateManyAndReturnArgs} args - Arguments to update many XrayScans.
     * @example
     * // Update many XrayScans
     * const xrayScan = await prisma.xrayScan.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more XrayScans and only return the `id`
     * const xrayScanWithIdOnly = await prisma.xrayScan.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends XrayScanUpdateManyAndReturnArgs>(args: SelectSubset<T, XrayScanUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one XrayScan.
     * @param {XrayScanUpsertArgs} args - Arguments to update or create a XrayScan.
     * @example
     * // Update or create a XrayScan
     * const xrayScan = await prisma.xrayScan.upsert({
     *   create: {
     *     // ... data to create a XrayScan
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the XrayScan we want to update
     *   }
     * })
     */
    upsert<T extends XrayScanUpsertArgs>(args: SelectSubset<T, XrayScanUpsertArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of XrayScans.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanCountArgs} args - Arguments to filter XrayScans to count.
     * @example
     * // Count the number of XrayScans
     * const count = await prisma.xrayScan.count({
     *   where: {
     *     // ... the filter for the XrayScans we want to count
     *   }
     * })
    **/
    count<T extends XrayScanCountArgs>(
      args?: Subset<T, XrayScanCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], XrayScanCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a XrayScan.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends XrayScanAggregateArgs>(args: Subset<T, XrayScanAggregateArgs>): Prisma.PrismaPromise<GetXrayScanAggregateType<T>>

    /**
     * Group by XrayScan.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {XrayScanGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends XrayScanGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: XrayScanGroupByArgs['orderBy'] }
        : { orderBy?: XrayScanGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, XrayScanGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetXrayScanGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the XrayScan model
   */
  readonly fields: XrayScanFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for XrayScan.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__XrayScanClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    patient<T extends PatientDefaultArgs<ExtArgs> = {}>(args?: Subset<T, PatientDefaultArgs<ExtArgs>>): Prisma__PatientClient<$Result.GetResult<Prisma.$PatientPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    metadata<T extends XrayScan$metadataArgs<ExtArgs> = {}>(args?: Subset<T, XrayScan$metadataArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the XrayScan model
   */ 
  interface XrayScanFieldRefs {
    readonly id: FieldRef<"XrayScan", 'String'>
    readonly referenceNumber: FieldRef<"XrayScan", 'String'>
    readonly patientId: FieldRef<"XrayScan", 'String'>
    readonly imageUrl: FieldRef<"XrayScan", 'String'>
    readonly result: FieldRef<"XrayScan", 'String'>
    readonly status: FieldRef<"XrayScan", 'String'>
    readonly createdAt: FieldRef<"XrayScan", 'DateTime'>
    readonly updatedAt: FieldRef<"XrayScan", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * XrayScan findUnique
   */
  export type XrayScanFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * Filter, which XrayScan to fetch.
     */
    where: XrayScanWhereUniqueInput
  }

  /**
   * XrayScan findUniqueOrThrow
   */
  export type XrayScanFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * Filter, which XrayScan to fetch.
     */
    where: XrayScanWhereUniqueInput
  }

  /**
   * XrayScan findFirst
   */
  export type XrayScanFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * Filter, which XrayScan to fetch.
     */
    where?: XrayScanWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of XrayScans to fetch.
     */
    orderBy?: XrayScanOrderByWithRelationInput | XrayScanOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for XrayScans.
     */
    cursor?: XrayScanWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` XrayScans from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` XrayScans.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of XrayScans.
     */
    distinct?: XrayScanScalarFieldEnum | XrayScanScalarFieldEnum[]
  }

  /**
   * XrayScan findFirstOrThrow
   */
  export type XrayScanFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * Filter, which XrayScan to fetch.
     */
    where?: XrayScanWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of XrayScans to fetch.
     */
    orderBy?: XrayScanOrderByWithRelationInput | XrayScanOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for XrayScans.
     */
    cursor?: XrayScanWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` XrayScans from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` XrayScans.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of XrayScans.
     */
    distinct?: XrayScanScalarFieldEnum | XrayScanScalarFieldEnum[]
  }

  /**
   * XrayScan findMany
   */
  export type XrayScanFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * Filter, which XrayScans to fetch.
     */
    where?: XrayScanWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of XrayScans to fetch.
     */
    orderBy?: XrayScanOrderByWithRelationInput | XrayScanOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing XrayScans.
     */
    cursor?: XrayScanWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` XrayScans from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` XrayScans.
     */
    skip?: number
    distinct?: XrayScanScalarFieldEnum | XrayScanScalarFieldEnum[]
  }

  /**
   * XrayScan create
   */
  export type XrayScanCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * The data needed to create a XrayScan.
     */
    data: XOR<XrayScanCreateInput, XrayScanUncheckedCreateInput>
  }

  /**
   * XrayScan createMany
   */
  export type XrayScanCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many XrayScans.
     */
    data: XrayScanCreateManyInput | XrayScanCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * XrayScan createManyAndReturn
   */
  export type XrayScanCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * The data used to create many XrayScans.
     */
    data: XrayScanCreateManyInput | XrayScanCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * XrayScan update
   */
  export type XrayScanUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * The data needed to update a XrayScan.
     */
    data: XOR<XrayScanUpdateInput, XrayScanUncheckedUpdateInput>
    /**
     * Choose, which XrayScan to update.
     */
    where: XrayScanWhereUniqueInput
  }

  /**
   * XrayScan updateMany
   */
  export type XrayScanUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update XrayScans.
     */
    data: XOR<XrayScanUpdateManyMutationInput, XrayScanUncheckedUpdateManyInput>
    /**
     * Filter which XrayScans to update
     */
    where?: XrayScanWhereInput
    /**
     * Limit how many XrayScans to update.
     */
    limit?: number
  }

  /**
   * XrayScan updateManyAndReturn
   */
  export type XrayScanUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * The data used to update XrayScans.
     */
    data: XOR<XrayScanUpdateManyMutationInput, XrayScanUncheckedUpdateManyInput>
    /**
     * Filter which XrayScans to update
     */
    where?: XrayScanWhereInput
    /**
     * Limit how many XrayScans to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * XrayScan upsert
   */
  export type XrayScanUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * The filter to search for the XrayScan to update in case it exists.
     */
    where: XrayScanWhereUniqueInput
    /**
     * In case the XrayScan found by the `where` argument doesn't exist, create a new XrayScan with this data.
     */
    create: XOR<XrayScanCreateInput, XrayScanUncheckedCreateInput>
    /**
     * In case the XrayScan was found with the provided `where` argument, update it with this data.
     */
    update: XOR<XrayScanUpdateInput, XrayScanUncheckedUpdateInput>
  }

  /**
   * XrayScan delete
   */
  export type XrayScanDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
    /**
     * Filter which XrayScan to delete.
     */
    where: XrayScanWhereUniqueInput
  }

  /**
   * XrayScan deleteMany
   */
  export type XrayScanDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which XrayScans to delete
     */
    where?: XrayScanWhereInput
    /**
     * Limit how many XrayScans to delete.
     */
    limit?: number
  }

  /**
   * XrayScan.metadata
   */
  export type XrayScan$metadataArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    where?: ScanMetadataWhereInput
  }

  /**
   * XrayScan without action
   */
  export type XrayScanDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the XrayScan
     */
    select?: XrayScanSelect<ExtArgs> | null
    /**
     * Omit specific fields from the XrayScan
     */
    omit?: XrayScanOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: XrayScanInclude<ExtArgs> | null
  }


  /**
   * Model ScanMetadata
   */

  export type AggregateScanMetadata = {
    _count: ScanMetadataCountAggregateOutputType | null
    _avg: ScanMetadataAvgAggregateOutputType | null
    _sum: ScanMetadataSumAggregateOutputType | null
    _min: ScanMetadataMinAggregateOutputType | null
    _max: ScanMetadataMaxAggregateOutputType | null
  }

  export type ScanMetadataAvgAggregateOutputType = {
    confidence: number | null
  }

  export type ScanMetadataSumAggregateOutputType = {
    confidence: number | null
  }

  export type ScanMetadataMinAggregateOutputType = {
    id: string | null
    scanId: string | null
    confidence: number | null
    pneumoniaType: string | null
    severity: string | null
    recommendedAction: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type ScanMetadataMaxAggregateOutputType = {
    id: string | null
    scanId: string | null
    confidence: number | null
    pneumoniaType: string | null
    severity: string | null
    recommendedAction: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type ScanMetadataCountAggregateOutputType = {
    id: number
    scanId: number
    confidence: number
    pneumoniaType: number
    severity: number
    recommendedAction: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type ScanMetadataAvgAggregateInputType = {
    confidence?: true
  }

  export type ScanMetadataSumAggregateInputType = {
    confidence?: true
  }

  export type ScanMetadataMinAggregateInputType = {
    id?: true
    scanId?: true
    confidence?: true
    pneumoniaType?: true
    severity?: true
    recommendedAction?: true
    createdAt?: true
    updatedAt?: true
  }

  export type ScanMetadataMaxAggregateInputType = {
    id?: true
    scanId?: true
    confidence?: true
    pneumoniaType?: true
    severity?: true
    recommendedAction?: true
    createdAt?: true
    updatedAt?: true
  }

  export type ScanMetadataCountAggregateInputType = {
    id?: true
    scanId?: true
    confidence?: true
    pneumoniaType?: true
    severity?: true
    recommendedAction?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type ScanMetadataAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which ScanMetadata to aggregate.
     */
    where?: ScanMetadataWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ScanMetadata to fetch.
     */
    orderBy?: ScanMetadataOrderByWithRelationInput | ScanMetadataOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: ScanMetadataWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ScanMetadata from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ScanMetadata.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned ScanMetadata
    **/
    _count?: true | ScanMetadataCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: ScanMetadataAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: ScanMetadataSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: ScanMetadataMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: ScanMetadataMaxAggregateInputType
  }

  export type GetScanMetadataAggregateType<T extends ScanMetadataAggregateArgs> = {
        [P in keyof T & keyof AggregateScanMetadata]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateScanMetadata[P]>
      : GetScalarType<T[P], AggregateScanMetadata[P]>
  }




  export type ScanMetadataGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: ScanMetadataWhereInput
    orderBy?: ScanMetadataOrderByWithAggregationInput | ScanMetadataOrderByWithAggregationInput[]
    by: ScanMetadataScalarFieldEnum[] | ScanMetadataScalarFieldEnum
    having?: ScanMetadataScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: ScanMetadataCountAggregateInputType | true
    _avg?: ScanMetadataAvgAggregateInputType
    _sum?: ScanMetadataSumAggregateInputType
    _min?: ScanMetadataMinAggregateInputType
    _max?: ScanMetadataMaxAggregateInputType
  }

  export type ScanMetadataGroupByOutputType = {
    id: string
    scanId: string
    confidence: number
    pneumoniaType: string | null
    severity: string | null
    recommendedAction: string | null
    createdAt: Date
    updatedAt: Date
    _count: ScanMetadataCountAggregateOutputType | null
    _avg: ScanMetadataAvgAggregateOutputType | null
    _sum: ScanMetadataSumAggregateOutputType | null
    _min: ScanMetadataMinAggregateOutputType | null
    _max: ScanMetadataMaxAggregateOutputType | null
  }

  type GetScanMetadataGroupByPayload<T extends ScanMetadataGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<ScanMetadataGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof ScanMetadataGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], ScanMetadataGroupByOutputType[P]>
            : GetScalarType<T[P], ScanMetadataGroupByOutputType[P]>
        }
      >
    >


  export type ScanMetadataSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    scanId?: boolean
    confidence?: boolean
    pneumoniaType?: boolean
    severity?: boolean
    recommendedAction?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    scan?: boolean | XrayScanDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["scanMetadata"]>

  export type ScanMetadataSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    scanId?: boolean
    confidence?: boolean
    pneumoniaType?: boolean
    severity?: boolean
    recommendedAction?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    scan?: boolean | XrayScanDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["scanMetadata"]>

  export type ScanMetadataSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    scanId?: boolean
    confidence?: boolean
    pneumoniaType?: boolean
    severity?: boolean
    recommendedAction?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    scan?: boolean | XrayScanDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["scanMetadata"]>

  export type ScanMetadataSelectScalar = {
    id?: boolean
    scanId?: boolean
    confidence?: boolean
    pneumoniaType?: boolean
    severity?: boolean
    recommendedAction?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type ScanMetadataOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "scanId" | "confidence" | "pneumoniaType" | "severity" | "recommendedAction" | "createdAt" | "updatedAt", ExtArgs["result"]["scanMetadata"]>
  export type ScanMetadataInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    scan?: boolean | XrayScanDefaultArgs<ExtArgs>
  }
  export type ScanMetadataIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    scan?: boolean | XrayScanDefaultArgs<ExtArgs>
  }
  export type ScanMetadataIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    scan?: boolean | XrayScanDefaultArgs<ExtArgs>
  }

  export type $ScanMetadataPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "ScanMetadata"
    objects: {
      scan: Prisma.$XrayScanPayload<ExtArgs>
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      scanId: string
      confidence: number
      pneumoniaType: string | null
      severity: string | null
      recommendedAction: string | null
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["scanMetadata"]>
    composites: {}
  }

  type ScanMetadataGetPayload<S extends boolean | null | undefined | ScanMetadataDefaultArgs> = $Result.GetResult<Prisma.$ScanMetadataPayload, S>

  type ScanMetadataCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<ScanMetadataFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: ScanMetadataCountAggregateInputType | true
    }

  export interface ScanMetadataDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['ScanMetadata'], meta: { name: 'ScanMetadata' } }
    /**
     * Find zero or one ScanMetadata that matches the filter.
     * @param {ScanMetadataFindUniqueArgs} args - Arguments to find a ScanMetadata
     * @example
     * // Get one ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends ScanMetadataFindUniqueArgs>(args: SelectSubset<T, ScanMetadataFindUniqueArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one ScanMetadata that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {ScanMetadataFindUniqueOrThrowArgs} args - Arguments to find a ScanMetadata
     * @example
     * // Get one ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends ScanMetadataFindUniqueOrThrowArgs>(args: SelectSubset<T, ScanMetadataFindUniqueOrThrowArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first ScanMetadata that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataFindFirstArgs} args - Arguments to find a ScanMetadata
     * @example
     * // Get one ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends ScanMetadataFindFirstArgs>(args?: SelectSubset<T, ScanMetadataFindFirstArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first ScanMetadata that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataFindFirstOrThrowArgs} args - Arguments to find a ScanMetadata
     * @example
     * // Get one ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends ScanMetadataFindFirstOrThrowArgs>(args?: SelectSubset<T, ScanMetadataFindFirstOrThrowArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more ScanMetadata that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.findMany()
     * 
     * // Get first 10 ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const scanMetadataWithIdOnly = await prisma.scanMetadata.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends ScanMetadataFindManyArgs>(args?: SelectSubset<T, ScanMetadataFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a ScanMetadata.
     * @param {ScanMetadataCreateArgs} args - Arguments to create a ScanMetadata.
     * @example
     * // Create one ScanMetadata
     * const ScanMetadata = await prisma.scanMetadata.create({
     *   data: {
     *     // ... data to create a ScanMetadata
     *   }
     * })
     * 
     */
    create<T extends ScanMetadataCreateArgs>(args: SelectSubset<T, ScanMetadataCreateArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many ScanMetadata.
     * @param {ScanMetadataCreateManyArgs} args - Arguments to create many ScanMetadata.
     * @example
     * // Create many ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends ScanMetadataCreateManyArgs>(args?: SelectSubset<T, ScanMetadataCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many ScanMetadata and returns the data saved in the database.
     * @param {ScanMetadataCreateManyAndReturnArgs} args - Arguments to create many ScanMetadata.
     * @example
     * // Create many ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many ScanMetadata and only return the `id`
     * const scanMetadataWithIdOnly = await prisma.scanMetadata.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends ScanMetadataCreateManyAndReturnArgs>(args?: SelectSubset<T, ScanMetadataCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a ScanMetadata.
     * @param {ScanMetadataDeleteArgs} args - Arguments to delete one ScanMetadata.
     * @example
     * // Delete one ScanMetadata
     * const ScanMetadata = await prisma.scanMetadata.delete({
     *   where: {
     *     // ... filter to delete one ScanMetadata
     *   }
     * })
     * 
     */
    delete<T extends ScanMetadataDeleteArgs>(args: SelectSubset<T, ScanMetadataDeleteArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one ScanMetadata.
     * @param {ScanMetadataUpdateArgs} args - Arguments to update one ScanMetadata.
     * @example
     * // Update one ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends ScanMetadataUpdateArgs>(args: SelectSubset<T, ScanMetadataUpdateArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more ScanMetadata.
     * @param {ScanMetadataDeleteManyArgs} args - Arguments to filter ScanMetadata to delete.
     * @example
     * // Delete a few ScanMetadata
     * const { count } = await prisma.scanMetadata.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends ScanMetadataDeleteManyArgs>(args?: SelectSubset<T, ScanMetadataDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more ScanMetadata.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends ScanMetadataUpdateManyArgs>(args: SelectSubset<T, ScanMetadataUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more ScanMetadata and returns the data updated in the database.
     * @param {ScanMetadataUpdateManyAndReturnArgs} args - Arguments to update many ScanMetadata.
     * @example
     * // Update many ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more ScanMetadata and only return the `id`
     * const scanMetadataWithIdOnly = await prisma.scanMetadata.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends ScanMetadataUpdateManyAndReturnArgs>(args: SelectSubset<T, ScanMetadataUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one ScanMetadata.
     * @param {ScanMetadataUpsertArgs} args - Arguments to update or create a ScanMetadata.
     * @example
     * // Update or create a ScanMetadata
     * const scanMetadata = await prisma.scanMetadata.upsert({
     *   create: {
     *     // ... data to create a ScanMetadata
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the ScanMetadata we want to update
     *   }
     * })
     */
    upsert<T extends ScanMetadataUpsertArgs>(args: SelectSubset<T, ScanMetadataUpsertArgs<ExtArgs>>): Prisma__ScanMetadataClient<$Result.GetResult<Prisma.$ScanMetadataPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of ScanMetadata.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataCountArgs} args - Arguments to filter ScanMetadata to count.
     * @example
     * // Count the number of ScanMetadata
     * const count = await prisma.scanMetadata.count({
     *   where: {
     *     // ... the filter for the ScanMetadata we want to count
     *   }
     * })
    **/
    count<T extends ScanMetadataCountArgs>(
      args?: Subset<T, ScanMetadataCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], ScanMetadataCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a ScanMetadata.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends ScanMetadataAggregateArgs>(args: Subset<T, ScanMetadataAggregateArgs>): Prisma.PrismaPromise<GetScanMetadataAggregateType<T>>

    /**
     * Group by ScanMetadata.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ScanMetadataGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends ScanMetadataGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: ScanMetadataGroupByArgs['orderBy'] }
        : { orderBy?: ScanMetadataGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, ScanMetadataGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetScanMetadataGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the ScanMetadata model
   */
  readonly fields: ScanMetadataFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for ScanMetadata.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__ScanMetadataClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    scan<T extends XrayScanDefaultArgs<ExtArgs> = {}>(args?: Subset<T, XrayScanDefaultArgs<ExtArgs>>): Prisma__XrayScanClient<$Result.GetResult<Prisma.$XrayScanPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the ScanMetadata model
   */ 
  interface ScanMetadataFieldRefs {
    readonly id: FieldRef<"ScanMetadata", 'String'>
    readonly scanId: FieldRef<"ScanMetadata", 'String'>
    readonly confidence: FieldRef<"ScanMetadata", 'Float'>
    readonly pneumoniaType: FieldRef<"ScanMetadata", 'String'>
    readonly severity: FieldRef<"ScanMetadata", 'String'>
    readonly recommendedAction: FieldRef<"ScanMetadata", 'String'>
    readonly createdAt: FieldRef<"ScanMetadata", 'DateTime'>
    readonly updatedAt: FieldRef<"ScanMetadata", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * ScanMetadata findUnique
   */
  export type ScanMetadataFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * Filter, which ScanMetadata to fetch.
     */
    where: ScanMetadataWhereUniqueInput
  }

  /**
   * ScanMetadata findUniqueOrThrow
   */
  export type ScanMetadataFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * Filter, which ScanMetadata to fetch.
     */
    where: ScanMetadataWhereUniqueInput
  }

  /**
   * ScanMetadata findFirst
   */
  export type ScanMetadataFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * Filter, which ScanMetadata to fetch.
     */
    where?: ScanMetadataWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ScanMetadata to fetch.
     */
    orderBy?: ScanMetadataOrderByWithRelationInput | ScanMetadataOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for ScanMetadata.
     */
    cursor?: ScanMetadataWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ScanMetadata from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ScanMetadata.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of ScanMetadata.
     */
    distinct?: ScanMetadataScalarFieldEnum | ScanMetadataScalarFieldEnum[]
  }

  /**
   * ScanMetadata findFirstOrThrow
   */
  export type ScanMetadataFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * Filter, which ScanMetadata to fetch.
     */
    where?: ScanMetadataWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ScanMetadata to fetch.
     */
    orderBy?: ScanMetadataOrderByWithRelationInput | ScanMetadataOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for ScanMetadata.
     */
    cursor?: ScanMetadataWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ScanMetadata from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ScanMetadata.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of ScanMetadata.
     */
    distinct?: ScanMetadataScalarFieldEnum | ScanMetadataScalarFieldEnum[]
  }

  /**
   * ScanMetadata findMany
   */
  export type ScanMetadataFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * Filter, which ScanMetadata to fetch.
     */
    where?: ScanMetadataWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ScanMetadata to fetch.
     */
    orderBy?: ScanMetadataOrderByWithRelationInput | ScanMetadataOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing ScanMetadata.
     */
    cursor?: ScanMetadataWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ScanMetadata from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ScanMetadata.
     */
    skip?: number
    distinct?: ScanMetadataScalarFieldEnum | ScanMetadataScalarFieldEnum[]
  }

  /**
   * ScanMetadata create
   */
  export type ScanMetadataCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * The data needed to create a ScanMetadata.
     */
    data: XOR<ScanMetadataCreateInput, ScanMetadataUncheckedCreateInput>
  }

  /**
   * ScanMetadata createMany
   */
  export type ScanMetadataCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many ScanMetadata.
     */
    data: ScanMetadataCreateManyInput | ScanMetadataCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * ScanMetadata createManyAndReturn
   */
  export type ScanMetadataCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * The data used to create many ScanMetadata.
     */
    data: ScanMetadataCreateManyInput | ScanMetadataCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * ScanMetadata update
   */
  export type ScanMetadataUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * The data needed to update a ScanMetadata.
     */
    data: XOR<ScanMetadataUpdateInput, ScanMetadataUncheckedUpdateInput>
    /**
     * Choose, which ScanMetadata to update.
     */
    where: ScanMetadataWhereUniqueInput
  }

  /**
   * ScanMetadata updateMany
   */
  export type ScanMetadataUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update ScanMetadata.
     */
    data: XOR<ScanMetadataUpdateManyMutationInput, ScanMetadataUncheckedUpdateManyInput>
    /**
     * Filter which ScanMetadata to update
     */
    where?: ScanMetadataWhereInput
    /**
     * Limit how many ScanMetadata to update.
     */
    limit?: number
  }

  /**
   * ScanMetadata updateManyAndReturn
   */
  export type ScanMetadataUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * The data used to update ScanMetadata.
     */
    data: XOR<ScanMetadataUpdateManyMutationInput, ScanMetadataUncheckedUpdateManyInput>
    /**
     * Filter which ScanMetadata to update
     */
    where?: ScanMetadataWhereInput
    /**
     * Limit how many ScanMetadata to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * ScanMetadata upsert
   */
  export type ScanMetadataUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * The filter to search for the ScanMetadata to update in case it exists.
     */
    where: ScanMetadataWhereUniqueInput
    /**
     * In case the ScanMetadata found by the `where` argument doesn't exist, create a new ScanMetadata with this data.
     */
    create: XOR<ScanMetadataCreateInput, ScanMetadataUncheckedCreateInput>
    /**
     * In case the ScanMetadata was found with the provided `where` argument, update it with this data.
     */
    update: XOR<ScanMetadataUpdateInput, ScanMetadataUncheckedUpdateInput>
  }

  /**
   * ScanMetadata delete
   */
  export type ScanMetadataDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
    /**
     * Filter which ScanMetadata to delete.
     */
    where: ScanMetadataWhereUniqueInput
  }

  /**
   * ScanMetadata deleteMany
   */
  export type ScanMetadataDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which ScanMetadata to delete
     */
    where?: ScanMetadataWhereInput
    /**
     * Limit how many ScanMetadata to delete.
     */
    limit?: number
  }

  /**
   * ScanMetadata without action
   */
  export type ScanMetadataDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ScanMetadata
     */
    select?: ScanMetadataSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ScanMetadata
     */
    omit?: ScanMetadataOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ScanMetadataInclude<ExtArgs> | null
  }


  /**
   * Enums
   */

  export const TransactionIsolationLevel: {
    ReadUncommitted: 'ReadUncommitted',
    ReadCommitted: 'ReadCommitted',
    RepeatableRead: 'RepeatableRead',
    Serializable: 'Serializable'
  };

  export type TransactionIsolationLevel = (typeof TransactionIsolationLevel)[keyof typeof TransactionIsolationLevel]


  export const UserScalarFieldEnum: {
    id: 'id',
    email: 'email',
    doctorId: 'doctorId',
    password: 'password',
    role: 'role',
    name: 'name',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type UserScalarFieldEnum = (typeof UserScalarFieldEnum)[keyof typeof UserScalarFieldEnum]


  export const PatientScalarFieldEnum: {
    id: 'id',
    name: 'name',
    referenceNumber: 'referenceNumber',
    dateOfBirth: 'dateOfBirth',
    doctorId: 'doctorId',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type PatientScalarFieldEnum = (typeof PatientScalarFieldEnum)[keyof typeof PatientScalarFieldEnum]


  export const XrayScanScalarFieldEnum: {
    id: 'id',
    referenceNumber: 'referenceNumber',
    patientId: 'patientId',
    imageUrl: 'imageUrl',
    result: 'result',
    status: 'status',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type XrayScanScalarFieldEnum = (typeof XrayScanScalarFieldEnum)[keyof typeof XrayScanScalarFieldEnum]


  export const ScanMetadataScalarFieldEnum: {
    id: 'id',
    scanId: 'scanId',
    confidence: 'confidence',
    pneumoniaType: 'pneumoniaType',
    severity: 'severity',
    recommendedAction: 'recommendedAction',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type ScanMetadataScalarFieldEnum = (typeof ScanMetadataScalarFieldEnum)[keyof typeof ScanMetadataScalarFieldEnum]


  export const SortOrder: {
    asc: 'asc',
    desc: 'desc'
  };

  export type SortOrder = (typeof SortOrder)[keyof typeof SortOrder]


  export const QueryMode: {
    default: 'default',
    insensitive: 'insensitive'
  };

  export type QueryMode = (typeof QueryMode)[keyof typeof QueryMode]


  export const NullsOrder: {
    first: 'first',
    last: 'last'
  };

  export type NullsOrder = (typeof NullsOrder)[keyof typeof NullsOrder]


  /**
   * Field references 
   */


  /**
   * Reference to a field of type 'String'
   */
  export type StringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String'>
    


  /**
   * Reference to a field of type 'String[]'
   */
  export type ListStringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String[]'>
    


  /**
   * Reference to a field of type 'UserRole'
   */
  export type EnumUserRoleFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'UserRole'>
    


  /**
   * Reference to a field of type 'UserRole[]'
   */
  export type ListEnumUserRoleFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'UserRole[]'>
    


  /**
   * Reference to a field of type 'DateTime'
   */
  export type DateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime'>
    


  /**
   * Reference to a field of type 'DateTime[]'
   */
  export type ListDateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime[]'>
    


  /**
   * Reference to a field of type 'Float'
   */
  export type FloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float'>
    


  /**
   * Reference to a field of type 'Float[]'
   */
  export type ListFloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float[]'>
    


  /**
   * Reference to a field of type 'Int'
   */
  export type IntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int'>
    


  /**
   * Reference to a field of type 'Int[]'
   */
  export type ListIntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int[]'>
    
  /**
   * Deep Input Types
   */


  export type UserWhereInput = {
    AND?: UserWhereInput | UserWhereInput[]
    OR?: UserWhereInput[]
    NOT?: UserWhereInput | UserWhereInput[]
    id?: StringFilter<"User"> | string
    email?: StringFilter<"User"> | string
    doctorId?: StringNullableFilter<"User"> | string | null
    password?: StringFilter<"User"> | string
    role?: EnumUserRoleFilter<"User"> | $Enums.UserRole
    name?: StringFilter<"User"> | string
    createdAt?: DateTimeFilter<"User"> | Date | string
    updatedAt?: DateTimeFilter<"User"> | Date | string
    patients?: PatientListRelationFilter
  }

  export type UserOrderByWithRelationInput = {
    id?: SortOrder
    email?: SortOrder
    doctorId?: SortOrderInput | SortOrder
    password?: SortOrder
    role?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    patients?: PatientOrderByRelationAggregateInput
  }

  export type UserWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    email?: string
    doctorId?: string
    AND?: UserWhereInput | UserWhereInput[]
    OR?: UserWhereInput[]
    NOT?: UserWhereInput | UserWhereInput[]
    password?: StringFilter<"User"> | string
    role?: EnumUserRoleFilter<"User"> | $Enums.UserRole
    name?: StringFilter<"User"> | string
    createdAt?: DateTimeFilter<"User"> | Date | string
    updatedAt?: DateTimeFilter<"User"> | Date | string
    patients?: PatientListRelationFilter
  }, "id" | "email" | "doctorId">

  export type UserOrderByWithAggregationInput = {
    id?: SortOrder
    email?: SortOrder
    doctorId?: SortOrderInput | SortOrder
    password?: SortOrder
    role?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: UserCountOrderByAggregateInput
    _max?: UserMaxOrderByAggregateInput
    _min?: UserMinOrderByAggregateInput
  }

  export type UserScalarWhereWithAggregatesInput = {
    AND?: UserScalarWhereWithAggregatesInput | UserScalarWhereWithAggregatesInput[]
    OR?: UserScalarWhereWithAggregatesInput[]
    NOT?: UserScalarWhereWithAggregatesInput | UserScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"User"> | string
    email?: StringWithAggregatesFilter<"User"> | string
    doctorId?: StringNullableWithAggregatesFilter<"User"> | string | null
    password?: StringWithAggregatesFilter<"User"> | string
    role?: EnumUserRoleWithAggregatesFilter<"User"> | $Enums.UserRole
    name?: StringWithAggregatesFilter<"User"> | string
    createdAt?: DateTimeWithAggregatesFilter<"User"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"User"> | Date | string
  }

  export type PatientWhereInput = {
    AND?: PatientWhereInput | PatientWhereInput[]
    OR?: PatientWhereInput[]
    NOT?: PatientWhereInput | PatientWhereInput[]
    id?: StringFilter<"Patient"> | string
    name?: StringFilter<"Patient"> | string
    referenceNumber?: StringFilter<"Patient"> | string
    dateOfBirth?: DateTimeFilter<"Patient"> | Date | string
    doctorId?: StringFilter<"Patient"> | string
    createdAt?: DateTimeFilter<"Patient"> | Date | string
    updatedAt?: DateTimeFilter<"Patient"> | Date | string
    doctor?: XOR<UserScalarRelationFilter, UserWhereInput>
    xrayScans?: XrayScanListRelationFilter
  }

  export type PatientOrderByWithRelationInput = {
    id?: SortOrder
    name?: SortOrder
    referenceNumber?: SortOrder
    dateOfBirth?: SortOrder
    doctorId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    doctor?: UserOrderByWithRelationInput
    xrayScans?: XrayScanOrderByRelationAggregateInput
  }

  export type PatientWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    referenceNumber?: string
    AND?: PatientWhereInput | PatientWhereInput[]
    OR?: PatientWhereInput[]
    NOT?: PatientWhereInput | PatientWhereInput[]
    name?: StringFilter<"Patient"> | string
    dateOfBirth?: DateTimeFilter<"Patient"> | Date | string
    doctorId?: StringFilter<"Patient"> | string
    createdAt?: DateTimeFilter<"Patient"> | Date | string
    updatedAt?: DateTimeFilter<"Patient"> | Date | string
    doctor?: XOR<UserScalarRelationFilter, UserWhereInput>
    xrayScans?: XrayScanListRelationFilter
  }, "id" | "referenceNumber">

  export type PatientOrderByWithAggregationInput = {
    id?: SortOrder
    name?: SortOrder
    referenceNumber?: SortOrder
    dateOfBirth?: SortOrder
    doctorId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: PatientCountOrderByAggregateInput
    _max?: PatientMaxOrderByAggregateInput
    _min?: PatientMinOrderByAggregateInput
  }

  export type PatientScalarWhereWithAggregatesInput = {
    AND?: PatientScalarWhereWithAggregatesInput | PatientScalarWhereWithAggregatesInput[]
    OR?: PatientScalarWhereWithAggregatesInput[]
    NOT?: PatientScalarWhereWithAggregatesInput | PatientScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"Patient"> | string
    name?: StringWithAggregatesFilter<"Patient"> | string
    referenceNumber?: StringWithAggregatesFilter<"Patient"> | string
    dateOfBirth?: DateTimeWithAggregatesFilter<"Patient"> | Date | string
    doctorId?: StringWithAggregatesFilter<"Patient"> | string
    createdAt?: DateTimeWithAggregatesFilter<"Patient"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"Patient"> | Date | string
  }

  export type XrayScanWhereInput = {
    AND?: XrayScanWhereInput | XrayScanWhereInput[]
    OR?: XrayScanWhereInput[]
    NOT?: XrayScanWhereInput | XrayScanWhereInput[]
    id?: StringFilter<"XrayScan"> | string
    referenceNumber?: StringFilter<"XrayScan"> | string
    patientId?: StringFilter<"XrayScan"> | string
    imageUrl?: StringFilter<"XrayScan"> | string
    result?: StringNullableFilter<"XrayScan"> | string | null
    status?: StringFilter<"XrayScan"> | string
    createdAt?: DateTimeFilter<"XrayScan"> | Date | string
    updatedAt?: DateTimeFilter<"XrayScan"> | Date | string
    patient?: XOR<PatientScalarRelationFilter, PatientWhereInput>
    metadata?: XOR<ScanMetadataNullableScalarRelationFilter, ScanMetadataWhereInput> | null
  }

  export type XrayScanOrderByWithRelationInput = {
    id?: SortOrder
    referenceNumber?: SortOrder
    patientId?: SortOrder
    imageUrl?: SortOrder
    result?: SortOrderInput | SortOrder
    status?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    patient?: PatientOrderByWithRelationInput
    metadata?: ScanMetadataOrderByWithRelationInput
  }

  export type XrayScanWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    referenceNumber?: string
    AND?: XrayScanWhereInput | XrayScanWhereInput[]
    OR?: XrayScanWhereInput[]
    NOT?: XrayScanWhereInput | XrayScanWhereInput[]
    patientId?: StringFilter<"XrayScan"> | string
    imageUrl?: StringFilter<"XrayScan"> | string
    result?: StringNullableFilter<"XrayScan"> | string | null
    status?: StringFilter<"XrayScan"> | string
    createdAt?: DateTimeFilter<"XrayScan"> | Date | string
    updatedAt?: DateTimeFilter<"XrayScan"> | Date | string
    patient?: XOR<PatientScalarRelationFilter, PatientWhereInput>
    metadata?: XOR<ScanMetadataNullableScalarRelationFilter, ScanMetadataWhereInput> | null
  }, "id" | "referenceNumber">

  export type XrayScanOrderByWithAggregationInput = {
    id?: SortOrder
    referenceNumber?: SortOrder
    patientId?: SortOrder
    imageUrl?: SortOrder
    result?: SortOrderInput | SortOrder
    status?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: XrayScanCountOrderByAggregateInput
    _max?: XrayScanMaxOrderByAggregateInput
    _min?: XrayScanMinOrderByAggregateInput
  }

  export type XrayScanScalarWhereWithAggregatesInput = {
    AND?: XrayScanScalarWhereWithAggregatesInput | XrayScanScalarWhereWithAggregatesInput[]
    OR?: XrayScanScalarWhereWithAggregatesInput[]
    NOT?: XrayScanScalarWhereWithAggregatesInput | XrayScanScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"XrayScan"> | string
    referenceNumber?: StringWithAggregatesFilter<"XrayScan"> | string
    patientId?: StringWithAggregatesFilter<"XrayScan"> | string
    imageUrl?: StringWithAggregatesFilter<"XrayScan"> | string
    result?: StringNullableWithAggregatesFilter<"XrayScan"> | string | null
    status?: StringWithAggregatesFilter<"XrayScan"> | string
    createdAt?: DateTimeWithAggregatesFilter<"XrayScan"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"XrayScan"> | Date | string
  }

  export type ScanMetadataWhereInput = {
    AND?: ScanMetadataWhereInput | ScanMetadataWhereInput[]
    OR?: ScanMetadataWhereInput[]
    NOT?: ScanMetadataWhereInput | ScanMetadataWhereInput[]
    id?: StringFilter<"ScanMetadata"> | string
    scanId?: StringFilter<"ScanMetadata"> | string
    confidence?: FloatFilter<"ScanMetadata"> | number
    pneumoniaType?: StringNullableFilter<"ScanMetadata"> | string | null
    severity?: StringNullableFilter<"ScanMetadata"> | string | null
    recommendedAction?: StringNullableFilter<"ScanMetadata"> | string | null
    createdAt?: DateTimeFilter<"ScanMetadata"> | Date | string
    updatedAt?: DateTimeFilter<"ScanMetadata"> | Date | string
    scan?: XOR<XrayScanScalarRelationFilter, XrayScanWhereInput>
  }

  export type ScanMetadataOrderByWithRelationInput = {
    id?: SortOrder
    scanId?: SortOrder
    confidence?: SortOrder
    pneumoniaType?: SortOrderInput | SortOrder
    severity?: SortOrderInput | SortOrder
    recommendedAction?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    scan?: XrayScanOrderByWithRelationInput
  }

  export type ScanMetadataWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    scanId?: string
    AND?: ScanMetadataWhereInput | ScanMetadataWhereInput[]
    OR?: ScanMetadataWhereInput[]
    NOT?: ScanMetadataWhereInput | ScanMetadataWhereInput[]
    confidence?: FloatFilter<"ScanMetadata"> | number
    pneumoniaType?: StringNullableFilter<"ScanMetadata"> | string | null
    severity?: StringNullableFilter<"ScanMetadata"> | string | null
    recommendedAction?: StringNullableFilter<"ScanMetadata"> | string | null
    createdAt?: DateTimeFilter<"ScanMetadata"> | Date | string
    updatedAt?: DateTimeFilter<"ScanMetadata"> | Date | string
    scan?: XOR<XrayScanScalarRelationFilter, XrayScanWhereInput>
  }, "id" | "scanId">

  export type ScanMetadataOrderByWithAggregationInput = {
    id?: SortOrder
    scanId?: SortOrder
    confidence?: SortOrder
    pneumoniaType?: SortOrderInput | SortOrder
    severity?: SortOrderInput | SortOrder
    recommendedAction?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: ScanMetadataCountOrderByAggregateInput
    _avg?: ScanMetadataAvgOrderByAggregateInput
    _max?: ScanMetadataMaxOrderByAggregateInput
    _min?: ScanMetadataMinOrderByAggregateInput
    _sum?: ScanMetadataSumOrderByAggregateInput
  }

  export type ScanMetadataScalarWhereWithAggregatesInput = {
    AND?: ScanMetadataScalarWhereWithAggregatesInput | ScanMetadataScalarWhereWithAggregatesInput[]
    OR?: ScanMetadataScalarWhereWithAggregatesInput[]
    NOT?: ScanMetadataScalarWhereWithAggregatesInput | ScanMetadataScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"ScanMetadata"> | string
    scanId?: StringWithAggregatesFilter<"ScanMetadata"> | string
    confidence?: FloatWithAggregatesFilter<"ScanMetadata"> | number
    pneumoniaType?: StringNullableWithAggregatesFilter<"ScanMetadata"> | string | null
    severity?: StringNullableWithAggregatesFilter<"ScanMetadata"> | string | null
    recommendedAction?: StringNullableWithAggregatesFilter<"ScanMetadata"> | string | null
    createdAt?: DateTimeWithAggregatesFilter<"ScanMetadata"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"ScanMetadata"> | Date | string
  }

  export type UserCreateInput = {
    id?: string
    email: string
    doctorId?: string | null
    password: string
    role: $Enums.UserRole
    name: string
    createdAt?: Date | string
    updatedAt?: Date | string
    patients?: PatientCreateNestedManyWithoutDoctorInput
  }

  export type UserUncheckedCreateInput = {
    id?: string
    email: string
    doctorId?: string | null
    password: string
    role: $Enums.UserRole
    name: string
    createdAt?: Date | string
    updatedAt?: Date | string
    patients?: PatientUncheckedCreateNestedManyWithoutDoctorInput
  }

  export type UserUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    email?: StringFieldUpdateOperationsInput | string
    doctorId?: NullableStringFieldUpdateOperationsInput | string | null
    password?: StringFieldUpdateOperationsInput | string
    role?: EnumUserRoleFieldUpdateOperationsInput | $Enums.UserRole
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    patients?: PatientUpdateManyWithoutDoctorNestedInput
  }

  export type UserUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    email?: StringFieldUpdateOperationsInput | string
    doctorId?: NullableStringFieldUpdateOperationsInput | string | null
    password?: StringFieldUpdateOperationsInput | string
    role?: EnumUserRoleFieldUpdateOperationsInput | $Enums.UserRole
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    patients?: PatientUncheckedUpdateManyWithoutDoctorNestedInput
  }

  export type UserCreateManyInput = {
    id?: string
    email: string
    doctorId?: string | null
    password: string
    role: $Enums.UserRole
    name: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type UserUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    email?: StringFieldUpdateOperationsInput | string
    doctorId?: NullableStringFieldUpdateOperationsInput | string | null
    password?: StringFieldUpdateOperationsInput | string
    role?: EnumUserRoleFieldUpdateOperationsInput | $Enums.UserRole
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    email?: StringFieldUpdateOperationsInput | string
    doctorId?: NullableStringFieldUpdateOperationsInput | string | null
    password?: StringFieldUpdateOperationsInput | string
    role?: EnumUserRoleFieldUpdateOperationsInput | $Enums.UserRole
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type PatientCreateInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    createdAt?: Date | string
    updatedAt?: Date | string
    doctor: UserCreateNestedOneWithoutPatientsInput
    xrayScans?: XrayScanCreateNestedManyWithoutPatientInput
  }

  export type PatientUncheckedCreateInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    doctorId: string
    createdAt?: Date | string
    updatedAt?: Date | string
    xrayScans?: XrayScanUncheckedCreateNestedManyWithoutPatientInput
  }

  export type PatientUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    doctor?: UserUpdateOneRequiredWithoutPatientsNestedInput
    xrayScans?: XrayScanUpdateManyWithoutPatientNestedInput
  }

  export type PatientUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    doctorId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    xrayScans?: XrayScanUncheckedUpdateManyWithoutPatientNestedInput
  }

  export type PatientCreateManyInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    doctorId: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type PatientUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type PatientUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    doctorId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type XrayScanCreateInput = {
    id?: string
    referenceNumber: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
    patient: PatientCreateNestedOneWithoutXrayScansInput
    metadata?: ScanMetadataCreateNestedOneWithoutScanInput
  }

  export type XrayScanUncheckedCreateInput = {
    id?: string
    referenceNumber: string
    patientId: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
    metadata?: ScanMetadataUncheckedCreateNestedOneWithoutScanInput
  }

  export type XrayScanUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    patient?: PatientUpdateOneRequiredWithoutXrayScansNestedInput
    metadata?: ScanMetadataUpdateOneWithoutScanNestedInput
  }

  export type XrayScanUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    patientId?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    metadata?: ScanMetadataUncheckedUpdateOneWithoutScanNestedInput
  }

  export type XrayScanCreateManyInput = {
    id?: string
    referenceNumber: string
    patientId: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type XrayScanUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type XrayScanUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    patientId?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ScanMetadataCreateInput = {
    id?: string
    confidence: number
    pneumoniaType?: string | null
    severity?: string | null
    recommendedAction?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    scan: XrayScanCreateNestedOneWithoutMetadataInput
  }

  export type ScanMetadataUncheckedCreateInput = {
    id?: string
    scanId: string
    confidence: number
    pneumoniaType?: string | null
    severity?: string | null
    recommendedAction?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type ScanMetadataUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    confidence?: FloatFieldUpdateOperationsInput | number
    pneumoniaType?: NullableStringFieldUpdateOperationsInput | string | null
    severity?: NullableStringFieldUpdateOperationsInput | string | null
    recommendedAction?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    scan?: XrayScanUpdateOneRequiredWithoutMetadataNestedInput
  }

  export type ScanMetadataUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    scanId?: StringFieldUpdateOperationsInput | string
    confidence?: FloatFieldUpdateOperationsInput | number
    pneumoniaType?: NullableStringFieldUpdateOperationsInput | string | null
    severity?: NullableStringFieldUpdateOperationsInput | string | null
    recommendedAction?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ScanMetadataCreateManyInput = {
    id?: string
    scanId: string
    confidence: number
    pneumoniaType?: string | null
    severity?: string | null
    recommendedAction?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type ScanMetadataUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    confidence?: FloatFieldUpdateOperationsInput | number
    pneumoniaType?: NullableStringFieldUpdateOperationsInput | string | null
    severity?: NullableStringFieldUpdateOperationsInput | string | null
    recommendedAction?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ScanMetadataUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    scanId?: StringFieldUpdateOperationsInput | string
    confidence?: FloatFieldUpdateOperationsInput | number
    pneumoniaType?: NullableStringFieldUpdateOperationsInput | string | null
    severity?: NullableStringFieldUpdateOperationsInput | string | null
    recommendedAction?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type StringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type StringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type EnumUserRoleFilter<$PrismaModel = never> = {
    equals?: $Enums.UserRole | EnumUserRoleFieldRefInput<$PrismaModel>
    in?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    notIn?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    not?: NestedEnumUserRoleFilter<$PrismaModel> | $Enums.UserRole
  }

  export type DateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type PatientListRelationFilter = {
    every?: PatientWhereInput
    some?: PatientWhereInput
    none?: PatientWhereInput
  }

  export type SortOrderInput = {
    sort: SortOrder
    nulls?: NullsOrder
  }

  export type PatientOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type UserCountOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    doctorId?: SortOrder
    password?: SortOrder
    role?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type UserMaxOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    doctorId?: SortOrder
    password?: SortOrder
    role?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type UserMinOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    doctorId?: SortOrder
    password?: SortOrder
    role?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type StringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type StringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type EnumUserRoleWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.UserRole | EnumUserRoleFieldRefInput<$PrismaModel>
    in?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    notIn?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    not?: NestedEnumUserRoleWithAggregatesFilter<$PrismaModel> | $Enums.UserRole
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumUserRoleFilter<$PrismaModel>
    _max?: NestedEnumUserRoleFilter<$PrismaModel>
  }

  export type DateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type UserScalarRelationFilter = {
    is?: UserWhereInput
    isNot?: UserWhereInput
  }

  export type XrayScanListRelationFilter = {
    every?: XrayScanWhereInput
    some?: XrayScanWhereInput
    none?: XrayScanWhereInput
  }

  export type XrayScanOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type PatientCountOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    referenceNumber?: SortOrder
    dateOfBirth?: SortOrder
    doctorId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type PatientMaxOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    referenceNumber?: SortOrder
    dateOfBirth?: SortOrder
    doctorId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type PatientMinOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    referenceNumber?: SortOrder
    dateOfBirth?: SortOrder
    doctorId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type PatientScalarRelationFilter = {
    is?: PatientWhereInput
    isNot?: PatientWhereInput
  }

  export type ScanMetadataNullableScalarRelationFilter = {
    is?: ScanMetadataWhereInput | null
    isNot?: ScanMetadataWhereInput | null
  }

  export type XrayScanCountOrderByAggregateInput = {
    id?: SortOrder
    referenceNumber?: SortOrder
    patientId?: SortOrder
    imageUrl?: SortOrder
    result?: SortOrder
    status?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type XrayScanMaxOrderByAggregateInput = {
    id?: SortOrder
    referenceNumber?: SortOrder
    patientId?: SortOrder
    imageUrl?: SortOrder
    result?: SortOrder
    status?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type XrayScanMinOrderByAggregateInput = {
    id?: SortOrder
    referenceNumber?: SortOrder
    patientId?: SortOrder
    imageUrl?: SortOrder
    result?: SortOrder
    status?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type FloatFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatFilter<$PrismaModel> | number
  }

  export type XrayScanScalarRelationFilter = {
    is?: XrayScanWhereInput
    isNot?: XrayScanWhereInput
  }

  export type ScanMetadataCountOrderByAggregateInput = {
    id?: SortOrder
    scanId?: SortOrder
    confidence?: SortOrder
    pneumoniaType?: SortOrder
    severity?: SortOrder
    recommendedAction?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type ScanMetadataAvgOrderByAggregateInput = {
    confidence?: SortOrder
  }

  export type ScanMetadataMaxOrderByAggregateInput = {
    id?: SortOrder
    scanId?: SortOrder
    confidence?: SortOrder
    pneumoniaType?: SortOrder
    severity?: SortOrder
    recommendedAction?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type ScanMetadataMinOrderByAggregateInput = {
    id?: SortOrder
    scanId?: SortOrder
    confidence?: SortOrder
    pneumoniaType?: SortOrder
    severity?: SortOrder
    recommendedAction?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type ScanMetadataSumOrderByAggregateInput = {
    confidence?: SortOrder
  }

  export type FloatWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedFloatFilter<$PrismaModel>
    _min?: NestedFloatFilter<$PrismaModel>
    _max?: NestedFloatFilter<$PrismaModel>
  }

  export type PatientCreateNestedManyWithoutDoctorInput = {
    create?: XOR<PatientCreateWithoutDoctorInput, PatientUncheckedCreateWithoutDoctorInput> | PatientCreateWithoutDoctorInput[] | PatientUncheckedCreateWithoutDoctorInput[]
    connectOrCreate?: PatientCreateOrConnectWithoutDoctorInput | PatientCreateOrConnectWithoutDoctorInput[]
    createMany?: PatientCreateManyDoctorInputEnvelope
    connect?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
  }

  export type PatientUncheckedCreateNestedManyWithoutDoctorInput = {
    create?: XOR<PatientCreateWithoutDoctorInput, PatientUncheckedCreateWithoutDoctorInput> | PatientCreateWithoutDoctorInput[] | PatientUncheckedCreateWithoutDoctorInput[]
    connectOrCreate?: PatientCreateOrConnectWithoutDoctorInput | PatientCreateOrConnectWithoutDoctorInput[]
    createMany?: PatientCreateManyDoctorInputEnvelope
    connect?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
  }

  export type StringFieldUpdateOperationsInput = {
    set?: string
  }

  export type NullableStringFieldUpdateOperationsInput = {
    set?: string | null
  }

  export type EnumUserRoleFieldUpdateOperationsInput = {
    set?: $Enums.UserRole
  }

  export type DateTimeFieldUpdateOperationsInput = {
    set?: Date | string
  }

  export type PatientUpdateManyWithoutDoctorNestedInput = {
    create?: XOR<PatientCreateWithoutDoctorInput, PatientUncheckedCreateWithoutDoctorInput> | PatientCreateWithoutDoctorInput[] | PatientUncheckedCreateWithoutDoctorInput[]
    connectOrCreate?: PatientCreateOrConnectWithoutDoctorInput | PatientCreateOrConnectWithoutDoctorInput[]
    upsert?: PatientUpsertWithWhereUniqueWithoutDoctorInput | PatientUpsertWithWhereUniqueWithoutDoctorInput[]
    createMany?: PatientCreateManyDoctorInputEnvelope
    set?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    disconnect?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    delete?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    connect?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    update?: PatientUpdateWithWhereUniqueWithoutDoctorInput | PatientUpdateWithWhereUniqueWithoutDoctorInput[]
    updateMany?: PatientUpdateManyWithWhereWithoutDoctorInput | PatientUpdateManyWithWhereWithoutDoctorInput[]
    deleteMany?: PatientScalarWhereInput | PatientScalarWhereInput[]
  }

  export type PatientUncheckedUpdateManyWithoutDoctorNestedInput = {
    create?: XOR<PatientCreateWithoutDoctorInput, PatientUncheckedCreateWithoutDoctorInput> | PatientCreateWithoutDoctorInput[] | PatientUncheckedCreateWithoutDoctorInput[]
    connectOrCreate?: PatientCreateOrConnectWithoutDoctorInput | PatientCreateOrConnectWithoutDoctorInput[]
    upsert?: PatientUpsertWithWhereUniqueWithoutDoctorInput | PatientUpsertWithWhereUniqueWithoutDoctorInput[]
    createMany?: PatientCreateManyDoctorInputEnvelope
    set?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    disconnect?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    delete?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    connect?: PatientWhereUniqueInput | PatientWhereUniqueInput[]
    update?: PatientUpdateWithWhereUniqueWithoutDoctorInput | PatientUpdateWithWhereUniqueWithoutDoctorInput[]
    updateMany?: PatientUpdateManyWithWhereWithoutDoctorInput | PatientUpdateManyWithWhereWithoutDoctorInput[]
    deleteMany?: PatientScalarWhereInput | PatientScalarWhereInput[]
  }

  export type UserCreateNestedOneWithoutPatientsInput = {
    create?: XOR<UserCreateWithoutPatientsInput, UserUncheckedCreateWithoutPatientsInput>
    connectOrCreate?: UserCreateOrConnectWithoutPatientsInput
    connect?: UserWhereUniqueInput
  }

  export type XrayScanCreateNestedManyWithoutPatientInput = {
    create?: XOR<XrayScanCreateWithoutPatientInput, XrayScanUncheckedCreateWithoutPatientInput> | XrayScanCreateWithoutPatientInput[] | XrayScanUncheckedCreateWithoutPatientInput[]
    connectOrCreate?: XrayScanCreateOrConnectWithoutPatientInput | XrayScanCreateOrConnectWithoutPatientInput[]
    createMany?: XrayScanCreateManyPatientInputEnvelope
    connect?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
  }

  export type XrayScanUncheckedCreateNestedManyWithoutPatientInput = {
    create?: XOR<XrayScanCreateWithoutPatientInput, XrayScanUncheckedCreateWithoutPatientInput> | XrayScanCreateWithoutPatientInput[] | XrayScanUncheckedCreateWithoutPatientInput[]
    connectOrCreate?: XrayScanCreateOrConnectWithoutPatientInput | XrayScanCreateOrConnectWithoutPatientInput[]
    createMany?: XrayScanCreateManyPatientInputEnvelope
    connect?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
  }

  export type UserUpdateOneRequiredWithoutPatientsNestedInput = {
    create?: XOR<UserCreateWithoutPatientsInput, UserUncheckedCreateWithoutPatientsInput>
    connectOrCreate?: UserCreateOrConnectWithoutPatientsInput
    upsert?: UserUpsertWithoutPatientsInput
    connect?: UserWhereUniqueInput
    update?: XOR<XOR<UserUpdateToOneWithWhereWithoutPatientsInput, UserUpdateWithoutPatientsInput>, UserUncheckedUpdateWithoutPatientsInput>
  }

  export type XrayScanUpdateManyWithoutPatientNestedInput = {
    create?: XOR<XrayScanCreateWithoutPatientInput, XrayScanUncheckedCreateWithoutPatientInput> | XrayScanCreateWithoutPatientInput[] | XrayScanUncheckedCreateWithoutPatientInput[]
    connectOrCreate?: XrayScanCreateOrConnectWithoutPatientInput | XrayScanCreateOrConnectWithoutPatientInput[]
    upsert?: XrayScanUpsertWithWhereUniqueWithoutPatientInput | XrayScanUpsertWithWhereUniqueWithoutPatientInput[]
    createMany?: XrayScanCreateManyPatientInputEnvelope
    set?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    disconnect?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    delete?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    connect?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    update?: XrayScanUpdateWithWhereUniqueWithoutPatientInput | XrayScanUpdateWithWhereUniqueWithoutPatientInput[]
    updateMany?: XrayScanUpdateManyWithWhereWithoutPatientInput | XrayScanUpdateManyWithWhereWithoutPatientInput[]
    deleteMany?: XrayScanScalarWhereInput | XrayScanScalarWhereInput[]
  }

  export type XrayScanUncheckedUpdateManyWithoutPatientNestedInput = {
    create?: XOR<XrayScanCreateWithoutPatientInput, XrayScanUncheckedCreateWithoutPatientInput> | XrayScanCreateWithoutPatientInput[] | XrayScanUncheckedCreateWithoutPatientInput[]
    connectOrCreate?: XrayScanCreateOrConnectWithoutPatientInput | XrayScanCreateOrConnectWithoutPatientInput[]
    upsert?: XrayScanUpsertWithWhereUniqueWithoutPatientInput | XrayScanUpsertWithWhereUniqueWithoutPatientInput[]
    createMany?: XrayScanCreateManyPatientInputEnvelope
    set?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    disconnect?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    delete?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    connect?: XrayScanWhereUniqueInput | XrayScanWhereUniqueInput[]
    update?: XrayScanUpdateWithWhereUniqueWithoutPatientInput | XrayScanUpdateWithWhereUniqueWithoutPatientInput[]
    updateMany?: XrayScanUpdateManyWithWhereWithoutPatientInput | XrayScanUpdateManyWithWhereWithoutPatientInput[]
    deleteMany?: XrayScanScalarWhereInput | XrayScanScalarWhereInput[]
  }

  export type PatientCreateNestedOneWithoutXrayScansInput = {
    create?: XOR<PatientCreateWithoutXrayScansInput, PatientUncheckedCreateWithoutXrayScansInput>
    connectOrCreate?: PatientCreateOrConnectWithoutXrayScansInput
    connect?: PatientWhereUniqueInput
  }

  export type ScanMetadataCreateNestedOneWithoutScanInput = {
    create?: XOR<ScanMetadataCreateWithoutScanInput, ScanMetadataUncheckedCreateWithoutScanInput>
    connectOrCreate?: ScanMetadataCreateOrConnectWithoutScanInput
    connect?: ScanMetadataWhereUniqueInput
  }

  export type ScanMetadataUncheckedCreateNestedOneWithoutScanInput = {
    create?: XOR<ScanMetadataCreateWithoutScanInput, ScanMetadataUncheckedCreateWithoutScanInput>
    connectOrCreate?: ScanMetadataCreateOrConnectWithoutScanInput
    connect?: ScanMetadataWhereUniqueInput
  }

  export type PatientUpdateOneRequiredWithoutXrayScansNestedInput = {
    create?: XOR<PatientCreateWithoutXrayScansInput, PatientUncheckedCreateWithoutXrayScansInput>
    connectOrCreate?: PatientCreateOrConnectWithoutXrayScansInput
    upsert?: PatientUpsertWithoutXrayScansInput
    connect?: PatientWhereUniqueInput
    update?: XOR<XOR<PatientUpdateToOneWithWhereWithoutXrayScansInput, PatientUpdateWithoutXrayScansInput>, PatientUncheckedUpdateWithoutXrayScansInput>
  }

  export type ScanMetadataUpdateOneWithoutScanNestedInput = {
    create?: XOR<ScanMetadataCreateWithoutScanInput, ScanMetadataUncheckedCreateWithoutScanInput>
    connectOrCreate?: ScanMetadataCreateOrConnectWithoutScanInput
    upsert?: ScanMetadataUpsertWithoutScanInput
    disconnect?: ScanMetadataWhereInput | boolean
    delete?: ScanMetadataWhereInput | boolean
    connect?: ScanMetadataWhereUniqueInput
    update?: XOR<XOR<ScanMetadataUpdateToOneWithWhereWithoutScanInput, ScanMetadataUpdateWithoutScanInput>, ScanMetadataUncheckedUpdateWithoutScanInput>
  }

  export type ScanMetadataUncheckedUpdateOneWithoutScanNestedInput = {
    create?: XOR<ScanMetadataCreateWithoutScanInput, ScanMetadataUncheckedCreateWithoutScanInput>
    connectOrCreate?: ScanMetadataCreateOrConnectWithoutScanInput
    upsert?: ScanMetadataUpsertWithoutScanInput
    disconnect?: ScanMetadataWhereInput | boolean
    delete?: ScanMetadataWhereInput | boolean
    connect?: ScanMetadataWhereUniqueInput
    update?: XOR<XOR<ScanMetadataUpdateToOneWithWhereWithoutScanInput, ScanMetadataUpdateWithoutScanInput>, ScanMetadataUncheckedUpdateWithoutScanInput>
  }

  export type XrayScanCreateNestedOneWithoutMetadataInput = {
    create?: XOR<XrayScanCreateWithoutMetadataInput, XrayScanUncheckedCreateWithoutMetadataInput>
    connectOrCreate?: XrayScanCreateOrConnectWithoutMetadataInput
    connect?: XrayScanWhereUniqueInput
  }

  export type FloatFieldUpdateOperationsInput = {
    set?: number
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type XrayScanUpdateOneRequiredWithoutMetadataNestedInput = {
    create?: XOR<XrayScanCreateWithoutMetadataInput, XrayScanUncheckedCreateWithoutMetadataInput>
    connectOrCreate?: XrayScanCreateOrConnectWithoutMetadataInput
    upsert?: XrayScanUpsertWithoutMetadataInput
    connect?: XrayScanWhereUniqueInput
    update?: XOR<XOR<XrayScanUpdateToOneWithWhereWithoutMetadataInput, XrayScanUpdateWithoutMetadataInput>, XrayScanUncheckedUpdateWithoutMetadataInput>
  }

  export type NestedStringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type NestedStringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type NestedEnumUserRoleFilter<$PrismaModel = never> = {
    equals?: $Enums.UserRole | EnumUserRoleFieldRefInput<$PrismaModel>
    in?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    notIn?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    not?: NestedEnumUserRoleFilter<$PrismaModel> | $Enums.UserRole
  }

  export type NestedDateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type NestedStringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type NestedIntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type NestedStringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type NestedIntNullableFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableFilter<$PrismaModel> | number | null
  }

  export type NestedEnumUserRoleWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.UserRole | EnumUserRoleFieldRefInput<$PrismaModel>
    in?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    notIn?: $Enums.UserRole[] | ListEnumUserRoleFieldRefInput<$PrismaModel>
    not?: NestedEnumUserRoleWithAggregatesFilter<$PrismaModel> | $Enums.UserRole
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumUserRoleFilter<$PrismaModel>
    _max?: NestedEnumUserRoleFilter<$PrismaModel>
  }

  export type NestedDateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type NestedFloatFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatFilter<$PrismaModel> | number
  }

  export type NestedFloatWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedFloatFilter<$PrismaModel>
    _min?: NestedFloatFilter<$PrismaModel>
    _max?: NestedFloatFilter<$PrismaModel>
  }

  export type PatientCreateWithoutDoctorInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    createdAt?: Date | string
    updatedAt?: Date | string
    xrayScans?: XrayScanCreateNestedManyWithoutPatientInput
  }

  export type PatientUncheckedCreateWithoutDoctorInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    createdAt?: Date | string
    updatedAt?: Date | string
    xrayScans?: XrayScanUncheckedCreateNestedManyWithoutPatientInput
  }

  export type PatientCreateOrConnectWithoutDoctorInput = {
    where: PatientWhereUniqueInput
    create: XOR<PatientCreateWithoutDoctorInput, PatientUncheckedCreateWithoutDoctorInput>
  }

  export type PatientCreateManyDoctorInputEnvelope = {
    data: PatientCreateManyDoctorInput | PatientCreateManyDoctorInput[]
    skipDuplicates?: boolean
  }

  export type PatientUpsertWithWhereUniqueWithoutDoctorInput = {
    where: PatientWhereUniqueInput
    update: XOR<PatientUpdateWithoutDoctorInput, PatientUncheckedUpdateWithoutDoctorInput>
    create: XOR<PatientCreateWithoutDoctorInput, PatientUncheckedCreateWithoutDoctorInput>
  }

  export type PatientUpdateWithWhereUniqueWithoutDoctorInput = {
    where: PatientWhereUniqueInput
    data: XOR<PatientUpdateWithoutDoctorInput, PatientUncheckedUpdateWithoutDoctorInput>
  }

  export type PatientUpdateManyWithWhereWithoutDoctorInput = {
    where: PatientScalarWhereInput
    data: XOR<PatientUpdateManyMutationInput, PatientUncheckedUpdateManyWithoutDoctorInput>
  }

  export type PatientScalarWhereInput = {
    AND?: PatientScalarWhereInput | PatientScalarWhereInput[]
    OR?: PatientScalarWhereInput[]
    NOT?: PatientScalarWhereInput | PatientScalarWhereInput[]
    id?: StringFilter<"Patient"> | string
    name?: StringFilter<"Patient"> | string
    referenceNumber?: StringFilter<"Patient"> | string
    dateOfBirth?: DateTimeFilter<"Patient"> | Date | string
    doctorId?: StringFilter<"Patient"> | string
    createdAt?: DateTimeFilter<"Patient"> | Date | string
    updatedAt?: DateTimeFilter<"Patient"> | Date | string
  }

  export type UserCreateWithoutPatientsInput = {
    id?: string
    email: string
    doctorId?: string | null
    password: string
    role: $Enums.UserRole
    name: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type UserUncheckedCreateWithoutPatientsInput = {
    id?: string
    email: string
    doctorId?: string | null
    password: string
    role: $Enums.UserRole
    name: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type UserCreateOrConnectWithoutPatientsInput = {
    where: UserWhereUniqueInput
    create: XOR<UserCreateWithoutPatientsInput, UserUncheckedCreateWithoutPatientsInput>
  }

  export type XrayScanCreateWithoutPatientInput = {
    id?: string
    referenceNumber: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
    metadata?: ScanMetadataCreateNestedOneWithoutScanInput
  }

  export type XrayScanUncheckedCreateWithoutPatientInput = {
    id?: string
    referenceNumber: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
    metadata?: ScanMetadataUncheckedCreateNestedOneWithoutScanInput
  }

  export type XrayScanCreateOrConnectWithoutPatientInput = {
    where: XrayScanWhereUniqueInput
    create: XOR<XrayScanCreateWithoutPatientInput, XrayScanUncheckedCreateWithoutPatientInput>
  }

  export type XrayScanCreateManyPatientInputEnvelope = {
    data: XrayScanCreateManyPatientInput | XrayScanCreateManyPatientInput[]
    skipDuplicates?: boolean
  }

  export type UserUpsertWithoutPatientsInput = {
    update: XOR<UserUpdateWithoutPatientsInput, UserUncheckedUpdateWithoutPatientsInput>
    create: XOR<UserCreateWithoutPatientsInput, UserUncheckedCreateWithoutPatientsInput>
    where?: UserWhereInput
  }

  export type UserUpdateToOneWithWhereWithoutPatientsInput = {
    where?: UserWhereInput
    data: XOR<UserUpdateWithoutPatientsInput, UserUncheckedUpdateWithoutPatientsInput>
  }

  export type UserUpdateWithoutPatientsInput = {
    id?: StringFieldUpdateOperationsInput | string
    email?: StringFieldUpdateOperationsInput | string
    doctorId?: NullableStringFieldUpdateOperationsInput | string | null
    password?: StringFieldUpdateOperationsInput | string
    role?: EnumUserRoleFieldUpdateOperationsInput | $Enums.UserRole
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserUncheckedUpdateWithoutPatientsInput = {
    id?: StringFieldUpdateOperationsInput | string
    email?: StringFieldUpdateOperationsInput | string
    doctorId?: NullableStringFieldUpdateOperationsInput | string | null
    password?: StringFieldUpdateOperationsInput | string
    role?: EnumUserRoleFieldUpdateOperationsInput | $Enums.UserRole
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type XrayScanUpsertWithWhereUniqueWithoutPatientInput = {
    where: XrayScanWhereUniqueInput
    update: XOR<XrayScanUpdateWithoutPatientInput, XrayScanUncheckedUpdateWithoutPatientInput>
    create: XOR<XrayScanCreateWithoutPatientInput, XrayScanUncheckedCreateWithoutPatientInput>
  }

  export type XrayScanUpdateWithWhereUniqueWithoutPatientInput = {
    where: XrayScanWhereUniqueInput
    data: XOR<XrayScanUpdateWithoutPatientInput, XrayScanUncheckedUpdateWithoutPatientInput>
  }

  export type XrayScanUpdateManyWithWhereWithoutPatientInput = {
    where: XrayScanScalarWhereInput
    data: XOR<XrayScanUpdateManyMutationInput, XrayScanUncheckedUpdateManyWithoutPatientInput>
  }

  export type XrayScanScalarWhereInput = {
    AND?: XrayScanScalarWhereInput | XrayScanScalarWhereInput[]
    OR?: XrayScanScalarWhereInput[]
    NOT?: XrayScanScalarWhereInput | XrayScanScalarWhereInput[]
    id?: StringFilter<"XrayScan"> | string
    referenceNumber?: StringFilter<"XrayScan"> | string
    patientId?: StringFilter<"XrayScan"> | string
    imageUrl?: StringFilter<"XrayScan"> | string
    result?: StringNullableFilter<"XrayScan"> | string | null
    status?: StringFilter<"XrayScan"> | string
    createdAt?: DateTimeFilter<"XrayScan"> | Date | string
    updatedAt?: DateTimeFilter<"XrayScan"> | Date | string
  }

  export type PatientCreateWithoutXrayScansInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    createdAt?: Date | string
    updatedAt?: Date | string
    doctor: UserCreateNestedOneWithoutPatientsInput
  }

  export type PatientUncheckedCreateWithoutXrayScansInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    doctorId: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type PatientCreateOrConnectWithoutXrayScansInput = {
    where: PatientWhereUniqueInput
    create: XOR<PatientCreateWithoutXrayScansInput, PatientUncheckedCreateWithoutXrayScansInput>
  }

  export type ScanMetadataCreateWithoutScanInput = {
    id?: string
    confidence: number
    pneumoniaType?: string | null
    severity?: string | null
    recommendedAction?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type ScanMetadataUncheckedCreateWithoutScanInput = {
    id?: string
    confidence: number
    pneumoniaType?: string | null
    severity?: string | null
    recommendedAction?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type ScanMetadataCreateOrConnectWithoutScanInput = {
    where: ScanMetadataWhereUniqueInput
    create: XOR<ScanMetadataCreateWithoutScanInput, ScanMetadataUncheckedCreateWithoutScanInput>
  }

  export type PatientUpsertWithoutXrayScansInput = {
    update: XOR<PatientUpdateWithoutXrayScansInput, PatientUncheckedUpdateWithoutXrayScansInput>
    create: XOR<PatientCreateWithoutXrayScansInput, PatientUncheckedCreateWithoutXrayScansInput>
    where?: PatientWhereInput
  }

  export type PatientUpdateToOneWithWhereWithoutXrayScansInput = {
    where?: PatientWhereInput
    data: XOR<PatientUpdateWithoutXrayScansInput, PatientUncheckedUpdateWithoutXrayScansInput>
  }

  export type PatientUpdateWithoutXrayScansInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    doctor?: UserUpdateOneRequiredWithoutPatientsNestedInput
  }

  export type PatientUncheckedUpdateWithoutXrayScansInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    doctorId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ScanMetadataUpsertWithoutScanInput = {
    update: XOR<ScanMetadataUpdateWithoutScanInput, ScanMetadataUncheckedUpdateWithoutScanInput>
    create: XOR<ScanMetadataCreateWithoutScanInput, ScanMetadataUncheckedCreateWithoutScanInput>
    where?: ScanMetadataWhereInput
  }

  export type ScanMetadataUpdateToOneWithWhereWithoutScanInput = {
    where?: ScanMetadataWhereInput
    data: XOR<ScanMetadataUpdateWithoutScanInput, ScanMetadataUncheckedUpdateWithoutScanInput>
  }

  export type ScanMetadataUpdateWithoutScanInput = {
    id?: StringFieldUpdateOperationsInput | string
    confidence?: FloatFieldUpdateOperationsInput | number
    pneumoniaType?: NullableStringFieldUpdateOperationsInput | string | null
    severity?: NullableStringFieldUpdateOperationsInput | string | null
    recommendedAction?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ScanMetadataUncheckedUpdateWithoutScanInput = {
    id?: StringFieldUpdateOperationsInput | string
    confidence?: FloatFieldUpdateOperationsInput | number
    pneumoniaType?: NullableStringFieldUpdateOperationsInput | string | null
    severity?: NullableStringFieldUpdateOperationsInput | string | null
    recommendedAction?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type XrayScanCreateWithoutMetadataInput = {
    id?: string
    referenceNumber: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
    patient: PatientCreateNestedOneWithoutXrayScansInput
  }

  export type XrayScanUncheckedCreateWithoutMetadataInput = {
    id?: string
    referenceNumber: string
    patientId: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type XrayScanCreateOrConnectWithoutMetadataInput = {
    where: XrayScanWhereUniqueInput
    create: XOR<XrayScanCreateWithoutMetadataInput, XrayScanUncheckedCreateWithoutMetadataInput>
  }

  export type XrayScanUpsertWithoutMetadataInput = {
    update: XOR<XrayScanUpdateWithoutMetadataInput, XrayScanUncheckedUpdateWithoutMetadataInput>
    create: XOR<XrayScanCreateWithoutMetadataInput, XrayScanUncheckedCreateWithoutMetadataInput>
    where?: XrayScanWhereInput
  }

  export type XrayScanUpdateToOneWithWhereWithoutMetadataInput = {
    where?: XrayScanWhereInput
    data: XOR<XrayScanUpdateWithoutMetadataInput, XrayScanUncheckedUpdateWithoutMetadataInput>
  }

  export type XrayScanUpdateWithoutMetadataInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    patient?: PatientUpdateOneRequiredWithoutXrayScansNestedInput
  }

  export type XrayScanUncheckedUpdateWithoutMetadataInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    patientId?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type PatientCreateManyDoctorInput = {
    id?: string
    name: string
    referenceNumber: string
    dateOfBirth: Date | string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type PatientUpdateWithoutDoctorInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    xrayScans?: XrayScanUpdateManyWithoutPatientNestedInput
  }

  export type PatientUncheckedUpdateWithoutDoctorInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    xrayScans?: XrayScanUncheckedUpdateManyWithoutPatientNestedInput
  }

  export type PatientUncheckedUpdateManyWithoutDoctorInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    dateOfBirth?: DateTimeFieldUpdateOperationsInput | Date | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type XrayScanCreateManyPatientInput = {
    id?: string
    referenceNumber: string
    imageUrl: string
    result?: string | null
    status?: string
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type XrayScanUpdateWithoutPatientInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    metadata?: ScanMetadataUpdateOneWithoutScanNestedInput
  }

  export type XrayScanUncheckedUpdateWithoutPatientInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    metadata?: ScanMetadataUncheckedUpdateOneWithoutScanNestedInput
  }

  export type XrayScanUncheckedUpdateManyWithoutPatientInput = {
    id?: StringFieldUpdateOperationsInput | string
    referenceNumber?: StringFieldUpdateOperationsInput | string
    imageUrl?: StringFieldUpdateOperationsInput | string
    result?: NullableStringFieldUpdateOperationsInput | string | null
    status?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }



  /**
   * Batch Payload for updateMany & deleteMany & createMany
   */

  export type BatchPayload = {
    count: number
  }

  /**
   * DMMF
   */
  export const dmmf: runtime.BaseDMMF
}