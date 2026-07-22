use sqlx::{Connection, PgConnection};

// "LASH_PGT" encoded as a positive i64. Every test process that uses the
// configured shared database must hold this session-level lock for its entire
// database interaction.
const SHARED_DATABASE_LOCK_KEY: i64 = 0x4c41_5348_5f50_4754;

pub struct SharedDatabaseLock {
    _connection: PgConnection,
}

impl SharedDatabaseLock {
    pub async fn acquire(database_url: &str) -> Self {
        let mut connection = PgConnection::connect(database_url)
            .await
            .expect("connect Postgres test advisory lock");
        sqlx::query("SELECT pg_advisory_lock($1)")
            .bind(SHARED_DATABASE_LOCK_KEY)
            .execute(&mut connection)
            .await
            .expect("acquire Postgres test advisory lock");
        Self {
            _connection: connection,
        }
    }
}
