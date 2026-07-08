use super::*;

pub(crate) fn processes_changed_since_conn(
    conn: &Connection,
    cursor: ProcessChangeCursor,
    limit: usize,
) -> Result<(Vec<ProcessRecord>, ProcessChangeCursor), lash_core::PluginError> {
    let mut stmt = conn
        .prepare(
            "SELECT change_seq, record_json FROM processes
             WHERE change_seq > ?1
             ORDER BY change_seq ASC, process_id ASC
             LIMIT ?2",
        )
        .map_err(process_sqlite_error)?;
    let rows = stmt
        .query_map(
            params![cursor.store_sequence() as i64, limit as i64],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(process_sqlite_error)?;
    let mut records = Vec::new();
    let mut next_cursor = cursor;
    for row in rows {
        let (change_seq, record_json) = row.map_err(process_sqlite_error)?;
        let record: ProcessRecord =
            serde_json::from_str(&record_json).map_err(process_decode_error)?;
        next_cursor = ProcessChangeCursor::from_store_sequence(change_seq as u64);
        records.push(record);
    }
    Ok((records, next_cursor))
}
