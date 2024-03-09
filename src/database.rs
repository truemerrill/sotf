use rusqlite::{Connection, Error};

static SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS generation (
    id INTEGER PRIMARY KEY,
    number INTEGER
);

CREATE TABLE IF NOT EXISTS strategy (
    id INTEGER PRIMARY KEY,
    prior BLOB,
    strategy BLOB,
    generation_id INTEGER,
    FOREIGN KEY (generation_id) REFERENCES generation(id)
);

CREATE TABLE IF NOT EXISTS strategy_data (
    id INTEGER PRIMARY KEY,
    strategy_id INTEGER,
    metric TEXT,
    value NUMERIC,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id)
);

CREATE TABLE IF NOT EXISTS generation_data (
    id INTEGER PRIMARY KEY,
    generation_id INTEGER,
    metric TEXT,
    value NUMERIC,
    FOREIGN KEY (generation_id) REFERENCES generation(id)
);
";

pub fn setup(conn: Connection) -> Result<(), Error> {
    conn.execute_batch(SCHEMA)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Result;

    #[test]
    fn test_setup() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        setup(conn)
    }
}
