use std::io::{self, Write};

use serde::Serialize;
use sha2::Digest;

struct Sha256Writer {
    hasher: sha2::Sha256,
}

impl Write for Sha256Writer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.hasher.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn stable_json_string<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize + ?Sized,
{
    serde_json::to_string(value)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

pub(crate) fn stable_json_sha256_hex<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize + ?Sized,
{
    let mut writer = Sha256Writer {
        hasher: sha2::Sha256::new(),
    };
    serde_json::to_writer(&mut writer, value)?;
    Ok(format!("{:x}", writer.hasher.finalize()))
}
