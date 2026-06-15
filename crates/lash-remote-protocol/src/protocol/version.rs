pub const REMOTE_PROTOCOL_VERSION: u32 = 4;

pub fn ensure_protocol_version(actual: u32) -> Result<(), RemoteProtocolError> {
    if actual == REMOTE_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(RemoteProtocolError::UnsupportedProtocolVersion {
            actual,
            expected: REMOTE_PROTOCOL_VERSION,
        })
    }
}
