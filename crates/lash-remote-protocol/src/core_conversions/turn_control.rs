impl From<lash_core::TurnCancelSource> for RemoteTurnCancelSource {
    fn from(value: lash_core::TurnCancelSource) -> Self {
        match value {
            lash_core::TurnCancelSource::UserInterrupt => Self::UserInterrupt,
            lash_core::TurnCancelSource::Host => Self::Host,
            lash_core::TurnCancelSource::Shutdown => Self::Shutdown,
            lash_core::TurnCancelSource::Superseded => Self::Superseded,
        }
    }
}

impl From<RemoteTurnCancelSource> for lash_core::TurnCancelSource {
    fn from(value: RemoteTurnCancelSource) -> Self {
        match value {
            RemoteTurnCancelSource::UserInterrupt => Self::UserInterrupt,
            RemoteTurnCancelSource::Host => Self::Host,
            RemoteTurnCancelSource::Shutdown => Self::Shutdown,
            RemoteTurnCancelSource::Superseded => Self::Superseded,
        }
    }
}

impl From<lash_core::TurnCancellationEvidence> for RemoteTurnCancellationEvidence {
    fn from(value: lash_core::TurnCancellationEvidence) -> Self {
        let lash_core::TurnCancellationEvidence {
            request_id,
            source,
            reason,
        } = value;
        Self {
            request_id,
            source: source.into(),
            reason,
        }
    }
}

impl From<RemoteTurnCancellationEvidence> for lash_core::TurnCancellationEvidence {
    fn from(value: RemoteTurnCancellationEvidence) -> Self {
        let RemoteTurnCancellationEvidence {
            request_id,
            source,
            reason,
        } = value;
        Self {
            request_id,
            source: source.into(),
            reason,
        }
    }
}

impl TryFrom<RemoteTurnCancelRequest> for lash_core::TurnCancelRequest {
    type Error = RemoteProtocolError;

    fn try_from(value: RemoteTurnCancelRequest) -> Result<Self, Self::Error> {
        value.validate()?;
        let RemoteTurnCancelRequest {
            protocol_version: _,
            session_id,
            turn_id,
            request_id,
            source,
            reason,
        } = value;
        Ok(Self {
            address: lash_core::TurnAddress::new(session_id, turn_id),
            request_id,
            source: source.into(),
            reason,
        })
    }
}

impl From<lash_core::TurnCancelRequest> for RemoteTurnCancelRequest {
    fn from(value: lash_core::TurnCancelRequest) -> Self {
        let lash_core::TurnCancelRequest {
            address,
            request_id,
            source,
            reason,
        } = value;
        Self {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            session_id: address.session_id,
            turn_id: address.turn_id,
            request_id,
            source: source.into(),
            reason,
        }
    }
}

impl From<lash_core::TurnCancelOutcome> for RemoteTurnCancelOutcome {
    fn from(value: lash_core::TurnCancelOutcome) -> Self {
        match value {
            lash_core::TurnCancelOutcome::Requested(cancellation) => Self::Requested {
                cancellation: cancellation.into(),
            },
            lash_core::TurnCancelOutcome::AlreadyRequested(cancellation) => {
                Self::AlreadyRequested {
                    cancellation: cancellation.into(),
                }
            }
            lash_core::TurnCancelOutcome::CompletionWonRace => Self::CompletionWonRace,
            lash_core::TurnCancelOutcome::UnknownOrRevoked => Self::UnknownOrRevoked,
        }
    }
}

impl From<RemoteTurnCancelOutcome> for lash_core::TurnCancelOutcome {
    fn from(value: RemoteTurnCancelOutcome) -> Self {
        match value {
            RemoteTurnCancelOutcome::Requested { cancellation } => {
                Self::Requested(cancellation.into())
            }
            RemoteTurnCancelOutcome::AlreadyRequested { cancellation } => {
                Self::AlreadyRequested(cancellation.into())
            }
            RemoteTurnCancelOutcome::CompletionWonRace => Self::CompletionWonRace,
            RemoteTurnCancelOutcome::UnknownOrRevoked => Self::UnknownOrRevoked,
        }
    }
}
