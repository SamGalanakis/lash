# Pending turn input is admission evidence

Pending Turn Input records are durable admission, claim, cancellation, and terminal evidence for user input; they are not a full in-flight turn recovery journal. Lash recovers executing turns through the scoped effect host's replay contract, while the runtime store keeps enough pending-input evidence for idempotent admission, edit/cancel reconciliation, claim fencing, and terminal outcomes. This deliberately borrows Flue's explicit admission evidence and terminal-state clarity without copying its submission-journal model, because Lash already has a separate durable effect-host boundary for in-flight turn recovery.

