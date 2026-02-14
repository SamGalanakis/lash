class ChatMsg {
  role string
  content string
}

// --- Context pruning types ---

// PartId is a dynamic enum — values are added at runtime via TypeBuilder.
// A placeholder value is required so the BAML compiler recognizes the type.
enum PartId {
  Placeholder @@alias("__placeholder__")
}

enum PartRole {
  Assistant
  System
}

class PartForTriage {
  id PartId       // dynamic enum, built at runtime via TypeBuilder
  role PartRole
  content string
}

class KeepDecision {
  id PartId
}

class DeleteDecision {
  id PartId
  breadcrumb string
}

class SummarizeDecision {
  id PartId
  summary string
}
