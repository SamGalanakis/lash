# Host originators carry named scopes

`ProcessOriginator::Host` was a single undifferentiated bucket: host-registered trigger
subscriptions could not be listed or cancelled (lifecycle matching keyed on session ids only),
so hosts faked per-concern grouping by registering under synthetic session ids that named no
real session — making provenance lie. We decided the Host originator carries an optional,
opaque **Host Scope** label (`Host { scope: Option<String> }`, `scope_id()` = `host` or
`host:{scope}`), and trigger-subscription list/cancel/deactivate match by registrant scope
uniformly across Host and Session originators.

The label is meaningless to lash — hosts choose their own grouping (a product might scope by
automation id, a CLI by profile). The alternatives were worse: matching the bare Host bucket
gives lifecycle over *all* host subscriptions but no grouping, so the fake-session workaround
survives; blessing synthetic sessions bakes a semantic lie into provenance that every
downstream consumer must know about. Additive and serde-compatible: an absent scope is
today's `host`.
