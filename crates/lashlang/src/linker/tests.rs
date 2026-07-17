#[cfg(test)]
mod tests {
    use super::*;

    fn resources() -> LashlangHostCatalog {
        let mut catalog = LashlangHostCatalog::new();
        catalog.add_module_operation(
            ["tools"],
            "Tools",
            "read_file",
            "read_file",
            TypeExpr::Object(vec![TypeField {
                name: "path".into(),
                ty: TypeExpr::Str,
                optional: false,
            }]),
            TypeExpr::Str,
        );
        catalog.add_module_operation(
            ["tools"],
            "Tools",
            "echo",
            "echo",
            TypeExpr::Any,
            TypeExpr::Any,
        );
        for (operation, input_ty) in [
            ("accept_str", TypeExpr::Str),
            ("accept_int", TypeExpr::Int),
            ("accept_float", TypeExpr::Float),
            (
                "accept_mode",
                TypeExpr::Enum(vec!["default".into(), "careful".into()]),
            ),
        ] {
            catalog.add_module_operation(
                ["tools"],
                "Tools",
                operation,
                operation,
                input_ty,
                TypeExpr::Null,
            );
        }
        catalog.add_module_operation(
            ["tools"],
            "Tools",
            "accept_config",
            "accept_config",
            TypeExpr::Object(vec![TypeField {
                name: "mode".into(),
                ty: TypeExpr::Enum(vec!["default".into()]),
                optional: false,
            }]),
            TypeExpr::Null,
        );
        crate::add_trigger_resource_operations(&mut catalog);
        catalog
            .add_trigger_source_constructor(
                ["timer", "Schedule"],
                TypeExpr::Object(vec![
                    TypeField {
                        name: "expr".into(),
                        ty: TypeExpr::Str,
                        optional: false,
                    },
                    TypeField {
                        name: "tz".into(),
                        ty: TypeExpr::Str,
                        optional: true,
                    },
                ]),
                NamedDataType::object(
                    "timer.Tick",
                    vec![TypeField {
                        name: "fired_at".into(),
                        ty: TypeExpr::Str,
                        optional: false,
                    }],
                )
                .expect("valid timer tick type"),
            )
            .expect("valid timer trigger source");
        catalog
    }

    fn full_host_environment() -> LashlangHostEnvironment {
        LashlangHostEnvironment::new(resources(), LashlangAbilities::all())
    }

    fn full_label_environment() -> LashlangHostEnvironment {
        full_host_environment()
            .with_language_features(LashlangLanguageFeatures::default().with_label_annotations())
    }

    fn timer_tick_type_with_field(field: &'static str) -> NamedDataType {
        NamedDataType::object(
            "timer.Tick",
            vec![TypeField {
                name: field.into(),
                ty: TypeExpr::Str,
                optional: false,
            }],
        )
        .expect("valid timer tick type")
    }

    fn resources_with_timer_event(event_type: NamedDataType) -> LashlangHostCatalog {
        let mut catalog = LashlangHostCatalog::new();
        crate::add_trigger_resource_operations(&mut catalog);
        catalog
            .add_trigger_source_constructor(
                ["timer", "Schedule"],
                TypeExpr::Object(vec![TypeField {
                    name: "expr".into(),
                    ty: TypeExpr::Str,
                    optional: false,
                }]),
                event_type,
            )
            .expect("valid timer trigger source");
        catalog
    }

    #[test]
    fn named_host_data_type_validation_rejects_invalid_shapes() {
        let duplicate_field = NamedDataType::object(
            "timer.Tick",
            vec![
                TypeField {
                    name: "fired_at".into(),
                    ty: TypeExpr::Str,
                    optional: false,
                },
                TypeField {
                    name: "fired_at".into(),
                    ty: TypeExpr::Str,
                    optional: false,
                },
            ],
        )
        .expect_err("duplicate fields should be rejected");
        assert!(matches!(
            duplicate_field,
            NamedDataTypeError::DuplicateField { .. }
        ));

        let nested_ref = NamedDataType::object(
            "timer.Tick",
            vec![TypeField {
                name: "nested".into(),
                ty: TypeExpr::Ref("Other.Type".into()),
                optional: false,
            }],
        )
        .expect_err("nested refs should be rejected");
        assert!(matches!(nested_ref, NamedDataTypeError::NestedRef { .. }));

        let duplicate_enum = NamedDataType::object(
            "timer.Tick",
            vec![TypeField {
                name: "kind".into(),
                ty: TypeExpr::Enum(vec!["Red".into(), "Red".into()]),
                optional: false,
            }],
        )
        .expect_err("duplicate enum values should be rejected");
        assert!(matches!(
            duplicate_enum,
            NamedDataTypeError::DuplicateEnumValue { .. }
        ));

        let simple_name = NamedDataType::object("Tick", vec![])
            .expect_err("host data type names must be qualified");
        assert!(matches!(
            simple_name,
            NamedDataTypeError::InvalidName { .. }
        ));
    }

    #[test]
    fn resource_catalog_rejects_conflicting_named_host_data_type_definitions() {
        let mut catalog = LashlangHostCatalog::new();
        catalog
            .add_named_data_type(timer_tick_type_with_field("fired_at"))
            .expect("first definition");
        let err = catalog
            .add_named_data_type(timer_tick_type_with_field("delivered_at"))
            .expect_err("same host type name with different shape should be rejected");

        assert!(matches!(
            err,
            LashlangHostCatalogError::ConflictingNamedDataType { .. }
        ));
    }

    #[test]
    fn linked_module_accepts_named_processes_resource_params_and_activations() {
        let program = crate::parse(
            r#"
            type ChangeEvent = { path: str }
            process scan(tool: Tools, event: ChangeEvent) {
              text = await tool.read_file({ path: "changed.txt" })?
              finish text
            }
            process watcher(run: any) signals { ready: any } {
              sleep for "0ms"
              signal = wait_signal("ready")
              signal_run(run, "ready", signal)
              finish signal
            }
            process from_tick(tick: timer.Tick) {
              finish tick.fired_at
            }
            source = timer.Schedule({ expr: "0 8 * * *", tz: "UTC" })
            handle = await triggers.register({
              source: source,
              target: from_tick,
              inputs: { tick: trigger.event },
              name: "changed"
            })?
            finish handle
            "#,
        )
        .expect("parse module");

        let linked = LinkedModule::link(program, full_host_environment()).expect("link module");

        assert!(
            linked
                .module_ref
                .as_str()
                .starts_with("lashlang:v1:sha256:")
        );
    }

    #[test]
    fn linked_module_allows_trigger_registration_name_to_match_target_process() {
        let program = crate::parse(
            r#"
            process changed(tick: timer.Tick) {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: changed,
              inputs: { tick: trigger.event },
              name: "changed"
            })?
            "#,
        )
        .expect("parse module");

        LinkedModule::link(program, full_host_environment())
            .expect("trigger registration names and process names occupy different namespaces");
    }

    #[test]
    fn linked_module_resolves_host_named_data_refs_for_fields_and_structural_assignability() {
        let direct_ref = crate::parse(
            r#"
            process from_tick(tick: timer.Tick) {
              finish tick.fired_at
            }
            finish true
            "#,
        )
        .expect("parse direct host data ref");
        LinkedModule::link(direct_ref, full_host_environment())
            .expect("host data ref fields should link");

        let structural_input = crate::parse(
            r#"
            process from_tick(tick: { fired_at: str }) {
              finish tick.fired_at
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: from_tick,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse structural target input");
        LinkedModule::link(structural_input, full_host_environment())
            .expect("host data shape should be structurally assignable");
    }

    #[test]
    fn linked_module_rejects_unknown_host_data_refs_and_opaque_source_field_access() {
        let unknown = crate::parse(
            r#"
            process from_tick(tick: foo.Tick) {
              finish true
            }
            finish true
            "#,
        )
        .expect("parse unknown host type");
        assert!(matches!(
            LinkedModule::link(unknown, full_host_environment()),
            Err(LinkError::UnknownType { name, .. }) if name == "foo.Tick"
        ));

        let opaque = crate::parse(
            r#"
            source = timer.Schedule({ expr: "0 8 * * *" })
            finish source.expr
            "#,
        )
        .expect("parse opaque source access");
        assert!(matches!(
            LinkedModule::link(opaque, full_host_environment()),
            Err(LinkError::OpaqueHostDescriptorAccess { type_name, .. }) if type_name == "timer.Schedule"
        ));
    }

    #[test]
    fn host_requirements_ref_tracks_host_named_data_type_shape_changes() {
        let program = crate::parse(
            r#"
            process from_tick(tick: any) {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: from_tick,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse trigger registration");
        let first = LinkedModule::link(
            program.clone(),
            LashlangHostEnvironment::new(
                resources_with_timer_event(timer_tick_type_with_field("fired_at")),
                LashlangAbilities::all(),
            ),
        )
        .expect("link first trigger occurrence shape");
        let second = LinkedModule::link(
            program,
            LashlangHostEnvironment::new(
                resources_with_timer_event(timer_tick_type_with_field("delivered_at")),
                LashlangAbilities::all(),
            ),
        )
        .expect("link changed trigger occurrence shape");

        assert_ne!(first.host_requirements_ref, second.host_requirements_ref);
    }

    #[test]
    fn linked_module_accepts_top_level_sleep() {
        let program = crate::parse("sleep for 1").expect("parse sleep");

        LinkedModule::link(program, full_host_environment()).expect("top-level sleep should link");
    }

    #[test]
    fn linked_module_rejects_process_lifecycle_outside_process_body() {
        let program = crate::parse("payload = wait_signal(\"ready\")").expect("parse wait_signal");

        let err = LinkedModule::link(program, full_host_environment())
            .expect_err("top-level process lifecycle should be rejected");

        assert!(
            matches!(
                err,
                LinkError::ProcessLifecycleOutsideProcess {
                    keyword: "wait_signal",
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn linked_module_accepts_top_level_signal_run() {
        // `signal_run` (sending) mirrors `await` / `cancel`: legal from the
        // foreground turn, unlike the process-only `wait_signal`.
        let program =
            crate::parse("signal_run(\"handle\", \"ready\", \"ping\")").expect("parse signal_run");

        LinkedModule::link(program, full_host_environment())
            .expect("top-level signal_run should link");
    }

    #[test]
    fn linked_module_rejects_bad_process_args_and_unresolved_operations() {
        let missing_arg = crate::parse(
            r#"
            process scan(tool: Tools, path: str) { finish path }
            start scan(tool: tools)
            "#,
        )
        .expect("parse missing arg");
        assert!(matches!(
            LinkedModule::link(missing_arg, full_host_environment()),
            Err(LinkError::MissingProcessArgument { arg, .. }) if arg == "path"
        ));

        let bad_operation = crate::parse(
            r#"
            process scan(tool: Tools) {
              finish await tool.missing({})?
            }
            "#,
        )
        .expect("parse bad operation");
        assert!(matches!(
            LinkedModule::link(bad_operation, full_host_environment()),
            Err(LinkError::UnknownResourceOperation { operation, .. }) if operation == "missing"
        ));
    }

    #[test]
    fn linked_module_rejects_disabled_abilities() {
        let process =
            crate::parse("process worker() { finish null }").expect("parse disabled process");
        assert!(matches!(
            LinkedModule::link(
                process,
                LashlangHostEnvironment::new(resources(), LashlangAbilities::default())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "processes",
                ..
            })
        ));

        let start = crate::parse("start worker()").expect("parse disabled start");
        assert!(matches!(
            LinkedModule::link(
                start,
                LashlangHostEnvironment::new(resources(), LashlangAbilities::default())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "processes",
                ..
            })
        ));

        let sleep = crate::parse("sleep for \"1s\"").expect("parse disabled sleep");
        assert!(matches!(
            LinkedModule::link(
                sleep,
                LashlangHostEnvironment::new(resources(), LashlangAbilities::default())
            ),
            Err(LinkError::FeatureDisabled {
                feature: "sleep",
                ..
            })
        ));

        let signal = crate::parse(
            "process worker() signals { ready: any } { payload = wait_signal(\"ready\") }",
        )
        .expect("parse disabled process signal");
        assert!(matches!(
            LinkedModule::link(
                signal,
                LashlangHostEnvironment::new(
                    resources(),
                    LashlangAbilities::default().with_processes()
                )
            ),
            Err(LinkError::FeatureDisabled {
                feature: "process signals",
                ..
            })
        ));

        let trigger = crate::parse(
            r#"
            process worker(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: worker,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse disabled trigger");
        assert!(matches!(
            LinkedModule::link(
                trigger,
                LashlangHostEnvironment::new(
                    resources(),
                    LashlangAbilities::default().with_processes()
                )
            ),
            Err(LinkError::FeatureDisabled {
                feature: "triggers",
                ..
            })
        ));
    }

    #[test]
    fn linked_module_validates_value_constructors_and_trigger_registry_ops() {
        let program = crate::parse(
            r#"
            process scan(tick: timer.Tick) -> bool {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *", tz: "UTC" })
            handle = await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event },
              name: "scan"
            })?
            registrations = await triggers.list({ target: scan })?
            cancelled = await triggers.cancel({ handle: handle })?
            finish { handle: handle, registrations: registrations, cancelled: cancelled }
            "#,
        )
        .expect("parse trigger registry program");
        assert!(LinkedModule::link(program, full_host_environment()).is_ok());
    }

    #[test]
    fn linked_module_accepts_explicit_trigger_input_mappings() {
        let repeated_event = crate::parse(
            r#"
            process scan(a: timer.Tick, b: { fired_at: str }) {
              finish { a: a.fired_at, b: b.fired_at }
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { a: trigger.event, b: trigger.event }
            })?
            "#,
        )
        .expect("parse repeated event mapping");
        LinkedModule::link(repeated_event, full_host_environment())
            .expect("event payload should map to multiple assignable params");

        let fixed_authority = crate::parse(
            r#"
            process scan(tick: timer.Tick, tool: Tools) {
              text = await tool.read_file({ path: tick.fired_at })?
              finish text
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event, tool: tools }
            })?
            "#,
        )
        .expect("parse fixed authority mapping");
        LinkedModule::link(fixed_authority, full_host_environment())
            .expect("fixed resource inputs should satisfy process authority params");
    }

    #[test]
    fn linked_module_captures_concrete_process_body_resources_statically() {
        let program = crate::parse(
            r#"
            process scan(tick: timer.Tick) {
              text = await tools.read_file({ path: tick.fired_at })?
              finish text
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse captured authority process");
        let linked = LinkedModule::link(program, full_host_environment())
            .expect("process body should capture concrete host resources");
        let process = linked
            .artifact
            .canonical_ir
            .process("scan")
            .expect("scan process");
        fn contains_resource_ref(expr: &Expr, path: &str) -> bool {
            matches!(expr, Expr::ResourceRef(resource) if resource.path_string() == path)
                || expr
                    .children()
                    .any(|child| contains_resource_ref(child, path))
        }
        assert!(
            contains_resource_ref(&process.body, "tools"),
            "linked process body should contain a persisted tools resource ref"
        );

        let shadowed = crate::parse(
            r#"
            tool = tools
            process scan(tick: timer.Tick) {
              text = await tool.read_file({ path: tick.fired_at })?
              finish text
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse foreground variable capture");
        assert!(matches!(
            LinkedModule::link(shadowed, full_host_environment()),
            Err(LinkError::UnknownName { name, .. }) if name == "tool"
        ));
    }

    #[test]
    fn linked_module_accepts_button_trigger_source_constructor() {
        let mut resources = resources();
        resources
            .add_trigger_source_constructor(
                ["ui", "button", "pressed"],
                TypeExpr::Object(vec![]),
                NamedDataType::object(
                    "ui.button.Pressed",
                    vec![
                        TypeField {
                            name: "button".into(),
                            ty: TypeExpr::Union(vec![
                                TypeExpr::Enum(vec!["Red".into()]),
                                TypeExpr::Enum(vec!["Blue".into()]),
                            ]),
                            optional: false,
                        },
                        TypeField {
                            name: "message".into(),
                            ty: TypeExpr::Str,
                            optional: false,
                        },
                        TypeField {
                            name: "pressed_at".into(),
                            ty: TypeExpr::Str,
                            optional: false,
                        },
                    ],
                )
                .expect("valid button event type"),
            )
            .expect("valid button trigger source");
        let program = crate::parse(
            r#"
            process on_button(event: ui.button.Pressed) {
              wake { kind: "button_pressed", button: event.button, message: event.message }
              finish true
            }

            handle = await triggers.register({
              source: ui.button.pressed({}),
              target: on_button,
              inputs: { event: trigger.event },
              name: "button watcher"
            })?
            finish handle
            "#,
        )
        .expect("parse button trigger source");

        LinkedModule::link(
            program,
            LashlangHostEnvironment::new(resources, LashlangAbilities::all()),
        )
        .expect("button trigger source should link");
    }

    #[test]
    fn linked_module_rejects_bad_trigger_registry_bindings() {
        let missing = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({ target: scan })?
            "#,
        )
        .expect("parse missing source");
        assert!(matches!(
            LinkedModule::link(missing, full_host_environment()),
            Err(LinkError::InvalidTriggerRegistration { .. })
        ));

        let missing_inputs = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({ source: source, target: scan })?
            "#,
        )
        .expect("parse missing inputs");
        assert!(matches!(
            LinkedModule::link(missing_inputs, full_host_environment()),
            Err(LinkError::InvalidTriggerRegistration { .. })
        ));

        let wrong_source = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            await triggers.register({
              source: { expr: "0 8 * * *" },
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse wrong source");
        assert!(matches!(
            LinkedModule::link(wrong_source, full_host_environment()),
            Err(LinkError::UnknownTriggerEventType { .. })
        ));

        let payload_mismatch = crate::parse(
            r#"
            process scan(tick: str) { finish tick }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse payload mismatch");
        assert!(matches!(
            LinkedModule::link(payload_mismatch, full_host_environment()),
            Err(LinkError::TriggerEventMismatch { .. })
        ));

        let unknown_input = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event, extra: "nope" }
            })?
            "#,
        )
        .expect("parse unknown input");
        assert!(matches!(
            LinkedModule::link(unknown_input, full_host_environment()),
            Err(LinkError::UnknownTriggerInput { input, .. }) if input == "extra"
        ));

        let duplicate_input = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event, tick: trigger.event }
            })?
            "#,
        )
        .expect("parse duplicate input");
        assert!(matches!(
            LinkedModule::link(duplicate_input, full_host_environment()),
            Err(LinkError::DuplicateTriggerInput { input, .. }) if input == "tick"
        ));

        let no_event_input = crate::parse(
            r#"
            process scan(tick: timer.Tick, label: str) { finish label }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: { fired_at: "static" }, label: "static" }
            })?
            "#,
        )
        .expect("parse no event input");
        assert!(matches!(
            LinkedModule::link(no_event_input, full_host_environment()),
            Err(LinkError::MissingTriggerEventInput { .. })
        ));

        let event_projection = crate::parse(
            r#"
            process scan(fired_at: str) { finish fired_at }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { fired_at: trigger.event.fired_at }
            })?
            "#,
        )
        .expect("parse event projection");
        assert!(matches!(
            LinkedModule::link(event_projection, full_host_environment()),
            Err(LinkError::TriggerEventProjection { .. })
        ));

        let event_outside_inputs = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            finish trigger.event
            "#,
        )
        .expect("parse event outside inputs");
        assert!(matches!(
            LinkedModule::link(event_outside_inputs, full_host_environment()),
            Err(LinkError::TriggerEventOutsideInputs { .. })
        ));

        let multi_input = crate::parse(
            r#"
            process scan(tick: timer.Tick, extra: str) { finish extra }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: scan,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse multi-input target");
        assert!(matches!(
            LinkedModule::link(multi_input, full_host_environment()),
            Err(LinkError::MissingTriggerInput { input, .. }) if input == "extra"
        ));

        let target_is_not_process = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: source,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse non-process target");
        assert!(matches!(
            LinkedModule::link(target_is_not_process, full_host_environment()),
            Err(LinkError::InvalidTriggerTarget { .. })
        ));

        let list_without_filters = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            await triggers.list({})?
            "#,
        )
        .expect("parse trigger list without filters");
        assert!(LinkedModule::link(list_without_filters, full_host_environment()).is_ok());

        let list_with_filters = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            await triggers.list({
              target: scan,
              name: "daily",
              source_type: "timer.Schedule",
              enabled: true
            })?
            "#,
        )
        .expect("parse trigger list filters");
        assert!(LinkedModule::link(list_with_filters, full_host_environment()).is_ok());

        let list_target_is_not_process = crate::parse(
            r#"
            process scan(tick: timer.Tick) { finish true }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.list({ target: source })?
            "#,
        )
        .expect("parse trigger list non-process target");
        assert!(matches!(
            LinkedModule::link(list_target_is_not_process, full_host_environment()),
            Err(LinkError::InvalidTriggerTarget { .. })
                | Err(LinkError::IncompatibleOperationInput { .. })
        ));

        let constructor_mismatch = crate::parse(
            r#"
            source = timer.Schedule({ expr: 1 })
            finish source
            "#,
        )
        .expect("parse constructor mismatch");
        assert!(matches!(
            LinkedModule::link(constructor_mismatch, full_host_environment()),
            Err(LinkError::IncompatibleConstructorInput { .. })
        ));

        let operation_mismatch = crate::parse(
            r#"
            await tools.read_file({ path: 1 })?
            "#,
        )
        .expect("parse operation mismatch");
        assert!(matches!(
            LinkedModule::link(operation_mismatch, full_host_environment()),
            Err(LinkError::IncompatibleOperationInput { .. })
        ));
    }

    #[test]
    fn linked_module_infers_process_output_and_validates_return_annotations() {
        let inferred = crate::parse(
            r#"
            process done(tick: timer.Tick) -> bool {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *" })
            await triggers.register({
              source: source,
              target: done,
              inputs: { tick: trigger.event }
            })?
            "#,
        )
        .expect("parse inferred output");
        assert!(LinkedModule::link(inferred, full_host_environment()).is_ok());

        let union_mismatch = crate::parse(
            r#"
            process done(tick: timer.Tick) -> bool {
              if true {
                finish true
              }
              finish "done"
            }
            "#,
        )
        .expect("parse union mismatch");
        assert!(matches!(
            LinkedModule::link(union_mismatch, full_host_environment()),
            Err(LinkError::IncompatibleProcessReturn { .. })
        ));
    }

    #[test]
    fn linked_module_hash_ignores_unused_host_abilities() {
        let program = crate::parse("finish 1").expect("parse");
        let minimal = LinkedModule::link(
            program.clone(),
            LashlangHostEnvironment::new(resources(), LashlangAbilities::default()),
        )
        .expect("link minimal");
        let processes = LinkedModule::link(
            program,
            LashlangHostEnvironment::new(
                resources(),
                LashlangAbilities::default().with_processes(),
            ),
        )
        .expect("link process ability");

        assert_eq!(minimal.module_ref, processes.module_ref);
        assert_eq!(
            minimal.host_requirements_ref,
            processes.host_requirements_ref
        );
    }

    #[test]
    fn label_annotations_require_enabled_language_feature() {
        let program = crate::parse(
            r#"
            @label(title: "Scan files")
            process scan(tool: Tools) {
              @label(title: "Read file")
              text = await tool.read_file({ path: "." })?
              finish text
            }
            "#,
        )
        .expect("parse annotated process");

        let err = LinkedModule::link(program.clone(), full_host_environment())
            .expect_err("default surface should reject label annotations");
        assert!(matches!(
            err,
            LinkError::FeatureDisabled {
                feature: "label annotations",
                ..
            }
        ));

        let linked = LinkedModule::link(program, full_label_environment())
            .expect("enabled surface should link");
        assert!(
            linked
                .artifact
                .host_requirements
                .language_features
                .label_annotations
        );
        let process = linked.program().process("scan").expect("linked process");
        assert_eq!(
            process.label.as_ref().map(|label| label.title.as_str()),
            Some("Scan files")
        );
    }

    #[test]
    fn label_annotation_text_inside_strings_does_not_require_feature() {
        let linked = LinkedModule::link(
            crate::parse(r####"finish r"""@label(title: "Plain text")""""####)
                .expect("parse string"),
            full_host_environment(),
        )
        .expect("disabled label annotations should not reject string text");

        assert!(
            !linked
                .artifact
                .host_requirements
                .language_features
                .label_annotations
        );
    }

    #[test]
    fn label_metadata_round_trips_and_changes_artifact_identity() {
        let first = LinkedModule::link(
            crate::parse(
                r#"
                @label(title: "Scan files")
                process scan(tool: Tools) {
                  @label(title: "Read file", description: "Load source text")
                  text = await tool.read_file({ path: "." })?
                  @label(title: "Finish")
                  finish text
                }
                "#,
            )
            .expect("parse first"),
            full_label_environment(),
        )
        .expect("link first");
        let changed = LinkedModule::link(
            crate::parse(
                r#"
                @label(title: "Scan files")
                process scan(tool: Tools) {
                  @label(title: "Read source", description: "Load source text")
                  text = await tool.read_file({ path: "." })?
                  @label(title: "Finish")
                  finish text
                }
                "#,
            )
            .expect("parse changed"),
            full_label_environment(),
        )
        .expect("link changed");

        let bytes = first
            .artifact
            .to_store_bytes()
            .expect("encode annotated artifact");
        let decoded = ModuleArtifact::from_store_bytes(&bytes).expect("decode annotated artifact");
        assert_eq!(decoded, first.artifact);
        assert_ne!(first.module_ref, changed.module_ref);
        assert_ne!(
            first.artifact.process_ref("scan"),
            changed.artifact.process_ref("scan")
        );
    }

    #[test]
    fn module_ref_ignores_spans_and_formatting() {
        let compact = LinkedModule::link(
            crate::parse("process scan(root: str) { finish root }").expect("parse compact"),
            full_host_environment(),
        )
        .expect("link compact");
        let formatted = LinkedModule::link(
            crate::parse(
                r#"
                process scan(root: str) {
                    finish root
                }
                "#,
            )
            .expect("parse formatted"),
            full_host_environment(),
        )
        .expect("link formatted");

        assert_eq!(compact.module_ref, formatted.module_ref);
    }

    #[test]
    fn process_ref_tracks_abi_and_body_but_not_local_binder_names() {
        let original = LinkedModule::link(
            crate::parse("process scan(root: str) { value = root\nfinish value }")
                .expect("parse original"),
            full_host_environment(),
        )
        .expect("link original");
        let renamed_local = LinkedModule::link(
            crate::parse("process scan(root: str) { renamed = root\nfinish renamed }")
                .expect("parse renamed local"),
            full_host_environment(),
        )
        .expect("link renamed local");
        let renamed_param = LinkedModule::link(
            crate::parse("process scan(path: str) { value = path\nfinish value }")
                .expect("parse renamed param"),
            full_host_environment(),
        )
        .expect("link renamed param");
        let changed_body = LinkedModule::link(
            crate::parse("process scan(root: str) { value = root\nfinish { value: value } }")
                .expect("parse changed body"),
            full_host_environment(),
        )
        .expect("link changed body");

        assert_eq!(
            original.artifact.process_ref("scan"),
            renamed_local.artifact.process_ref("scan")
        );
        assert_ne!(
            original.artifact.process_ref("scan"),
            renamed_param.artifact.process_ref("scan")
        );
        assert_ne!(
            original.artifact.process_ref("scan"),
            changed_body.artifact.process_ref("scan")
        );
    }

    #[test]
    fn host_requirements_ref_tracks_resource_requirements_not_unrelated_tools() {
        let mut with_extra = resources();
        with_extra.add_module_operation(
            ["tools"],
            "Tools",
            "unrelated",
            "unrelated",
            TypeExpr::Any,
            TypeExpr::Any,
        );
        let program = crate::parse(
            "process scan(tool: Tools) { finish (await tool.read_file({ path: \".\" }))? }",
        )
        .expect("parse process");

        let base = LinkedModule::link(program.clone(), full_host_environment()).expect("link base");
        let extra = LinkedModule::link(
            program.clone(),
            LashlangHostEnvironment::new(with_extra, LashlangAbilities::all()),
        )
        .expect("link extra");
        let changed_requirement = LinkedModule::link(
            crate::parse(
                "process scan(tool: Tools) { finish (await tool.echo({ value: \".\" }))? }",
            )
            .expect("parse changed resource"),
            full_host_environment(),
        )
        .expect("link changed requirement");

        assert_eq!(base.module_ref, extra.module_ref);
        assert_eq!(base.host_requirements_ref, extra.host_requirements_ref);
        assert_ne!(
            base.host_requirements_ref,
            changed_requirement.host_requirements_ref
        );
    }

    #[test]
    fn module_aliases_sharing_resource_type_route_to_distinct_host_operations() {
        let mut catalog = LashlangHostCatalog::new();
        catalog.add_module_operation(
            ["inbox", "work"],
            "Inbox",
            "send",
            "inbox__work__send",
            TypeExpr::Any,
            TypeExpr::Any,
        );
        catalog.add_module_operation(
            ["inbox", "personal"],
            "Inbox",
            "send",
            "inbox__personal__send",
            TypeExpr::Any,
            TypeExpr::Any,
        );

        assert_eq!(
            catalog
                .resolve_module_operation("Inbox", "inbox.work", "send")
                .map(|binding| binding.host_operation.as_str()),
            Some("inbox__work__send")
        );
        assert_eq!(
            catalog
                .resolve_module_operation("Inbox", "inbox.personal", "send")
                .map(|binding| binding.host_operation.as_str()),
            Some("inbox__personal__send")
        );
    }

    #[test]
    fn reusing_module_alias_for_different_resource_type_fails() {
        let mut catalog = LashlangHostCatalog::new();
        catalog
            .add_module_instance(["tools"], "Tools")
            .expect("initial module instance");

        assert!(matches!(
            catalog.add_module_instance(["tools"], "Inbox"),
            Err(LashlangHostCatalogError::ConflictingModuleInstance {
                alias,
                existing,
                incoming,
            }) if alias == "tools" && existing == "Tools" && incoming == "Inbox"
        ));
    }

    // --- behaviour-pinning tests for the single linking walk -------------
    //
    // These lock in the error *set*, *ordering*, and *spans* the linker
    // produced when validation and lowering were two separate passes, so the
    // fold into one walk stays behaviour-preserving.

    #[test]
    fn declaration_errors_report_before_main_errors() {
        // The process body references an unknown name AND the main block
        // references a different unknown name. The declaration error must win.
        let program = crate::parse(
            r#"
            process scan() { finish missing_in_body }
            finish missing_in_main
            "#,
        )
        .expect("parse");
        let err = LinkedModule::link(program, full_host_environment())
            .expect_err("both bodies reference unknowns");
        assert!(
            matches!(&err, LinkError::UnknownName { name, .. } if name == "missing_in_body"),
            "{err:?}"
        );
    }

    #[test]
    fn unknown_name_in_process_body_carries_declaration_span() {
        let program = crate::parse("process scan() { finish missing }").expect("parse");
        let err = LinkedModule::link(program, full_host_environment()).expect_err("unknown name");
        let LinkError::UnknownName { name, span } = &err else {
            panic!("expected UnknownName, got {err:?}");
        };
        assert_eq!(name, "missing");
        assert!(span.is_some(), "declaration-body error should carry a span");
    }

    #[test]
    fn linker_reproduces_full_error_set() {
        // One representative source per error variant that the expression walk
        // is responsible for raising.
        // Top-level scope allows unknown globals (they become runtime errors),
        // so unknown-name checks must be exercised inside a process body.
        type ErrorCase = (&'static str, fn(&LinkError) -> bool);
        let cases: &[ErrorCase] = &[
            (
                "process scan() { finish missing }",
                |err| matches!(err, LinkError::UnknownName { name, .. } if name == "missing"),
            ),
            (
                "process scan() { missing[0] = 1 }",
                |err| matches!(err, LinkError::UnknownName { name, .. } if name == "missing"),
            ),
            (
                "finish not_a_builtin(1)",
                |err| matches!(err, LinkError::UnknownBuiltin { name, .. } if name == "not_a_builtin"),
            ),
            (
                "x = 1\nfinish x.read_file({})",
                |err| matches!(err, LinkError::UnresolvedReceiver { operation, .. } if operation == "read_file"),
            ),
            (
                "process scan() { finish 1 }\nstart scan(extra: 1)",
                |err| matches!(err, LinkError::UnexpectedProcessArgument { arg, .. } if arg == "extra"),
            ),
            (
                "process scan(needed: str) { finish needed }\nstart scan()",
                |err| matches!(err, LinkError::MissingProcessArgument { arg, .. } if arg == "needed"),
            ),
            (
                "start ghost()",
                |err| matches!(err, LinkError::UnknownProcess { name, .. } if name == "ghost"),
            ),
        ];

        for (source, predicate) in cases {
            let program =
                crate::parse(source).unwrap_or_else(|err| panic!("parse {source:?}: {err}"));
            let err = LinkedModule::link(program, full_host_environment())
                .err()
                .unwrap_or_else(|| panic!("{source:?} should fail to link"));
            assert!(predicate(&err), "unexpected error for {source:?}: {err:?}");
        }
    }

    #[test]
    fn unknown_resource_operation_still_rejected_after_receiver_resolves() {
        let program = crate::parse(
            r#"
            process scan(tool: Tools) { finish await tool.does_not_exist({})? }
            "#,
        )
        .expect("parse");
        let err =
            LinkedModule::link(program, full_host_environment()).expect_err("operation missing");
        assert!(
            matches!(&err, LinkError::UnknownResourceOperation { operation, .. } if operation == "does_not_exist"),
            "{err:?}"
        );
    }

    #[test]
    fn expected_enum_slots_reject_wrong_literals_but_admit_members_and_broad_strings() {
        let wrong = crate::parse(r#"await tools.accept_mode("nope")?"#)
            .expect("parse wrong enum literal");
        assert!(matches!(
            LinkedModule::link(wrong, full_host_environment()),
            Err(LinkError::IncompatibleExpectedLiteral { .. })
        ));

        let member = crate::parse(r#"await tools.accept_mode("default")?"#)
            .expect("parse enum member");
        LinkedModule::link(member, full_host_environment()).expect("enum member should link");

        let broad = crate::parse(
            r#"
            process forward(mode: str) {
              await tools.accept_mode(mode)?
            }
            "#,
        )
        .expect("parse broad string input");
        LinkedModule::link(broad, full_host_environment())
            .expect("broad strings remain gradually consistent with enums");

        let nested = crate::parse(r#"await tools.accept_config({ mode: "nope" })?"#)
            .expect("parse nested wrong enum literal");
        assert!(matches!(
            LinkedModule::link(nested, full_host_environment()),
            Err(LinkError::IncompatibleExpectedLiteral { .. })
        ));

        let container = crate::parse(
            r#"
            process mutate(state: { mode: enum["default"] }) {
              state.mode = "nope"
            }
            "#,
        )
        .expect("parse container expected type");
        assert!(matches!(
            LinkedModule::link(container, full_host_environment()),
            Err(LinkError::IncompatibleExpectedLiteral { .. })
        ));

        let declared_return = crate::parse(
            r#"process choose() -> enum["default"] { finish "nope" }"#,
        )
        .expect("parse declared enum return");
        assert!(matches!(
            LinkedModule::link(declared_return, full_host_environment()),
            Err(LinkError::IncompatibleExpectedLiteral { .. })
        ));

        let declared_argument = crate::parse(
            r#"
            process select(mode: enum["default"]) { finish mode }
            start select(mode: "nope")
            "#,
        )
        .expect("parse declared enum argument");
        assert!(matches!(
            LinkedModule::link(declared_argument, full_host_environment()),
            Err(LinkError::IncompatibleExpectedLiteral { .. })
        ));
    }

    #[test]
    fn union_expected_types_do_not_treat_dict_as_accepting_string_literals() {
        let program = crate::parse(
            r#"
            process select(mode: enum["a"] | dict) { finish mode }
            start select(mode: "nope")
            "#,
        )
        .expect("parse union expected type");

        assert!(matches!(
            LinkedModule::link(program, full_host_environment()),
            Err(LinkError::IncompatibleExpectedLiteral { .. })
        ));
    }

    #[test]
    fn branch_assignments_join_to_a_union_instead_of_first_wins() {
        let program = crate::parse(
            r#"
            process choose(flag: bool) {
              value = "initial"
              if flag { value = "text" } else { value = 1 }
              await tools.accept_str(value)?
            }
            "#,
        )
        .expect("parse branch join");

        assert!(matches!(
            LinkedModule::link(program, full_host_environment()),
            Err(LinkError::IncompatibleOperationInput { actual, .. }) if actual.contains('|')
        ));
    }

    #[test]
    fn for_bindings_use_list_elements_and_unknown_iterables_remain_gradual() {
        let known = crate::parse(
            r#"
            process consume() {
              for item in ["one", "two"] {
                await tools.accept_int(item)?
              }
            }
            "#,
        )
        .expect("parse known iterable");
        assert!(matches!(
            LinkedModule::link(known, full_host_environment()),
            Err(LinkError::IncompatibleOperationInput { .. })
        ));

        let unknown = crate::parse(
            r#"
            process consume(items: any) {
              for item in items {
                await tools.accept_int(item)?
              }
            }
            "#,
        )
        .expect("parse unknown iterable");
        LinkedModule::link(unknown, full_host_environment())
            .expect("unknown iterable elements should remain gradual");
    }

    #[test]
    fn awaited_process_handles_carry_the_inferred_process_output() {
        let program = crate::parse(
            r#"
            process child() { finish { value: "text" } }
            result = await start child()
            await tools.accept_int(result.value)?
            "#,
        )
        .expect("parse awaited process output");

        assert!(matches!(
            LinkedModule::link(program, full_host_environment()),
            Err(LinkError::IncompatibleOperationInput { actual, .. }) if actual.contains("str")
        ));
    }

    #[test]
    fn field_assignments_update_the_tracked_object_field_type() {
        let program = crate::parse(
            r#"
            state = { value: "text" }
            state.value = 1
            await tools.accept_str(state.value)?
            "#,
        )
        .expect("parse state mutation");

        assert!(matches!(
            LinkedModule::link(program, full_host_environment()),
            Err(LinkError::IncompatibleOperationInput { actual, .. }) if actual == "float"
        ));
    }

    #[test]
    fn missing_known_object_fields_are_errors_but_open_shapes_stay_gradual() {
        let known = crate::parse("value = { present: 1 }\nfinish value.missing")
            .expect("parse known object field access");
        assert!(matches!(
            LinkedModule::link(known, full_host_environment()),
            Err(LinkError::UnknownObjectField { field, .. }) if field == "missing"
        ));

        let open = crate::parse(
            r#"
            process inspect(map: dict, unknown: any) {
              finish [map.missing, unknown.missing]
            }
            "#,
        )
        .expect("parse open shape field access");
        LinkedModule::link(open, full_host_environment())
            .expect("dict and any field access should stay gradual");
    }

    #[test]
    fn union_field_assignments_update_matching_members_and_reject_unknown_fields() {
        let matching = crate::parse(
            r#"
            process mutate(flag: bool) {
              value = { a: 0 }
              if flag { value = { a: 0 } } else { value = { b: 0 } }
              value.a = 1
            }
            "#,
        )
        .expect("parse union field assignment");
        LinkedModule::link(matching, full_host_environment())
            .expect("a field present on one union member should remain assignable");

        let missing = crate::parse(
            r#"
            process mutate(flag: bool) {
              value = { a: 0 }
              if flag { value = { a: 0 } } else { value = { b: 0 } }
              value.c = 1
            }
            "#,
        )
        .expect("parse missing union field assignment");
        assert!(matches!(
            LinkedModule::link(missing, full_host_environment()),
            Err(LinkError::UnknownObjectField { field, .. }) if field == "c"
        ));
    }

    #[test]
    fn binary_operators_reject_known_category_errors_but_admit_unknown_maps() {
        let known = crate::parse("finish {} + 1").expect("parse bad binary operands");
        assert!(matches!(
            LinkedModule::link(known, full_host_environment()),
            Err(LinkError::IncompatibleBinaryOperands { .. })
        ));

        let gradual = crate::parse(
            r#"
            process combine(map: dict, unknown: any) {
              left = map + 1
              finish left + unknown
            }
            "#,
        )
        .expect("parse gradual binary operands");
        LinkedModule::link(gradual, full_host_environment())
            .expect("dict and any operands should stay gradual");
    }

    #[test]
    fn equality_accepts_a_compatible_union_member_but_rejects_known_category_mismatches() {
        let union = crate::parse(
            r#"
            process compare(flag: bool, number: int) {
              value = "initial"
              if flag { value = number } else { value = "text" }
              equal = value == "text"
              not_equal = "text" != value
              finish [equal, not_equal]
            }
            "#,
        )
        .expect("parse union equality");
        LinkedModule::link(union, full_host_environment())
            .expect("equality should accept a category-compatible union member");

        let incompatible = crate::parse("finish {} == 1").expect("parse incompatible equality");
        assert!(matches!(
            LinkedModule::link(incompatible, full_host_environment()),
            Err(LinkError::IncompatibleBinaryOperands { .. })
        ));
    }

    #[test]
    fn loop_carried_mutation_is_widened_after_one_forward_pass() {
        let program = crate::parse(
            r#"
            state = { value: "before" }
            while true {
              state.value = 1
            }
            await tools.accept_float(state.value)?
            "#,
        )
        .expect("parse loop-carried mutation");

        assert!(matches!(
            LinkedModule::link(program, full_host_environment()),
            Err(LinkError::IncompatibleOperationInput { actual, .. }) if actual.contains("str") && actual.contains("float")
        ));
    }

    #[tokio::test]
    async fn module_artifact_store_bytes_reject_corruption() {
        use crate::LashlangArtifactStore;

        let linked = LinkedModule::link(
            crate::parse("process scan() { finish 1 }").expect("parse module"),
            full_host_environment(),
        )
        .expect("link module");
        let store = crate::InMemoryLashlangArtifactStore::new();

        store
            .put_module_artifact(&linked.artifact)
            .await
            .expect("put artifact");
        assert_eq!(
            store
                .get_module_artifact(&linked.module_ref)
                .await
                .expect("get artifact")
                .expect("artifact exists")
                .module_ref,
            linked.module_ref
        );

        assert!(ModuleArtifact::from_store_bytes(b"not json").is_err());
    }
}
