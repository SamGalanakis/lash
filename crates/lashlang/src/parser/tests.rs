#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Declaration, Expr};

    fn block(program: &Program) -> &[Expr] {
        let Expr::Block(expressions) = &program.main else {
            panic!("program root should be a block");
        };
        expressions
    }

    #[test]
    fn top_level_source_becomes_root_block() {
        let program = parse("x = 1\nsubmit x").expect("program should parse");
        let expressions = block(&program);
        assert_eq!(expressions.len(), 2);
        assert!(matches!(expressions[0], Expr::Assign { .. }));
        assert!(matches!(expressions[1], Expr::Submit(Some(_))));
    }

    #[test]
    fn deeply_nested_input_errors_instead_of_overflowing() {
        // Adversarial model-emitted source: thousands of nested parens. Must
        // return a bounded error, not recurse until the native stack aborts.
        let source = format!("x = {}{}", "(".repeat(5000), ")".repeat(5000));
        let err = parse(&source).expect_err("over-deep nesting should be rejected");
        assert!(
            matches!(err, ParseError::NestingTooDeep { .. }),
            "expected NestingTooDeep, got {err:?}"
        );
    }

    #[test]
    fn nesting_within_limit_still_parses() {
        // Well below MAX_NESTING_DEPTH — a realistic program must keep working.
        let source = format!("x = {}1{}", "(".repeat(32), ")".repeat(32));
        parse(&source).expect("nesting within the limit should parse");
    }

    #[test]
    fn deeply_nested_blocks_error_instead_of_overflowing() {
        // Blocks recurse via parse_if -> parse_block, bypassing the expression
        // guard; they must hit the same bounded error, not overflow the stack.
        let source = format!(
            "{}finish null{}",
            "if true {\n".repeat(5000),
            "\n}".repeat(5000)
        );
        let err = parse(&source).expect_err("over-deep block nesting should be rejected");
        assert!(
            matches!(err, ParseError::NestingTooDeep { .. }),
            "expected NestingTooDeep, got {err:?}"
        );
    }

    #[test]
    fn ternary_desugars_to_if_expression() {
        let program = parse("answer = ok ? 1 : 2").expect("program should parse");
        let Expr::Assign { expr, .. } = &block(&program)[0] else {
            panic!("expected assignment");
        };
        assert!(matches!(expr.as_ref(), Expr::If { .. }));
    }

    #[test]
    fn await_record_of_process_starts_parses_directly() {
        let program = parse(
            "process one() { finish null }\nprocess two() { finish null }\nresult = await { a: start one(), b: start two() }",
        )
            .expect("program should parse");
        let Expr::Assign { expr, .. } = &block(&program)[0] else {
            panic!("expected assignment");
        };
        let Expr::Await(inner) = expr.as_ref() else {
            panic!("expected await expression");
        };
        let Expr::Record(entries) = inner.as_ref() else {
            panic!("await should target a record");
        };
        assert_eq!(entries.len(), 2);
        assert!(
            entries
                .iter()
                .all(|(_, expr)| matches!(expr, Expr::StartProcess(_)))
        );
    }

    #[test]
    fn await_list_of_process_starts_parses_directly() {
        let program = parse(
            "process one() { finish null }\nprocess two() { finish null }\nsubmit await [start one(), start two()]",
        )
            .expect("program should parse");
        let Expr::Submit(Some(expr)) = &block(&program)[0] else {
            panic!("expected submit");
        };
        let Expr::Await(inner) = expr.as_ref() else {
            panic!("expected await expression");
        };
        let Expr::List(items) = inner.as_ref() else {
            panic!("await should target a list");
        };
        assert_eq!(items.len(), 2);
        assert!(
            items
                .iter()
                .all(|expr| matches!(expr, Expr::StartProcess(_)))
        );
    }

    #[test]
    fn module_declarations_parse_process_values_constructors_and_receiver_calls() {
        let program = parse(
            r#"
            type EmailInput = { source: "gmail" | "manual", message_id: string? }
            process triage(gmail: GMAIL, input: EmailInput) -> null {
              msg = await gmail.get_message(input.message_id)?
              finish msg
            }
            process digest(tick: timer.Tick) -> bool {
              finish true
            }
            source = timer.Schedule({ expr: "0 8 * * *", tz: "UTC" })
            handle = await triggers.register({
              source: source,
              target: digest,
              inputs: { tick: trigger.event },
              name: "daily_digest"
            })?
            submit handle
            "#,
        )
        .expect("module should parse");
        assert_eq!(program.declarations.len(), 3);
        assert!(matches!(program.declarations[0], Declaration::Type(_)));
        assert!(matches!(program.declarations[1], Declaration::Process(_)));
        assert!(matches!(program.declarations[2], Declaration::Process(_)));
        let expressions = block(&program);
        assert_eq!(expressions.len(), 3);
        assert!(matches!(expressions[0], Expr::Assign { .. }));
        assert!(matches!(expressions[1], Expr::Assign { .. }));
        assert!(matches!(expressions[2], Expr::Submit(Some(_))));
    }

    #[test]
    fn label_annotations_parse_on_processes_and_visual_process_statements() {
        let program = parse(
            r#"
            @label(title: "Scan", description: "Reads one file")
            process scan(tool: Tools) {
              @label(title: "Read file", description: "Host operation")
              text = await tool.read_file({ path: "." })?
              @label(title: "Branch")
              if text {
                @label(title: "Wake agent")
                wake text
              } else {
                @label(title: "Finish empty")
                finish null
              }
            }
            "#,
        )
        .expect("annotations should parse");
        let Declaration::Process(process) = &program.declarations[0] else {
            panic!("expected process");
        };
        assert_eq!(
            process.label.as_ref().map(|label| label.title.as_str()),
            Some("Scan")
        );
        let Expr::Block(expressions) = &process.body else {
            panic!("expected process block");
        };
        let Expr::LabelAnnotated { expr, .. } = &expressions[0] else {
            panic!("expected annotated assignment");
        };
        assert!(matches!(expr.as_ref(), Expr::Assign { .. }));
        let Expr::LabelAnnotated { expr, .. } = &expressions[1] else {
            panic!("expected annotated branch");
        };
        assert!(matches!(expr.as_ref(), Expr::If { .. }));
    }

    #[test]
    fn label_annotations_parse_on_top_level_statements() {
        let program = parse(
            r#"
            @label(title: "Setup")
            value = 1
            @label(title: "Submit", description: "Return the value")
            submit value
            "#,
        )
        .expect("top-level labels should parse");
        let Expr::Block(expressions) = &program.main else {
            panic!("expected top-level block");
        };
        assert_eq!(expressions.len(), 2);
        let Expr::LabelAnnotated { label, expr } = &expressions[0] else {
            panic!("expected annotated setup");
        };
        assert_eq!(label.title.as_str(), "Setup");
        assert!(matches!(expr.as_ref(), Expr::Assign { .. }));
        let Expr::LabelAnnotated { label, expr } = &expressions[1] else {
            panic!("expected annotated submit");
        };
        assert_eq!(label.title.as_str(), "Submit");
        assert_eq!(
            label
                .description
                .as_ref()
                .map(|description| description.as_str()),
            Some("Return the value")
        );
        assert!(matches!(expr.as_ref(), Expr::Submit { .. }));
    }

    #[test]
    fn label_annotations_reject_invalid_syntax_and_targets() {
        let cases = [
            r#"process p() { @label(description: "missing") finish null }"#,
            r#"process p() { @label(title: value) finish null }"#,
            r#"process p() { @label(title: "x", color: "red") finish null }"#,
            r#"process p() { @label(title: "x", title: "y") finish null }"#,
            r#"process p() { @label(title: "x") @label(title: "y") finish null }"#,
            r#"@label(title: "Shape") type Shape { value: str }"#,
        ];
        for source in cases {
            let err = parse(source).expect_err(source);
            assert!(
                matches!(
                    err,
                    ParseError::InvalidLabelAnnotation { .. }
                        | ParseError::InvalidLabelTarget { .. }
                        | ParseError::Expected { .. }
                ),
                "unexpected error for {source}: {err:?}"
            );
        }
    }

    #[test]
    fn label_annotation_target_error_rejects_non_process_declarations() {
        let err = parse(r#"@label(title: "Shape") type Shape { value: str }"#)
            .expect_err("type declarations are not label targets");
        let message = err.to_string();
        assert!(message.contains("statements or process declarations"));
        assert!(message.contains("other declarations"));
        assert!(!message.contains("process-map"));
    }

    #[test]
    fn label_annotation_text_inside_strings_is_plain_text() {
        let program = parse(
            r####"
            submit r'''@label(title: "Plain text")
@label(title: "Still text") finish null'''
            "####,
        )
        .expect("label-like text inside strings should parse as text");
        let Expr::Submit(Some(expr)) = &block(&program)[0] else {
            panic!("expected submit");
        };
        let Expr::String(value) = expr.as_ref() else {
            panic!("expected string");
        };
        assert_eq!(
            value.as_str(),
            "@label(title: \"Plain text\")\n@label(title: \"Still text\") finish null"
        );
    }

    #[test]
    fn declarative_trigger_syntax_is_rejected() {
        let err = parse(
            r#"
            process triage(event: any) { finish event }
            trigger personal_mail on GMAIL.personal.new_message as event
              -> triage(event: event)
            "#,
        )
        .expect_err("declarative trigger syntax should be rejected");
        assert!(matches!(err, ParseError::DeclarativeTriggerRemoved { .. }));
    }

    #[test]
    fn static_graph_includes_process_activation_resources_and_branches() {
        let program = parse(
            r#"
            type EmailInput = { source: "gmail" | "manual", message_id: string? }
            process triage(gmail: Gmail, input: EmailInput) -> null {
              if input.source == "gmail" {
                msg = await gmail.get_message(input.message_id)?
              } else {
                msg = null
              }
              finish msg
            }
            source = timer.Schedule({ expr: "0 8 * * *", tz: "UTC" })
            handle = await triggers.register({
              source: source,
              target: triage,
              inputs: { input: trigger.event, gmail: gmail.work },
              name: "daily_digest"
            })?
            submit handle
            "#,
        )
        .expect("module should parse");

        let graph = crate::static_graph_json(&program, "v1");
        let nodes = graph["nodes"].as_array().expect("graph nodes");
        let edges = graph["edges"].as_array().expect("graph edges");
        assert!(nodes.iter().any(|node| node["kind"] == "process"));
        assert!(
            nodes
                .iter()
                .any(|node| node["kind"] == "resource_operation")
        );
        assert!(nodes.iter().any(|node| node["kind"] == "branch"));
        assert!(nodes.iter().any(|node| node["kind"] == "terminal"));
        assert!(edges.iter().any(|edge| edge["label"] == "calls"));
        assert!(
            nodes
                .iter()
                .all(|node| node["span"]["end"].as_u64() > node["span"]["start"].as_u64())
        );
        assert!(
            edges
                .iter()
                .all(|edge| edge["span"]["end"].as_u64() > edge["span"]["start"].as_u64())
        );
    }

    #[test]
    fn process_body_parses_passed_authority_calls() {
        parse(
            r#"
            process ok(mail: Gmail) {
              await mail.get_message({ id: "id" })?
            }
            "#,
        )
        .expect("passed authority call should parse inside process bodies");
    }

    #[test]
    fn removed_parallel_keyword_is_rejected() {
        let err = parse("parallel { x = 1 }").expect_err("keyword should be rejected");
        assert!(matches!(err, ParseError::Unexpected { .. }));
    }
}
