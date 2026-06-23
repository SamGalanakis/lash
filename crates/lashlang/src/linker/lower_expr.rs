impl<'module> Linker<'module> {
    fn lower_expr(
        &self,
        expr: &Expr,
        scope: &mut Scope,
    ) -> Result<(Expr, Option<Binding>), LinkError> {
        self.reject_trigger_event_special_form(expr, scope.span)?;
        if matches!(expr, Expr::Variable(_) | Expr::Field { .. })
            && let Some(resource) = self.resolve_module_expr(expr, scope)
        {
            return Ok((
                Expr::ResourceRef(resource.clone()),
                Some(Binding::Resource {
                    resource_type: resource.resource_type.to_string(),
                }),
            ));
        }
        Ok(match expr {
            Expr::Block(expressions) => {
                let mut lowered = Vec::with_capacity(expressions.len());
                let mut last = None;
                for expression in expressions {
                    let (expr, binding) = self.lower_expr(expression, scope)?;
                    lowered.push(expr);
                    last = binding;
                }
                (Expr::Block(lowered), last)
            }
            Expr::LabelAnnotated { label, expr } => {
                self.ensure_feature(
                    self.surface.language_features.label_annotations,
                    "label annotations",
                    scope.span,
                )?;
                let (expr, binding) = self.lower_expr(expr, scope)?;
                (
                    Expr::LabelAnnotated {
                        label: label.clone(),
                        expr: Box::new(expr),
                    },
                    binding,
                )
            }
            Expr::Variable(name) => {
                if let Some(binding) = scope.get(name) {
                    (Expr::Variable(name.clone()), Some(binding))
                } else if let Some(process_ty) = self.process_types.get(name.as_str()) {
                    (
                        Expr::ProcessRef {
                            process: name.clone(),
                        },
                        Some(Binding::Value(process_ty.clone())),
                    )
                } else if scope.allow_unknown_globals {
                    // Top-level unknown globals are permitted; they surface as
                    // runtime errors rather than link errors.
                    (
                        Expr::Variable(name.clone()),
                        Some(Binding::Value(TypeExpr::Any)),
                    )
                } else {
                    return Err(LinkError::UnknownName {
                        name: name.to_string(),
                        span: scope.span,
                    });
                }
            }
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Break
            | Expr::Continue
            | Expr::TypeLiteral(_) => (expr.clone(), Some(Binding::Value(literal_type(expr)))),
            Expr::Tuple(items) => {
                let mut lowered = Vec::with_capacity(items.len());
                let mut item_types = Vec::with_capacity(items.len());
                for item in items {
                    let (item, binding) = self.lower_expr(item, scope)?;
                    lowered.push(item);
                    item_types.push(binding_type(binding.as_ref()));
                }
                (
                    Expr::Tuple(lowered),
                    Some(Binding::Value(TypeExpr::List(Box::new(union_type(
                        item_types,
                    ))))),
                )
            }
            Expr::List(items) => {
                let mut lowered = Vec::with_capacity(items.len());
                let mut item_types = Vec::with_capacity(items.len());
                for item in items {
                    let (item, binding) = self.lower_expr(item, scope)?;
                    lowered.push(item);
                    item_types.push(binding_type(binding.as_ref()));
                }
                (
                    Expr::List(lowered),
                    Some(Binding::Value(TypeExpr::List(Box::new(union_type(
                        item_types,
                    ))))),
                )
            }
            Expr::ListComprehension { element, clauses } => {
                let mut lowered_clauses = Vec::with_capacity(clauses.len());
                let mut previous_bindings = Vec::new();
                for clause in clauses {
                    match clause {
                        ListComprehensionClause::For { binding, iterable } => {
                            let (iterable, iterable_binding) = self.lower_expr(iterable, scope)?;
                            let item_ty =
                                self.index_type(&binding_type(iterable_binding.as_ref()), scope.span)?;
                            previous_bindings.push((
                                binding.to_string(),
                                scope.bind(binding.as_str(), self.binding_for_type(&item_ty)),
                            ));
                            lowered_clauses.push(ListComprehensionClause::For {
                                binding: binding.clone(),
                                iterable,
                            });
                        }
                        ListComprehensionClause::If { condition } => {
                            let condition = self.lower_expr(condition, scope)?.0;
                            lowered_clauses.push(ListComprehensionClause::If { condition });
                        }
                    }
                }
                let (element, binding) = self.lower_expr(element, scope)?;
                for (name, previous) in previous_bindings.into_iter().rev() {
                    scope.restore(name.as_str(), previous);
                }
                (
                    Expr::ListComprehension {
                        element: Box::new(element),
                        clauses: lowered_clauses,
                    },
                    Some(Binding::Value(TypeExpr::List(Box::new(binding_type(
                        binding.as_ref(),
                    ))))),
                )
            }
            Expr::Record(entries) => {
                let mut lowered = Vec::with_capacity(entries.len());
                let mut fields = Vec::with_capacity(entries.len());
                for (name, value) in entries {
                    let (value, binding) = self.lower_expr(value, scope)?;
                    fields.push(TypeField {
                        name: name.clone(),
                        ty: binding_type(binding.as_ref()),
                        optional: false,
                    });
                    lowered.push((name.clone(), value));
                }
                (
                    Expr::Record(lowered),
                    Some(Binding::Value(TypeExpr::Object(fields))),
                )
            }
            Expr::Assign { target, expr } => {
                for step in &target.steps {
                    if let AssignPathStep::Index(index) = step {
                        self.lower_expr(index, scope)?;
                    }
                }
                let (lowered, binding) = self.lower_expr(expr, scope)?;
                if target.steps.is_empty() {
                    scope.bind(
                        target.root.as_str(),
                        binding.clone().unwrap_or(any_binding()),
                    );
                } else if scope.get(&target.root).is_none() && !scope.allow_unknown_globals {
                    return Err(LinkError::UnknownName {
                        name: target.root.to_string(),
                        span: scope.span,
                    });
                }
                (
                    Expr::Assign {
                        target: target.clone(),
                        expr: Box::new(lowered),
                    },
                    binding,
                )
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                let condition = self.lower_expr(condition, scope)?.0;
                let mut then_scope = scope.clone();
                let (then_block, then_binding) = self.lower_expr(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                let (else_block, else_binding) = self.lower_expr(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                (
                    Expr::If {
                        condition: Box::new(condition),
                        then_block: Box::new(then_block),
                        else_block: Box::new(else_block),
                    },
                    Some(Binding::Value(union_type(vec![
                        binding_type(then_binding.as_ref()),
                        binding_type(else_binding.as_ref()),
                    ]))),
                )
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                let iterable = self.lower_expr(iterable, scope)?.0;
                let previous = scope.bind(binding.as_str(), Binding::Value(TypeExpr::Any));
                let body = self.lower_expr(body, scope)?.0;
                scope.restore(binding.as_str(), previous);
                (
                    Expr::For {
                        binding: binding.clone(),
                        iterable: Box::new(iterable),
                        body: Box::new(body),
                    },
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::While { condition, body } => {
                let condition = self.lower_expr(condition, scope)?.0;
                let body = self.lower_expr(body, scope)?.0;
                (
                    Expr::While {
                        condition: Box::new(condition),
                        body: Box::new(body),
                    },
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::StartProcess(start) => {
                self.ensure_feature(self.surface.abilities.processes, "processes", scope.span)?;
                let Some(process) = self.program.process(start.process.as_str()) else {
                    return Err(LinkError::UnknownProcess {
                        name: start.process.to_string(),
                        span: scope.span,
                    });
                };
                let mut seen = BTreeSet::new();
                let mut lowered_args = Vec::with_capacity(start.args.len());
                for (arg, value) in &start.args {
                    if !seen.insert(arg.to_string()) {
                        return Err(LinkError::DuplicateProcessArgument {
                            arg: arg.to_string(),
                            span: scope.span,
                        });
                    }
                    let Some(param) = process.params.iter().find(|param| param.name == *arg) else {
                        return Err(LinkError::UnexpectedProcessArgument {
                            process: process.name.to_string(),
                            arg: arg.to_string(),
                            span: scope.span,
                        });
                    };
                    let (lowered, binding) = self.lower_expr(value, scope)?;
                    self.validate_process_arg_binding(
                        process.name.as_str(),
                        arg.as_str(),
                        &param.ty,
                        binding.as_ref(),
                        scope.span,
                    )?;
                    lowered_args.push((arg.clone(), lowered));
                }
                for param in &process.params {
                    if !seen.contains(param.name.as_str()) {
                        return Err(LinkError::MissingProcessArgument {
                            process: process.name.to_string(),
                            arg: param.name.to_string(),
                            span: scope.span,
                        });
                    }
                }
                (
                    Expr::StartProcess(crate::ast::ProcessStartExpr {
                        process: start.process.clone(),
                        args: lowered_args,
                    }),
                    Some(Binding::Value(TypeExpr::Any)),
                )
            }
            Expr::ProcessRef { process } => {
                let Some(process_ty) = self.process_types.get(process.as_str()) else {
                    return Err(LinkError::UnknownProcess {
                        name: process.to_string(),
                        span: scope.span,
                    });
                };
                (
                    Expr::ProcessRef {
                        process: process.clone(),
                    },
                    Some(Binding::Value(process_ty.clone())),
                )
            }
            Expr::HostDescriptorConstructor { type_name, input } => (
                Expr::HostDescriptorConstructor {
                    type_name: type_name.clone(),
                    input: Box::new(self.lower_expr(input, scope)?.0),
                },
                Some(Binding::Value(TypeExpr::Ref(type_name.clone()))),
            ),
            Expr::ResourceRef(resource) => {
                let resource = self.validate_resource_ref(resource, scope.span)?;
                (
                    Expr::ResourceRef(resource.clone()),
                    Some(Binding::Resource {
                        resource_type: resource.resource_type.to_string(),
                    }),
                )
            }
            Expr::ReceiverCall {
                receiver,
                operation,
                args,
            } => {
                if let Some(mut path) = module_path_for_expr(receiver) {
                    path.push(operation.clone());
                    if let Some(constructor) =
                        self.surface.resources.resolve_value_constructor(&path)
                    {
                        if args.len() != 1 {
                            return Err(LinkError::IncompatibleConstructorInput {
                                path: module_path_key(&path),
                                expected: format_type_expr(&constructor.input_ty),
                                actual: format!("{} arguments", args.len()),
                                span: scope.span,
                            });
                        }
                        let (input, input_binding) = self.lower_expr(&args[0], scope)?;
                        let actual_ty = binding_type(input_binding.as_ref());
                        if !self.is_type_assignable(&actual_ty, &constructor.input_ty) {
                            return Err(LinkError::IncompatibleConstructorInput {
                                path: module_path_key(&path),
                                expected: format_type_expr(
                                    &self.resolve_type_aliases(&constructor.input_ty),
                                ),
                                actual: format_type_expr(&self.resolve_type_aliases(&actual_ty)),
                                span: scope.span,
                            });
                        }
                        return Ok((
                            Expr::HostDescriptorConstructor {
                                type_name: constructor.type_name.clone().into(),
                                input: Box::new(input),
                            },
                            Some(Binding::Value(constructor.output_ty.clone())),
                        ));
                    }
                }
                let resolved_receiver = self.resolve_module_expr(receiver, scope);
                let (lowered_receiver, resource_type, receiver_alias) =
                    if let Some(resource) = resolved_receiver.as_ref() {
                        (
                            Expr::ResourceRef(resource.clone()),
                            Some(resource.resource_type.to_string()),
                            Some(resource.alias.to_string()),
                        )
                    } else {
                        let (lowered_receiver, binding) = self.lower_expr(receiver, scope)?;
                        let resource_type = match binding {
                            Some(Binding::Resource { resource_type }) => Some(resource_type),
                            _ => None,
                        };
                        (lowered_receiver, resource_type, None)
                    };
                let Some(resource_type) = resource_type else {
                    if let Some(path) = module_path_for_expr(receiver) {
                        let suggestions = self
                            .surface
                            .resources
                            .operation_suggestions_for_prefix(&path, operation.as_str());
                        if !suggestions.is_empty() {
                            return Err(LinkError::AmbiguousModuleOperation {
                                module_path: module_path_key(&path),
                                operation: operation.to_string(),
                                suggestions,
                                span: scope.span,
                            });
                        }
                    }
                    return Err(LinkError::UnresolvedReceiver {
                        operation: operation.to_string(),
                        span: scope.span,
                    });
                };
                if let Some(alias) = receiver_alias.as_deref()
                    && self
                        .surface
                        .resources
                        .resolve_module_operation(&resource_type, alias, operation.as_str())
                        .is_none()
                {
                    return Err(LinkError::UnknownResourceOperation {
                        resource_type: resource_type.clone(),
                        operation: operation.to_string(),
                        span: scope.span,
                    });
                }
                let Some(operation_binding) = self
                    .surface
                    .resources
                    .resolve_operation(&resource_type, operation)
                    .cloned()
                else {
                    return Err(LinkError::UnknownResourceOperation {
                        resource_type: resource_type.clone(),
                        operation: operation.to_string(),
                        span: scope.span,
                    });
                };
                let trigger_operation = if crate::is_trigger_resource_type(&resource_type) {
                    crate::TriggerHostOperation::from_receiver_method(operation.as_str())
                } else {
                    None
                };
                if let Some(trigger_operation) = trigger_operation {
                    self.ensure_feature(self.surface.abilities.triggers, "triggers", scope.span)?;
                    let (lowered_args, output_ty) =
                        self.lower_trigger_operation_args(trigger_operation, args, scope)?;
                    return Ok((
                        Expr::ReceiverCall {
                            receiver: Box::new(lowered_receiver),
                            operation: operation.clone(),
                            args: lowered_args,
                        },
                        Some(Binding::Value(output_ty)),
                    ));
                }
                let mut lowered_args = Vec::with_capacity(args.len());
                let mut arg_types = Vec::with_capacity(args.len());
                for arg in args {
                    let (arg, binding) = self.lower_expr(arg, scope)?;
                    lowered_args.push(arg);
                    arg_types.push(binding_type(binding.as_ref()));
                }
                let actual_input = call_input_type(arg_types);
                if !self.is_type_assignable(&actual_input, &operation_binding.input_ty) {
                    return Err(LinkError::IncompatibleOperationInput {
                        operation: operation.to_string(),
                        expected: format_type_expr(
                            &self.resolve_type_aliases(&operation_binding.input_ty),
                        ),
                        actual: format_type_expr(&self.resolve_type_aliases(&actual_input)),
                        span: scope.span,
                    });
                }
                (
                    Expr::ReceiverCall {
                        receiver: Box::new(lowered_receiver),
                        operation: operation.clone(),
                        args: lowered_args,
                    },
                    Some(Binding::Value(operation_binding.output_ty.clone())),
                )
            }
            Expr::Await(inner) => {
                let (inner, binding) = self.lower_expr(inner, scope)?;
                (Expr::Await(Box::new(inner)), binding)
            }
            Expr::SleepFor(inner) => {
                self.ensure_feature(self.surface.abilities.sleep, "sleep", scope.span)?;
                (
                    Expr::SleepFor(Box::new(self.lower_expr(inner, scope)?.0)),
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::SleepUntil(inner) => {
                self.ensure_feature(self.surface.abilities.sleep, "sleep", scope.span)?;
                (
                    Expr::SleepUntil(Box::new(self.lower_expr(inner, scope)?.0)),
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::WaitSignal { name } => {
                self.ensure_feature(
                    self.surface.abilities.process_signals,
                    "process signals",
                    scope.span,
                )?;
                if !scope.process_body {
                    return Err(LinkError::ProcessLifecycleOutsideProcess {
                        keyword: "wait_signal",
                        span: scope.span,
                    });
                }
                (
                    Expr::WaitSignal { name: name.clone() },
                    Some(Binding::Value(TypeExpr::Any)),
                )
            }
            Expr::SignalRun { run, name, payload } => {
                self.ensure_feature(
                    self.surface.abilities.process_signals,
                    "process signals",
                    scope.span,
                )?;
                // `signal_run` (sending) is a control-plane op like `await` /
                // `cancel`, valid from the foreground turn as well as inside a
                // process body. Only `wait_signal` (receiving) is process-only.
                (
                    Expr::SignalRun {
                        run: Box::new(self.lower_expr(run, scope)?.0),
                        name: name.clone(),
                        payload: Box::new(self.lower_expr(payload, scope)?.0),
                    },
                    Some(Binding::Value(TypeExpr::Null)),
                )
            }
            Expr::ResultUnwrap(inner) => {
                let (inner, binding) = self.lower_expr(inner, scope)?;
                (Expr::ResultUnwrap(Box::new(inner)), binding)
            }
            Expr::Cancel(inner) => (
                Expr::Cancel(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Any)),
            ),
            Expr::Print(inner) => (
                Expr::Print(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Submit(inner) => (
                Expr::Submit(
                    inner
                        .as_deref()
                        .map(|inner| {
                            self.lower_expr(inner, scope)
                                .map(|(expr, _)| Box::new(expr))
                        })
                        .transpose()?,
                ),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Yield(inner) => (
                Expr::Yield(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Wake(inner) => (
                Expr::Wake(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::Finish(inner) => {
                let mut finish_ty = TypeExpr::Null;
                let inner = inner
                    .as_deref()
                    .map(|inner| {
                        let (expr, binding) = self.lower_expr(inner, scope)?;
                        finish_ty = binding_type(binding.as_ref());
                        Ok(Box::new(expr))
                    })
                    .transpose()?;
                (Expr::Finish(inner), Some(Binding::Value(finish_ty)))
            }
            Expr::Fail(inner) => (
                Expr::Fail(Box::new(self.lower_expr(inner, scope)?.0)),
                Some(Binding::Value(TypeExpr::Null)),
            ),
            Expr::BuiltinCall { name, args } => {
                if !crate::builtins::is_builtin(name.as_str()) {
                    if let Some(suggestion) = self
                        .surface
                        .resources
                        .operation_suggestions_for_host(name.as_str())
                        .into_iter()
                        .next()
                    {
                        return Err(LinkError::BareToolCall {
                            name: name.to_string(),
                            suggestion,
                            span: scope.span,
                        });
                    }
                    return Err(LinkError::UnknownBuiltin {
                        name: name.to_string(),
                        span: scope.span,
                    });
                }
                (
                    Expr::BuiltinCall {
                        name: name.clone(),
                        args: args
                            .iter()
                            .map(|arg| self.lower_expr(arg, scope).map(|(expr, _)| expr))
                            .collect::<Result<Vec<_>, _>>()?,
                    },
                    Some(Binding::Value(builtin_return_type(name.as_str()))),
                )
            }
            Expr::Field { target, field } => {
                let (target, binding) = self.lower_expr(target, scope)?;
                let ty =
                    self.field_type(&binding_type(binding.as_ref()), field.as_str(), scope.span)?;
                (
                    Expr::Field {
                        target: Box::new(target),
                        field: field.clone(),
                    },
                    Some(Binding::Value(ty)),
                )
            }
            Expr::Index { target, index } => {
                let (target, target_binding) = self.lower_expr(target, scope)?;
                let index = self.lower_expr(index, scope)?.0;
                (
                    Expr::Index {
                        target: Box::new(target),
                        index: Box::new(index),
                    },
                    Some(Binding::Value(self.index_type(
                        &binding_type(target_binding.as_ref()),
                        scope.span,
                    )?)),
                )
            }
            Expr::Unary { op, expr } => (
                Expr::Unary {
                    op: *op,
                    expr: Box::new(self.lower_expr(expr, scope)?.0),
                },
                Some(Binding::Value(match op {
                    crate::ast::UnaryOp::Not => TypeExpr::Bool,
                    crate::ast::UnaryOp::Negate => TypeExpr::Float,
                })),
            ),
            Expr::Binary { left, op, right } => (
                Expr::Binary {
                    left: Box::new(self.lower_expr(left, scope)?.0),
                    op: *op,
                    right: Box::new(self.lower_expr(right, scope)?.0),
                },
                Some(Binding::Value(binary_return_type(*op))),
            ),
        })
    }

    fn resolve_module_expr(&self, expr: &Expr, scope: &Scope) -> Option<ResourceRefExpr> {
        let path = module_path_for_expr(expr)?;
        if path
            .first()
            .and_then(|root| scope.get_str(root.as_str()))
            .is_some()
        {
            return None;
        }
        self.surface.resources.resolve_module_path(&path)
    }

    fn reject_trigger_event_special_form(
        &self,
        expr: &Expr,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        if is_trigger_event_projection_expr(expr) {
            return Err(LinkError::TriggerEventProjection { span });
        }
        if is_trigger_event_expr(expr) {
            return Err(LinkError::TriggerEventOutsideInputs { span });
        }
        Ok(())
    }

}
