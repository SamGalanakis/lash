impl<'module> Linker<'module> {
    fn validate_process_arg_binding(
        &self,
        process: &str,
        arg: &str,
        expected_ty: &TypeExpr,
        actual: Option<&Binding>,
        span: Option<Span>,
    ) -> Result<(), LinkError> {
        let Some(expected_resource) = self.resource_type_for_type(expected_ty) else {
            return Ok(());
        };
        match actual {
            Some(Binding::Resource { resource_type }) if *resource_type == expected_resource => {
                Ok(())
            }
            Some(Binding::Resource { resource_type }) => {
                Err(LinkError::IncompatibleProcessArgument {
                    process: process.to_string(),
                    arg: arg.to_string(),
                    expected: expected_resource,
                    actual: resource_type.clone(),
                    span,
                })
            }
            _ => Err(LinkError::IncompatibleProcessArgument {
                process: process.to_string(),
                arg: arg.to_string(),
                expected: expected_resource,
                actual: "value".to_string(),
                span,
            }),
        }
    }

    fn validate_trigger_operation_args(
        &self,
        operation: crate::TriggerHostOperation,
        args: &[Expr],
        scope: &Scope,
    ) -> Result<TypeExpr, LinkError> {
        match operation {
            crate::TriggerHostOperation::Register => {
                let call = crate::register_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerRegistration { span: scope.span })?;
                let source_ty = self.infer_expr_type(call.source, &mut scope.clone())?;
                let event_ty = self
                    .surface
                    .resources
                    .trigger_source_event(&source_ty)
                    .ok_or_else(|| LinkError::UnknownTriggerEventType {
                        source_ty: format_type_expr(&source_ty),
                        span: scope.span,
                    })?;
                let target_ty = self.infer_expr_type(call.target, &mut scope.clone())?;
                let params = self.trigger_target_params(call.target, &target_ty, scope.span)?;
                let mut validation_scope = scope.clone();
                self.lower_trigger_input_record(
                    trigger_target_process_label(call.target).as_str(),
                    &params,
                    &event_ty,
                    call.inputs,
                    &mut validation_scope,
                )?;
                Ok(TypeExpr::TriggerHandle(Box::new(event_ty)))
            }
            crate::TriggerHostOperation::List => {
                let call = crate::list_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerList { span: scope.span })?;
                for (name, expr) in call.entries {
                    match name.as_str() {
                        "target" => {
                            let target_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !matches!(target_ty, TypeExpr::Process { .. }) {
                                return Err(LinkError::InvalidTriggerTarget {
                                    actual: format_type_expr(&target_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "name" | "source_type" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Str) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.receiver_method().to_string(),
                                    expected: format_type_expr(&TypeExpr::Str),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "enabled" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Bool) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.receiver_method().to_string(),
                                    expected: format_type_expr(&TypeExpr::Bool),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        _ => unreachable!("list_call_args rejects unknown trigger filters"),
                    }
                }
                Ok(operation.output_ty())
            }
            crate::TriggerHostOperation::Cancel => {
                crate::cancel_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerCancel { span: scope.span })?;
                Ok(operation.output_ty())
            }
        }
    }

    fn lower_trigger_operation_args(
        &self,
        operation: crate::TriggerHostOperation,
        args: &[Expr],
        scope: &mut Scope,
    ) -> Result<(Vec<Expr>, TypeExpr), LinkError> {
        match operation {
            crate::TriggerHostOperation::Register => {
                self.lower_trigger_registration_args(args, scope)
            }
            crate::TriggerHostOperation::List => {
                let call = crate::list_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerList { span: scope.span })?;
                let mut entries = Vec::with_capacity(call.entries.len());
                for (name, expr) in call.entries {
                    match name.as_str() {
                        "target" => {
                            let target_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !matches!(target_ty, TypeExpr::Process { .. }) {
                                return Err(LinkError::InvalidTriggerTarget {
                                    actual: format_type_expr(&target_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "name" | "source_type" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Str) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.receiver_method().to_string(),
                                    expected: format_type_expr(&TypeExpr::Str),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        "enabled" => {
                            let filter_ty = self.infer_expr_type(expr, &mut scope.clone())?;
                            if !self.is_type_assignable(&filter_ty, &TypeExpr::Bool) {
                                return Err(LinkError::IncompatibleOperationInput {
                                    operation: operation.receiver_method().to_string(),
                                    expected: format_type_expr(&TypeExpr::Bool),
                                    actual: format_type_expr(&filter_ty),
                                    span: scope.span,
                                });
                            }
                        }
                        _ => unreachable!("list_call_args rejects unknown trigger filters"),
                    }
                    entries.push((name.clone(), self.lower_expr(expr, scope)?.0));
                }
                Ok((vec![Expr::Record(entries)], operation.output_ty()))
            }
            crate::TriggerHostOperation::Cancel => {
                let call = crate::cancel_call_args(args)
                    .map_err(|_| LinkError::InvalidTriggerCancel { span: scope.span })?;
                Ok((
                    vec![Expr::Record(vec![(
                        "handle".into(),
                        self.lower_expr(call.handle, scope)?.0,
                    )])],
                    operation.output_ty(),
                ))
            }
        }
    }

    fn lower_trigger_registration_args(
        &self,
        args: &[Expr],
        scope: &mut Scope,
    ) -> Result<(Vec<Expr>, TypeExpr), LinkError> {
        let call = crate::register_call_args(args)
            .map_err(|_| LinkError::InvalidTriggerRegistration { span: scope.span })?;
        let source_ty = self.infer_expr_type(call.source, &mut scope.clone())?;
        let event_ty = self
            .surface
            .resources
            .trigger_source_event(&source_ty)
            .ok_or_else(|| LinkError::UnknownTriggerEventType {
                source_ty: format_type_expr(&source_ty),
                span: scope.span,
            })?;
        let target_ty = self.infer_expr_type(call.target, &mut scope.clone())?;
        let params = self.trigger_target_params(call.target, &target_ty, scope.span)?;
        let process = trigger_target_process_label(call.target);

        let source = self.lower_expr(call.source, scope)?.0;
        let target = self.lower_expr(call.target, scope)?.0;
        let inputs = self.lower_trigger_input_record(
            process.as_str(),
            &params,
            &event_ty,
            call.inputs,
            scope,
        )?;
        let mut entries = vec![
            ("source".into(), source),
            ("target".into(), target),
            ("inputs".into(), inputs),
        ];
        if let Some(name) = call.name {
            entries.push(("name".into(), self.lower_expr(name, scope)?.0));
        }
        Ok((
            vec![Expr::Record(entries)],
            TypeExpr::TriggerHandle(Box::new(event_ty)),
        ))
    }

    fn lower_trigger_input_record(
        &self,
        process: &str,
        params: &[ProcessParam],
        event_ty: &TypeExpr,
        inputs: &Expr,
        scope: &mut Scope,
    ) -> Result<Expr, LinkError> {
        let Expr::Record(entries) = inputs else {
            return Err(LinkError::InvalidTriggerInputs { span: scope.span });
        };
        let mut seen = BTreeSet::new();
        let mut saw_event = false;
        let mut lowered = Vec::with_capacity(entries.len());
        for (name, value) in entries {
            if !seen.insert(name.to_string()) {
                return Err(LinkError::DuplicateTriggerInput {
                    input: name.to_string(),
                    span: scope.span,
                });
            }
            let Some(param) = params.iter().find(|param| param.name == *name) else {
                return Err(LinkError::UnknownTriggerInput {
                    process: process.to_string(),
                    input: name.to_string(),
                    span: scope.span,
                });
            };
            if is_trigger_event_projection_expr(value) {
                return Err(LinkError::TriggerEventProjection { span: scope.span });
            }
            if is_trigger_event_expr(value) {
                saw_event = true;
                if !self.is_type_assignable(event_ty, &param.ty) {
                    return Err(LinkError::TriggerEventMismatch {
                        event: format_type_expr(&self.resolve_type_aliases(event_ty)),
                        input_name: name.to_string(),
                        input: format_type_expr(&self.resolve_type_aliases(&param.ty)),
                        span: scope.span,
                    });
                }
                lowered.push((name.clone(), crate::trigger_event_placeholder_expr()));
                continue;
            }
            let (lowered_value, binding) = self.lower_expr(value, scope)?;
            self.validate_process_arg_binding(
                process,
                name.as_str(),
                &param.ty,
                binding.as_ref(),
                scope.span,
            )?;
            lowered.push((name.clone(), lowered_value));
        }
        for param in params {
            if !seen.contains(param.name.as_str()) {
                return Err(LinkError::MissingTriggerInput {
                    process: process.to_string(),
                    input: param.name.to_string(),
                    span: scope.span,
                });
            }
        }
        if !saw_event {
            return Err(LinkError::MissingTriggerEventInput { span: scope.span });
        }
        Ok(Expr::Record(lowered))
    }

    fn trigger_target_params(
        &self,
        target: &Expr,
        target_ty: &TypeExpr,
        span: Option<Span>,
    ) -> Result<Vec<ProcessParam>, LinkError> {
        if let Some(process_name) = trigger_target_process_name(target)
            && let Some(process) = self.program.process(process_name.as_str())
        {
            return Ok(process.params.clone());
        }
        let TypeExpr::Process {
            input, input_count, ..
        } = target_ty
        else {
            return Err(LinkError::InvalidTriggerTarget {
                actual: format_type_expr(target_ty),
                span,
            });
        };
        match (input_count, input.as_ref()) {
            (0, _) => Ok(Vec::new()),
            (count, TypeExpr::Object(fields)) if *count > 1 => Ok(fields
                .iter()
                .map(|field| ProcessParam {
                    name: field.name.clone(),
                    ty: field.ty.clone(),
                })
                .collect()),
            _ => Err(LinkError::InvalidTriggerTarget {
                actual: format_type_expr(target_ty),
                span,
            }),
        }
    }

    fn infer_process_output(
        &self,
        process: &ProcessDecl,
        span: Option<Span>,
    ) -> Result<TypeExpr, LinkError> {
        let mut scope = Scope::new(false, true, span);
        for param in &process.params {
            scope.bind(param.name.as_str(), self.binding_for_type(&param.ty));
        }
        scope.bind("input", Binding::Value(process_input_type(process)));
        scope.bind("inputs", Binding::Value(process_input_record_type(process)));
        let completion = self.infer_completion(&process.body, &mut scope)?;
        let mut outputs = completion.finishes;
        if completion.can_fallthrough {
            outputs.push(TypeExpr::Null);
        }
        Ok(union_type(outputs))
    }

    fn infer_completion(&self, expr: &Expr, scope: &mut Scope) -> Result<Completion, LinkError> {
        match expr {
            Expr::LabelAnnotated { expr, .. } => self.infer_completion(expr, scope),
            Expr::Finish(value) => Ok(Completion {
                finishes: vec![self.infer_expr_type(value, scope)?],
                can_fallthrough: false,
            }),
            Expr::Fail(_) => Ok(Completion {
                finishes: Vec::new(),
                can_fallthrough: false,
            }),
            Expr::Block(expressions) => {
                let mut finishes = Vec::new();
                let mut can_fallthrough = true;
                for expression in expressions {
                    if !can_fallthrough {
                        break;
                    }
                    let completion = self.infer_completion(expression, scope)?;
                    finishes.extend(completion.finishes);
                    can_fallthrough = completion.can_fallthrough;
                }
                Ok(Completion {
                    finishes,
                    can_fallthrough,
                })
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                self.infer_expr_type(condition, scope)?;
                let mut then_scope = scope.clone();
                let then_completion = self.infer_completion(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                let else_completion = self.infer_completion(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                let mut finishes = then_completion.finishes;
                finishes.extend(else_completion.finishes);
                Ok(Completion {
                    finishes,
                    can_fallthrough: then_completion.can_fallthrough
                        || else_completion.can_fallthrough,
                })
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                self.infer_expr_type(iterable, scope)?;
                let previous = scope.bind(binding.as_str(), Binding::Value(TypeExpr::Any));
                let mut completion = self.infer_completion(body, scope)?;
                scope.restore(binding.as_str(), previous);
                completion.can_fallthrough = true;
                Ok(completion)
            }
            Expr::While { condition, body } => {
                self.infer_expr_type(condition, scope)?;
                let mut completion = self.infer_completion(body, scope)?;
                completion.can_fallthrough = true;
                Ok(completion)
            }
            Expr::Assign { target, expr } if target.steps.is_empty() => {
                let ty = self.infer_expr_type(expr, scope)?;
                scope.bind(target.root.as_str(), self.binding_for_type(&ty));
                Ok(Completion::fallthrough())
            }
            other => {
                self.infer_expr_type(other, scope)?;
                Ok(Completion::fallthrough())
            }
        }
    }

    fn infer_expr_type(&self, expr: &Expr, scope: &mut Scope) -> Result<TypeExpr, LinkError> {
        self.reject_trigger_event_special_form(expr, scope.span)?;
        if matches!(expr, Expr::Variable(_) | Expr::Field { .. })
            && let Some(resource) = self.resolve_module_expr(expr, scope)
        {
            return Ok(TypeExpr::Ref(resource.resource_type));
        }
        Ok(match expr {
            Expr::LabelAnnotated { expr, .. } => self.infer_expr_type(expr, scope)?,
            Expr::Block(expressions) => {
                let mut last = TypeExpr::Null;
                for expression in expressions {
                    last = self.infer_expr_type(expression, scope)?;
                }
                last
            }
            Expr::Null
            | Expr::Bool(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Break
            | Expr::Continue
            | Expr::TypeLiteral(_) => literal_type(expr),
            Expr::Variable(name) => {
                if let Some(binding) = scope.get(name) {
                    binding_type(Some(&binding))
                } else if let Some(process_ty) = self.process_types.get(name.as_str()) {
                    process_ty.clone()
                } else if scope.allow_unknown_globals {
                    TypeExpr::Any
                } else {
                    return Err(LinkError::UnknownName {
                        name: name.to_string(),
                        span: scope.span,
                    });
                }
            }
            Expr::ProcessRef { process } => self
                .process_types
                .get(process.as_str())
                .cloned()
                .ok_or_else(|| LinkError::UnknownProcess {
                    name: process.to_string(),
                    span: scope.span,
                })?,
            Expr::HostDescriptorConstructor { type_name, .. } => TypeExpr::Ref(type_name.clone()),
            Expr::Tuple(items) => TypeExpr::List(Box::new(union_type(
                items
                    .iter()
                    .map(|item| self.infer_expr_type(item, scope))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            Expr::List(items) => TypeExpr::List(Box::new(union_type(
                items
                    .iter()
                    .map(|item| self.infer_expr_type(item, scope))
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            Expr::ListComprehension { element, clauses } => {
                let mut previous_bindings = Vec::new();
                for clause in clauses {
                    match clause {
                        ListComprehensionClause::For { binding, iterable } => {
                            let iterable_ty = self.infer_expr_type(iterable, scope)?;
                            let item_ty = self.index_type(&iterable_ty, scope.span)?;
                            previous_bindings.push((
                                binding.to_string(),
                                scope.bind(binding.as_str(), self.binding_for_type(&item_ty)),
                            ));
                        }
                        ListComprehensionClause::If { condition } => {
                            self.infer_expr_type(condition, scope)?;
                        }
                    }
                }
                let element_ty = self.infer_expr_type(element, scope)?;
                for (name, previous) in previous_bindings.into_iter().rev() {
                    scope.restore(name.as_str(), previous);
                }
                TypeExpr::List(Box::new(element_ty))
            }
            Expr::Record(entries) => TypeExpr::Object(
                entries
                    .iter()
                    .map(|(name, value)| {
                        Ok(TypeField {
                            name: name.clone(),
                            ty: self.infer_expr_type(value, scope)?,
                            optional: false,
                        })
                    })
                    .collect::<Result<Vec<_>, LinkError>>()?,
            ),
            Expr::Assign { target, expr } => {
                let ty = self.infer_expr_type(expr, scope)?;
                if target.steps.is_empty() {
                    scope.bind(target.root.as_str(), self.binding_for_type(&ty));
                }
                ty
            }
            Expr::If {
                condition,
                then_block,
                else_block,
            } => {
                self.infer_expr_type(condition, scope)?;
                let mut then_scope = scope.clone();
                let then_ty = self.infer_expr_type(then_block, &mut then_scope)?;
                let mut else_scope = scope.clone();
                let else_ty = self.infer_expr_type(else_block, &mut else_scope)?;
                scope.merge_from(then_scope);
                scope.merge_from(else_scope);
                union_type(vec![then_ty, else_ty])
            }
            Expr::For {
                binding,
                iterable,
                body,
            } => {
                self.infer_expr_type(iterable, scope)?;
                let previous = scope.bind(binding.as_str(), Binding::Value(TypeExpr::Any));
                self.infer_expr_type(body, scope)?;
                scope.restore(binding.as_str(), previous);
                TypeExpr::Null
            }
            Expr::While { condition, body } => {
                self.infer_expr_type(condition, scope)?;
                self.infer_expr_type(body, scope)?;
                TypeExpr::Null
            }
            Expr::StartProcess(_) => TypeExpr::Any,
            Expr::ResourceRef(resource) => TypeExpr::Ref(resource.resource_type.clone()),
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
                        return Ok(constructor.output_ty.clone());
                    }
                }
                let resolved_receiver = self.resolve_module_expr(receiver, scope);
                let (resource_type, receiver_alias) =
                    if let Some(resource) = resolved_receiver.as_ref() {
                        (
                            resource.resource_type.to_string(),
                            Some(resource.alias.to_string()),
                        )
                    } else {
                        let receiver_ty = self.infer_expr_type(receiver, scope)?;
                        (
                            self.resource_type_for_type(&receiver_ty).ok_or_else(|| {
                                LinkError::UnresolvedReceiver {
                                    operation: operation.to_string(),
                                    span: scope.span,
                                }
                            })?,
                            None,
                        )
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
                let binding = self
                    .surface
                    .resources
                    .resolve_operation(&resource_type, operation)
                    .ok_or_else(|| LinkError::UnknownResourceOperation {
                        resource_type: resource_type.clone(),
                        operation: operation.to_string(),
                        span: scope.span,
                    })?;
                if crate::is_trigger_resource_type(&resource_type)
                    && let Some(trigger_operation) =
                        crate::TriggerHostOperation::from_receiver_method(operation.as_str())
                {
                    self.validate_trigger_operation_args(trigger_operation, args, scope)?
                } else {
                    binding.output_ty.clone()
                }
            }
            Expr::Await(inner) | Expr::ResultUnwrap(inner) => self.infer_expr_type(inner, scope)?,
            Expr::SleepFor(_) | Expr::SleepUntil(_) => TypeExpr::Null,
            Expr::WaitSignal { .. } => TypeExpr::Any,
            Expr::SignalRun { .. }
            | Expr::Cancel(_)
            | Expr::Print(_)
            | Expr::Yield(_)
            | Expr::Wake(_)
            | Expr::Fail(_) => TypeExpr::Null,
            Expr::Finish(inner) => self.infer_expr_type(inner, scope)?,
            Expr::BuiltinCall { name, .. } => builtin_return_type(name.as_str()),
            Expr::Field { target, field } => {
                self.field_type(&self.infer_expr_type(target, scope)?, field, scope.span)?
            }
            Expr::Index { target, .. } => {
                self.index_type(&self.infer_expr_type(target, scope)?, scope.span)?
            }
            Expr::Unary { op, .. } => match op {
                crate::ast::UnaryOp::Not => TypeExpr::Bool,
                crate::ast::UnaryOp::Negate => TypeExpr::Float,
            },
            Expr::Binary { op, .. } => binary_return_type(*op),
        })
    }
}
