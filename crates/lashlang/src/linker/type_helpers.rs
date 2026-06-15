#[derive(Clone)]
struct Scope {
    bindings: BTreeMap<String, Binding>,
    allow_unknown_globals: bool,
    process_body: bool,
    span: Option<Span>,
}

impl Scope {
    fn new(allow_unknown_globals: bool, process_body: bool, span: Option<Span>) -> Self {
        Self {
            bindings: BTreeMap::new(),
            allow_unknown_globals,
            process_body,
            span,
        }
    }

    fn bind(&mut self, name: &str, binding: Binding) -> Option<Binding> {
        self.bindings.insert(name.to_string(), binding)
    }

    fn restore(&mut self, name: &str, previous: Option<Binding>) {
        match previous {
            Some(binding) => {
                self.bindings.insert(name.to_string(), binding);
            }
            None => {
                self.bindings.remove(name);
            }
        }
    }

    fn get(&self, name: &AstString) -> Option<Binding> {
        self.bindings.get(name.as_str()).cloned()
    }

    fn get_str(&self, name: &str) -> Option<Binding> {
        self.bindings.get(name).cloned()
    }

    fn merge_from(&mut self, other: Scope) {
        for (name, binding) in other.bindings {
            self.bindings.entry(name).or_insert(binding);
        }
    }
}

struct Completion {
    finishes: Vec<TypeExpr>,
    can_fallthrough: bool,
}

impl Completion {
    fn fallthrough() -> Self {
        Self {
            finishes: Vec::new(),
            can_fallthrough: true,
        }
    }
}

fn any_binding() -> Binding {
    Binding::Value(TypeExpr::Any)
}

fn binding_type(binding: Option<&Binding>) -> TypeExpr {
    match binding {
        Some(Binding::Value(ty)) => ty.clone(),
        Some(Binding::Resource { resource_type }) => TypeExpr::Ref(resource_type.as_str().into()),
        None => TypeExpr::Any,
    }
}

fn literal_type(expr: &Expr) -> TypeExpr {
    match expr {
        Expr::Null => TypeExpr::Null,
        Expr::Bool(_) => TypeExpr::Bool,
        Expr::Number(_) => TypeExpr::Float,
        Expr::String(_) => TypeExpr::Str,
        Expr::TypeLiteral(_) => TypeExpr::Any,
        Expr::Break | Expr::Continue => TypeExpr::Null,
        Expr::LabelAnnotated { expr, .. } => literal_type(expr),
        _ => TypeExpr::Any,
    }
}

fn union_type(items: Vec<TypeExpr>) -> TypeExpr {
    let mut flattened = Vec::new();
    for item in items {
        match item {
            TypeExpr::Union(items) => flattened.extend(items),
            other => flattened.push(other),
        }
    }
    let mut unique = Vec::new();
    for item in flattened {
        if !unique.contains(&item) {
            unique.push(item);
        }
    }
    match unique.as_slice() {
        [] => TypeExpr::Null,
        [one] => one.clone(),
        _ => TypeExpr::Union(unique),
    }
}

fn call_input_type(arg_types: Vec<TypeExpr>) -> TypeExpr {
    match arg_types.as_slice() {
        [] => TypeExpr::Null,
        [one] => one.clone(),
        _ => TypeExpr::List(Box::new(union_type(arg_types))),
    }
}

fn field_type(
    target: &TypeExpr,
    field: &str,
    span: Option<Span>,
    is_opaque: impl Fn(&str) -> bool + Copy,
) -> Result<TypeExpr, LinkError> {
    match target {
        TypeExpr::Any | TypeExpr::Dict => Ok(TypeExpr::Any),
        TypeExpr::Ref(name) if is_opaque(name.as_str()) => {
            Err(LinkError::OpaqueHostDescriptorAccess {
                type_name: name.to_string(),
                access: format!(".{field}"),
                span,
            })
        }
        TypeExpr::Ref(_) => Ok(TypeExpr::Any),
        TypeExpr::Object(fields) => Ok(fields
            .iter()
            .find(|candidate| candidate.name.as_str() == field)
            .map(|field| field.ty.clone())
            .unwrap_or(TypeExpr::Any)),
        TypeExpr::Union(items) => {
            let fields = items
                .iter()
                .map(|item| field_type(item, field, span, is_opaque))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(union_type(fields))
        }
        _ => Ok(TypeExpr::Any),
    }
}

fn index_type(
    target: &TypeExpr,
    span: Option<Span>,
    is_opaque: impl Fn(&str) -> bool + Copy,
) -> Result<TypeExpr, LinkError> {
    match target {
        TypeExpr::List(item) => Ok(*item.clone()),
        TypeExpr::Ref(name) if is_opaque(name.as_str()) => {
            Err(LinkError::OpaqueHostDescriptorAccess {
                type_name: name.to_string(),
                access: "[]".to_string(),
                span,
            })
        }
        TypeExpr::Ref(_) => Ok(TypeExpr::Any),
        TypeExpr::Union(items) => {
            let items = items
                .iter()
                .map(|item| index_type(item, span, is_opaque))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(union_type(items))
        }
        _ => Ok(TypeExpr::Any),
    }
}

fn builtin_return_type(name: &str) -> TypeExpr {
    match name {
        "len" | "find" | "to_int" | "ceil_div" | "floor_div" => TypeExpr::Int,
        "empty" | "contains" | "starts_with" | "ends_with" => TypeExpr::Bool,
        "to_float" => TypeExpr::Float,
        "to_string" | "trim" | "join" => TypeExpr::Str,
        "keys" | "values" | "split" | "grep_text" | "range" | "push" => {
            TypeExpr::List(Box::new(TypeExpr::Any))
        }
        "json_parse" | "validate" | "format" => TypeExpr::Any,
        _ => TypeExpr::Any,
    }
}

fn binary_return_type(op: crate::ast::BinaryOp) -> TypeExpr {
    match op {
        crate::ast::BinaryOp::Equal
        | crate::ast::BinaryOp::NotEqual
        | crate::ast::BinaryOp::Less
        | crate::ast::BinaryOp::LessEqual
        | crate::ast::BinaryOp::Greater
        | crate::ast::BinaryOp::GreaterEqual
        | crate::ast::BinaryOp::And
        | crate::ast::BinaryOp::Or => TypeExpr::Bool,
        crate::ast::BinaryOp::Add
        | crate::ast::BinaryOp::Subtract
        | crate::ast::BinaryOp::Multiply
        | crate::ast::BinaryOp::Divide
        | crate::ast::BinaryOp::Modulo => TypeExpr::Float,
    }
}

fn process_input_type(process: &ProcessDecl) -> TypeExpr {
    match process.params.as_slice() {
        [] => TypeExpr::Null,
        [param] => param.ty.clone(),
        _ => process_input_record_type(process),
    }
}

fn process_input_record_type(process: &ProcessDecl) -> TypeExpr {
    TypeExpr::Object(
        process
            .params
            .iter()
            .map(|param| TypeField {
                name: param.name.clone(),
                ty: param.ty.clone(),
                optional: false,
            })
            .collect(),
    )
}

fn process_type_for_decl(process: &ProcessDecl, output: TypeExpr) -> TypeExpr {
    TypeExpr::Process {
        input: Box::new(process_input_type(process)),
        output: Box::new(output),
        input_count: process.params.len(),
    }
}

fn module_path_for_expr(expr: &Expr) -> Option<Vec<AstString>> {
    match expr {
        Expr::LabelAnnotated { expr, .. } => module_path_for_expr(expr),
        Expr::Variable(name) => Some(vec![name.clone()]),
        Expr::Field { target, field } => {
            let mut path = module_path_for_expr(target)?;
            path.push(field.clone());
            Some(path)
        }
        Expr::ResourceRef(resource) => Some(resource.path.clone()),
        _ => None,
    }
}

fn is_trigger_event_expr(expr: &Expr) -> bool {
    matches!(
        module_path_for_expr(expr).as_deref(),
        Some([trigger, event]) if trigger.as_str() == "trigger" && event.as_str() == "event"
    )
}

fn is_trigger_event_projection_expr(expr: &Expr) -> bool {
    module_path_for_expr(expr).is_some_and(|path| {
        path.len() > 2 && path[0].as_str() == "trigger" && path[1].as_str() == "event"
    })
}

fn trigger_target_process_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::LabelAnnotated { expr, .. } => trigger_target_process_name(expr),
        Expr::Variable(name) | Expr::ProcessRef { process: name } => Some(name.to_string()),
        _ => None,
    }
}

fn trigger_target_process_label(expr: &Expr) -> String {
    trigger_target_process_name(expr).unwrap_or_else(|| "target".to_string())
}

fn expr_has_label_annotation(expr: &Expr) -> bool {
    match expr {
        Expr::LabelAnnotated { .. } => true,
        other => other.children().any(expr_has_label_annotation),
    }
}
