use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::artifact::{
    HostRequirements, ModuleArtifact, host_requirements_for_program_with_catalog,
};
use crate::ast::{
    AssignPathStep, AstString, Declaration, Expr, ListComprehensionClause, ProcessDecl,
    ProcessParam, Program, ResourceRefExpr, TypeExpr, TypeField, format_type_expr,
};
use crate::lexer::Span;

include!("linker/catalog.rs");
include!("linker/host.rs");
include!("linker/errors.rs");
include!("linker/pass_setup.rs");
include!("linker/lower_expr.rs");
include!("linker/pass_validation.rs");
include!("linker/type_helpers.rs");
include!("linker/tests.rs");
