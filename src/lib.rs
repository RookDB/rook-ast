//! RookDB SQL Query Plan AST — shared between parser, planner, and execution engine.
//!
//! This crate defines the typed intermediate representation of parsed SQL statements.
//! It replaces the earlier JSON-based wire format between `rook-parser` and `rookdb-cli`.

use serde::{Deserialize, Serialize};

// ── Top-level query plan ──────────────────────────────────────────────────────

/// The complete representation of a parsed SQL statement.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPlan {
    Select(SelectPlan),
    Insert(InsertPlan),
    Update(UpdatePlan),
    Delete(DeletePlan),
    CreateTable(CreateTablePlan),
    CreateDatabase(CreateDatabasePlan),
    CreateIndex(CreateIndexPlan),
    DropIndex(DropIndexPlan),
    DropTable(DropTablePlan),
    Truncate(TruncatePlan),
    AlterTable(AlterTablePlan),
    CreateView(CreateViewPlan),
    DropView(DropViewPlan),
    CreateTableAsSelect(CreateTableAsSelectPlan),
    SetOperation(SetOperationPlan),
    DropDatabase(DropDatabasePlan),
    /// `VACUUM [TABLE] <name>` — reclaim space from soft-deleted rows and
    /// rebuild the table's indexes (maintenance statement).
    Vacuum(VacuumPlan),
    /// `ANALYZE [TABLE] <name>` — collect column-level histograms, HLL distinct
    /// counts, and row statistics to persist in `sys_statistics` for the CBO.
    Analyze(AnalyzePlan),
    ShowTables,
    ShowDatabases,
    UseDatabase(String),
    /// Catch-all for statement types not yet modeled.
    Unknown(String),
}

/// Plan node for `VACUUM [TABLE] <table>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VacuumPlan {
    pub table: String,
}

/// Plan node for `ANALYZE [TABLE] <table>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalyzePlan {
    pub table: String,
}

// ── Expressions ───────────────────────────────────────────────────────────────

/// A literal constant value that can appear in expressions or predicates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConstantValue {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
}

/// Arithmetic operators for expression trees.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A typed expression node — column reference, constant, arithmetic combo, or cast.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    /// A single-column identifier.
    Column(String),
    /// A compound identifier like `table.column`.
    Compound(Vec<String>),
    /// A literal constant.
    Constant(ConstantValue),
    /// Nested arithmetic expression.
    Binary {
        left: Box<ExprNode>,
        op: ArithOp,
        right: Box<ExprNode>,
    },
    /// SQL `CAST(expr AS type)` — explicit type conversion.
    Cast {
        expr: Box<ExprNode>,
        /// Target type as a string (e.g. "INT", "VARCHAR(100)", "DOUBLE PRECISION").
        data_type: String,
    },
    /// Scalar subquery — `(SELECT expr FROM ...)` used as an expression.
    /// The planner materializes this during physical plan construction.
    ScalarSubquery(SubqueryInfo),
    /// Aggregate / scalar function call — `COUNT(*)`, `SUM(price)`, etc.
    Function {
        name: String,
        args: Vec<FunctionArg>,
        distinct: bool,
    },
    /// `CASE WHEN cond1 THEN expr1 [WHEN cond2 THEN expr2] [ELSE exprN] END`.
    Case {
        when_then_pairs: Vec<(Box<ExprNode>, Box<ExprNode>)>,
        else_result: Option<Box<ExprNode>>,
    },
}

/// An argument to a function call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FunctionArg {
    /// `*` — the wildcard argument (e.g. `COUNT(*)`).
    Star,
    /// A regular expression argument.
    Expr(Box<ExprNode>),
}

// ── Predicates (WHERE / HAVING) ───────────────────────────────────────────────

/// Comparison operators used in WHERE conditions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Binary logical operators (AND / OR).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BinaryOp {
    And,
    Or,
}

/// Predicate tree — the WHERE / HAVING clause represented as a typed AST.
///
/// This is a lightweight version that gets converted into `storage_manager`'s
/// richer `Predicate` type (with VM compilation) at plan time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PredicateNode {
    /// Logical AND / OR of two sub-predicates.
    BinaryOp {
        left: Box<PredicateNode>,
        op: BinaryOp,
        right: Box<PredicateNode>,
    },
    /// NOT predicate.
    Not(Box<PredicateNode>),
    /// Column-operator-literal comparison.
    Compare {
        left: Box<ExprNode>,
        op: ComparisonOp,
        right: Box<ExprNode>,
    },
    /// `IS NULL`
    IsNull(Box<ExprNode>),
    /// `IS NOT NULL`
    IsNotNull(Box<ExprNode>),
    /// `BETWEEN low AND high`
    Between {
        expr: Box<ExprNode>,
        low: Box<ExprNode>,
        high: Box<ExprNode>,
    },
    /// `IN (list)`
    InList {
        expr: Box<ExprNode>,
        list: Vec<ExprNode>,
    },
    /// `LIKE 'pattern'`
    Like {
        expr: Box<ExprNode>,
        pattern: String,
        escape_char: Option<char>,
    },
    /// `EXISTS (subquery)` — true if the subquery produces any rows.
    Exists(SubqueryInfo),
    /// `expr IN (subquery)` — true if expr matches any row from subquery.
    InSubquery {
        expr: Box<ExprNode>,
        subquery: SubqueryInfo,
        negated: bool,
    },
    /// `IS DISTINCT FROM` — NULL-safe inequality.
    /// `a IS DISTINCT FROM b` is true when values differ or one is NULL.
    IsDistinctFrom {
        left: Box<ExprNode>,
        right: Box<ExprNode>,
    },
    /// Boolean test: `IS TRUE`, `IS FALSE`, `IS UNKNOWN` (and negated variants).
    IsBoolean {
        expr: Box<ExprNode>,
        test: BooleanTest,
        negated: bool,
    },
}

/// The kind of boolean test in an `IS TRUE`/`IS FALSE`/`IS UNKNOWN` predicate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BooleanTest {
    True,
    False,
    Unknown,
}
///
/// The planner extracts this, builds a `LogicalPlan`, and materializes
/// the result during physical plan construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubqueryInfo {
    pub select: Box<SelectPlan>,
}

/// A CTE definition — may be recursive or non-recursive.
///
/// For non-recursive CTEs (`WITH name AS (query)`), only `query` is populated.
/// For recursive CTEs (`WITH RECURSIVE name AS (non_recursive UNION ALL recursive)`),
/// `recursive_term` contains the recursive part and `union_all` indicates UNION vs UNION ALL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CteDef {
    /// The CTE alias name (used in subsequent FROM clauses).
    pub name: String,
    /// The body of the CTE (a full SELECT plan). For recursive CTEs this is the non-recursive term.
    pub query: Box<SelectPlan>,
    /// For recursive CTEs: the recursive term (the right-hand side of UNION ALL).
    pub recursive_term: Option<Box<SelectPlan>>,
    /// Whether the set operation is UNION ALL (vs UNION DISTINCT). Only meaningful for recursive CTEs.
    pub union_all: bool,
}

// ── SELECT-specific types ─────────────────────────────────────────────────────

/// A projection item in the SELECT list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectExpr {
    /// A bare expression with no alias.
    UnnamedExpr(ExprNode),
    /// Expression with an explicit alias (`expr AS alias`).
    ExprWithAlias {
        expr: ExprNode,
        alias: String,
    },
    /// `SELECT *`
    Wildcard,
    /// `SELECT table.*`
    QualifiedWildcard(String),
}

/// A table reference in the FROM clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableRef {
    pub name: String,
    pub alias: Option<String>,
}

/// Join type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Natural,
}

/// A single JOIN clause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JoinClause {
    pub relation: TableRef,
    pub join_type: JoinType,
    pub condition: Option<PredicateNode>,
}

/// ORDER BY expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderByExpr {
    pub expr: ExprNode,
    pub ascending: bool,
}

/// LIMIT / OFFSET.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LimitClause {
    pub limit: u64,
    pub offset: Option<u64>,
}

/// The full SELECT query plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectPlan {
    /// Non-recursive CTEs declared with `WITH name AS (...)` before this query.
    pub ctes: Vec<CteDef>,
    pub projections: Vec<SelectExpr>,
    pub from: Vec<TableRef>,
    pub joins: Vec<JoinClause>,
    pub selection: Option<PredicateNode>,
    pub group_by: Vec<ExprNode>,
    pub having: Option<PredicateNode>,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<LimitClause>,
    pub distinct: bool,
}

// ── DML types ─────────────────────────────────────────────────────────────────

/// A single SET assignment for UPDATE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetAssignment {
    pub column: String,
    pub value: ExprNode,
}

/// INSERT plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertPlan {
    pub table: String,
    pub columns: Vec<String>,
    pub values: Vec<Vec<ExprNode>>,
    /// Optional source SELECT for `INSERT INTO ... SELECT`.
    pub source_select: Option<Box<SelectPlan>>,
}

/// UPDATE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePlan {
    pub table: String,
    pub assignments: Vec<SetAssignment>,
    pub selection: Option<PredicateNode>,
}

/// DELETE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletePlan {
    pub table: String,
    pub selection: Option<PredicateNode>,
}

// ── DDL types ─────────────────────────────────────────────────────────────────

/// Column definition for CREATE TABLE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: String,
    pub constraints: Vec<String>,
}

/// Table-level constraint for CREATE TABLE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableConstraintDef {
    pub definition: String,
}

/// CREATE TABLE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTablePlan {
    pub table: String,
    pub if_not_exists: bool,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraintDef>,
}

/// CREATE DATABASE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDatabasePlan {
    pub database: String,
    pub if_not_exists: bool,
}

/// CREATE INDEX plan.
/// Plan node for `CREATE INDEX [name] ON table (col [, col ...])`.
///
/// Multiple columns form a composite key (ANALYSIS.md Tier 2 #8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateIndexPlan {
    pub index_name: String,
    pub table_name: String,
    /// Indexed columns in key order (one entry = single-column index).
    pub columns: Vec<String>,
}

impl CreateIndexPlan {
    /// Convenience accessor for single-column indexes.
    pub fn column_name(&self) -> &str {
        self.columns.first().map(String::as_str).unwrap_or("")
    }
}

/// DROP INDEX plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropIndexPlan {
    pub index_name: String,
    pub table_name: String,
    pub column_name: String,
    pub if_exists: bool,
}

/// DROP TABLE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropTablePlan {
    pub table: String,
    pub if_exists: bool,
    pub cascade: bool,
}

/// TRUNCATE TABLE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncatePlan {
    pub table: String,
}

/// DROP DATABASE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropDatabasePlan {
    pub database: String,
    pub if_exists: bool,
}

/// DROP VIEW plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropViewPlan {
    pub name: String,
    pub if_exists: bool,
}

/// ALTER TABLE action — single column-level operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlterTableAction {
    AddColumn {
        column_def: ColumnDef,
    },
    DropColumn {
        column: String,
    },
    RenameColumn {
        old_name: String,
        new_name: String,
    },
    RenameTable {
        new_name: String,
    },
    /// `ALTER COLUMN col SET DEFAULT literal`.
    SetDefault {
        column: String,
        /// Raw default value expression as a string (e.g. `"0"`, `"'active'"`).
        default_expr: String,
    },
    /// `ALTER COLUMN col DROP DEFAULT`.
    DropDefault {
        column: String,
    },
    /// `ALTER COLUMN col SET NOT NULL`.
    SetNotNull {
        column: String,
    },
    /// `ALTER COLUMN col DROP NOT NULL`.
    DropNotNull {
        column: String,
    },
}

/// ALTER TABLE plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlterTablePlan {
    pub table: String,
    pub action: AlterTableAction,
}

/// CREATE VIEW plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateViewPlan {
    pub name: String,
    pub or_replace: bool,
    pub query: Box<SelectPlan>,
}

/// CREATE TABLE ... AS SELECT plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTableAsSelectPlan {
    pub table: String,
    pub query: Box<SelectPlan>,
}

/// A UNION / INTERSECT / EXCEPT query plan.
///
/// Uses a string for `op` ("UNION", "INTERSECT", "EXCEPT") to avoid naming
/// collision with `rook_ast::logical::SetOpType`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetOperationPlan {
    pub left: SelectPlan,
    pub right: SelectPlan,
    /// One of "UNION", "INTERSECT", "EXCEPT".
    pub op: String,
    /// Whether this is ALL (e.g. UNION ALL) or DISTINCT (e.g. UNION).
    pub all: bool,
    /// CTEs declared before this set operation.
    pub ctes: Vec<CteDef>,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<LimitClause>,
}

pub mod logical;

// ── Conversion helpers ────────────────────────────────────────────────────────

impl QueryPlan {
    /// Return the category (DQL / DML / DDL) of this statement.
    pub fn category(&self) -> &str {
        match self {
            QueryPlan::Select(_) => "DQL",
            QueryPlan::Insert(_) | QueryPlan::Update(_) | QueryPlan::Delete(_) => "DML",
            QueryPlan::CreateTable(_)
            | QueryPlan::CreateDatabase(_)
            | QueryPlan::DropDatabase(_)
            | QueryPlan::CreateIndex(_)
            | QueryPlan::DropIndex(_)
            | QueryPlan::DropTable(_)
            | QueryPlan::Truncate(_)
            | QueryPlan::AlterTable(_)
            |            QueryPlan::CreateView(_)
            | QueryPlan::DropView(_)
            | QueryPlan::CreateTableAsSelect(_)
            | QueryPlan::Vacuum(_)
            | QueryPlan::Analyze(_)
            | QueryPlan::SetOperation(_) => "DDL",
            QueryPlan::ShowTables | QueryPlan::ShowDatabases => "DQL",
            QueryPlan::UseDatabase(_) => "DDL",
            QueryPlan::Unknown(_) => "UNKNOWN",
        }
    }

    /// Return the statement type as a human-readable string (mostly for logging).
    pub fn statement_type(&self) -> &str {
        match self {
            QueryPlan::Select(_) => "Select",
            QueryPlan::Insert(_) => "Insert",
            QueryPlan::Update(_) => "Update",
            QueryPlan::Delete(_) => "Delete",
            QueryPlan::CreateTable(_) => "CreateTable",
            QueryPlan::CreateDatabase(_) => "CreateDatabase",
            QueryPlan::CreateIndex(_) => "CreateIndex",
            QueryPlan::DropIndex(_) => "DropIndex",
            QueryPlan::DropTable(_) => "DropTable",
            QueryPlan::Truncate(_) => "Truncate",
            QueryPlan::AlterTable(_) => "AlterTable",
            QueryPlan::CreateView(_) => "CreateView",
            QueryPlan::DropView(_) => "DropView",
            QueryPlan::CreateTableAsSelect(_) => "CreateTableAsSelect",
            QueryPlan::SetOperation(_) => "SetOperation",
            QueryPlan::DropDatabase(_) => "DropDatabase",
            QueryPlan::Vacuum(_) => "Vacuum",
            QueryPlan::Analyze(_) => "Analyze",
            QueryPlan::ShowTables => "ShowTables",
            QueryPlan::ShowDatabases => "ShowDatabases",
            QueryPlan::UseDatabase(_) => "UseDatabase",
            QueryPlan::Unknown(_) => "Unknown",
        }
    }
}
