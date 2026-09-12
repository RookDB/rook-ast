//! Logical Plan Types — the fully-resolved, semantically-validated query plan.
//!
//! A `LogicalPlan` tree represents what the query *intends* to do, independent of
//! how it will be physically executed. Step 3 (optimizer) rewrites this tree, and
//! Step 6 (Volcano engine) maps it to physical operators.

use crate::{ExprNode, JoinType, OrderByExpr, PredicateNode};

// ── Logical Plan Enum ─────────────────────────────────────────────────────────

/// A node in the logical query plan tree.
///
/// Leaf nodes (`TableScan`, `CteScan`) sit at the bottom; interior nodes wrap a child plan.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    TableScan(LogicalTableScan),
    Filter(LogicalFilter),
    Project(LogicalProject),
    Distinct(LogicalDistinct),
    Sort(LogicalSort),
    Limit(LogicalLimit),
    Aggregate(LogicalAggregate),
    Join(LogicalJoin),
    SetOp(LogicalSetOp),
    Subquery(LogicalSubquery),
    /// A WITH clause binding one or more CTEs for the outer query.
    Cte(LogicalCte),
    /// A recursive CTE binding (`WITH RECURSIVE`).
    RecursiveCte(LogicalRecursiveCte),
    /// A scan over a CTE that has already been materialised by the enclosing Cte node.
    CteScan(LogicalCteScan),
    /// Insert rows from a SELECT plan into a table (INSERT INTO ... SELECT).
    Insert(LogicalInsert),
}

// ── Leaf nodes ────────────────────────────────────────────────────────────────

/// Scan a single table, yielding all rows.
///
/// When `system_table_name` is `Some(name)`, this scan targets a system table
/// under `database/system/{name}.dat` (e.g. `"tables"`, `"columns"`,
/// `"databases"`).  The planner sets this flag when the SQL query references
/// `information_schema.<table_name>`, routing the scan to the physical system
/// table instead of a regular user table.
#[derive(Debug, Clone)]
pub struct LogicalTableScan {
    /// Unqualified table name (resolved from catalog).
    pub table: String,
    /// Optional alias from the FROM clause.
    pub alias: Option<String>,
    /// The resolved column schema of this table.
    pub schema: ColumnSchema,
    /// If `Some(name)`, this scan reads from `database/system/{name}.dat`.
    /// Set by the planner when the SQL query references `information_schema.xxx`.
    pub system_table_name: Option<String>,
}

// ── Operator nodes ────────────────────────────────────────────────────────────

/// Apply a predicate to filter rows from the child plan.
#[derive(Debug, Clone)]
pub struct LogicalFilter {
    pub predicate: PredicateNode,
    pub child: Box<LogicalPlan>,
}

/// Project a subset of columns / expressions from the child plan.
#[derive(Debug, Clone)]
pub struct LogicalProject {
    pub expressions: Vec<NamedExpr>,
    pub child: Box<LogicalPlan>,
}

/// Eliminate duplicate rows (`SELECT DISTINCT`).
#[derive(Debug, Clone)]
pub struct LogicalDistinct {
    pub child: Box<LogicalPlan>,
}

/// Sort rows according to ORDER BY expressions.
///
/// If `limit` is `Some(k)`, only the top-k rows are needed. This enables the
/// physical sort operator to use a bounded heap (O(N log k)) instead of a
/// full sort (O(N log N)). Set during the limit-pushdown optimization pass.
#[derive(Debug, Clone)]
pub struct LogicalSort {
    pub order_by: Vec<OrderByExpr>,
    pub child: Box<LogicalPlan>,
    /// Optional bound — when set, only the top-k rows are needed.
    pub limit: Option<u64>,
}

/// Restrict the number of output rows (`LIMIT n / OFFSET m`).
#[derive(Debug, Clone)]
pub struct LogicalLimit {
    pub limit: u64,
    pub offset: u64,
    pub child: Box<LogicalPlan>,
}

/// GROUP BY aggregation, optionally filtered by HAVING.
#[derive(Debug, Clone)]
pub struct LogicalAggregate {
    pub group_by: Vec<ExprNode>,
    pub aggregates: Vec<AggregateExpr>,
    pub having: Option<PredicateNode>,
    pub child: Box<LogicalPlan>,
}

/// Combine two child plans via a join predicate.
#[derive(Debug, Clone)]
pub struct LogicalJoin {
    pub left: Box<LogicalPlan>,
    pub right: Box<LogicalPlan>,
    pub join_type: JoinType,
    pub condition: Option<PredicateNode>,
}

/// Set operation between two child plans (UNION / INTERSECT / EXCEPT).
#[derive(Debug, Clone)]
pub struct LogicalSetOp {
    pub op: SetOpType,
    pub left: Box<LogicalPlan>,
    pub right: Box<LogicalPlan>,
    pub all: bool,
}

/// A subquery (table-valued or scalar).
#[derive(Debug, Clone)]
pub struct LogicalSubquery {
    pub subquery: Box<LogicalPlan>,
    pub alias: Option<String>,
}

/// A non-recursive CTE binding.
///
/// During physical planning the executor evaluates `inner` first, materialises
/// its tuples in memory, then plans `outer` with a `CteScan` leaf available
/// for each reference to `name`.
#[derive(Debug, Clone)]
pub struct LogicalCte {
    /// The CTE alias name.
    pub name: String,
    /// The CTE body — evaluated once and cached.
    pub inner: Box<LogicalPlan>,
    /// The outer SELECT that consumes the CTE.
    pub outer: Box<LogicalPlan>,
}

/// A scan over a CTE that has been materialised by an enclosing `LogicalCte` node.
///
/// The physical planner resolves this against the executor's CTE registry.
#[derive(Debug, Clone)]
pub struct LogicalCteScan {
    /// Name of the CTE to scan (matches `LogicalCte::name`).
    pub name: String,
    /// Schema of the CTE output (resolved from the inner plan).
    pub schema: ColumnSchema,
}

/// Insert rows produced by a child SELECT plan into a table.
///
/// Used for `INSERT INTO ... SELECT`. The child plan produces the rows
/// to insert, and the operator handles column mapping and heap insertion.
#[derive(Debug, Clone)]
pub struct LogicalInsert {
    /// Target table name.
    pub table: String,
    /// Explicit target columns (may be empty for `INSERT INTO t SELECT ...`).
    pub columns: Vec<String>,
    /// The SELECT plan that produces rows to insert.
    pub child: Box<LogicalPlan>,
}

/// A recursive CTE binding (`WITH RECURSIVE name AS (non_recursive UNION ALL recursive)`).
///
/// During physical planning the executor evaluates `non_recursive` first to seed the
/// working table, then iteratively evaluates `recursive` (which references the CTE via
/// CteScan) until no new tuples are produced. The accumulated result is passed to `outer`.
#[derive(Debug, Clone)]
pub struct LogicalRecursiveCte {
    /// The CTE alias name.
    pub name: String,
    /// The non-recursive term — evaluated once to seed the working table.
    pub non_recursive: Box<LogicalPlan>,
    /// The recursive term — re-evaluated each iteration. Contains CteScan leaves referencing `name`.
    pub recursive: Box<LogicalPlan>,
    /// Whether UNION ALL (keep duplicates across iterations) vs UNION DISTINCT (dedup).
    pub union_all: bool,
    /// The schema of the CTE output (resolved from the non-recursive term).
    pub schema: ColumnSchema,
    /// The outer SELECT that consumes the CTE.
    pub outer: Box<LogicalPlan>,
}

// ── Supporting types ──────────────────────────────────────────────────────────

/// Type of set operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    Union,
    Intersect,
    Except,
}

/// A named expression (used in projections).
#[derive(Debug, Clone)]
pub struct NamedExpr {
    pub name: String,
    pub expr: ExprNode,
}

/// An aggregate function call.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    pub function: AggregateFunction,
    pub args: Vec<ExprNode>,
    pub alias: Option<String>,
    pub distinct: bool,
}

/// Supported aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Schema of a table or intermediate result — column metadata after semantic
/// analysis has resolved names and types.
#[derive(Debug, Clone)]
pub struct ColumnSchema {
    pub columns: Vec<ColumnInfo>,
}

/// Metadata for a single column in a schema.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    /// Human-readable type string (e.g. \"INT\", \"VARCHAR(100)\").
    pub data_type: String,
    pub nullable: bool,
}

impl ColumnSchema {
    pub fn empty() -> Self {
        Self { columns: Vec::new() }
    }

    pub fn find(&self, name: &str) -> Option<&ColumnInfo> {
        self.columns.iter().find(|c| c.name.eq_ignore_ascii_case(name))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
    }
}

// ── LogicalPlan helpers ───────────────────────────────────────────────────────

impl LogicalPlan {
    /// Return a human-readable label for this node type.
    pub fn label(&self) -> &str {
        match self {
            LogicalPlan::TableScan(_) => "TableScan",
            LogicalPlan::Filter(_) => "Filter",
            LogicalPlan::Project(_) => "Project",
            LogicalPlan::Distinct(_) => "Distinct",
            LogicalPlan::Sort(_) => "Sort",
            LogicalPlan::Limit(_) => "Limit",
            LogicalPlan::Aggregate(_) => "Aggregate",
            LogicalPlan::Join(_) => "Join",
            LogicalPlan::SetOp(_) => "SetOp",
            LogicalPlan::Subquery(_) => "Subquery",
            LogicalPlan::Cte(_) => "Cte",
            LogicalPlan::RecursiveCte(_) => "RecursiveCte",
            LogicalPlan::CteScan(_) => "CteScan",
            LogicalPlan::Insert(_) => "Insert",
        }
    }

    /// Pretty-print the plan tree to stdout (debug mode).
    pub fn debug_print(&self) {
        self.debug_print_indent(0);
    }

    fn debug_print_indent(&self, depth: usize) {
        let indent = "  ".repeat(depth);
        match self {
            LogicalPlan::TableScan(t) => {
                println!(
                    "{}TableScan: {} [{} columns]",
                    indent,
                    t.table,
                    t.schema.columns.len()
                );
            }
            LogicalPlan::Filter(f) => {
                println!("{}Filter: {:?}", indent, f.predicate);
                f.child.debug_print_indent(depth + 1);
            }
            LogicalPlan::Project(p) => {
                let names: Vec<&str> = p.expressions.iter().map(|e| e.name.as_str()).collect();
                println!("{}Project: [{}]", indent, names.join(", "));
                p.child.debug_print_indent(depth + 1);
            }
            LogicalPlan::Distinct(d) => {
                println!("{}Distinct", indent);
                d.child.debug_print_indent(depth + 1);
            }
            LogicalPlan::Sort(s) => {
                let limit_str = s.limit.map(|l| format!(" (top={})", l)).unwrap_or_default();
                println!("{}Sort: {} columns{}", indent, s.order_by.len(), limit_str);
                s.child.debug_print_indent(depth + 1);
            }
            LogicalPlan::Limit(l) => {
                println!("{}Limit: {} (offset: {})", indent, l.limit, l.offset);
                l.child.debug_print_indent(depth + 1);
            }
            LogicalPlan::Aggregate(a) => {
                println!(
                    "{}Aggregate: group by {} cols, {} aggregates",
                    indent,
                    a.group_by.len(),
                    a.aggregates.len()
                );
                a.child.debug_print_indent(depth + 1);
            }
            LogicalPlan::Join(j) => {
                println!(
                    "{}Join: {:?} {:?}",
                    indent, j.join_type, j.condition
                );
                j.left.debug_print_indent(depth + 1);
                j.right.debug_print_indent(depth + 1);
            }
            LogicalPlan::SetOp(s) => {
                println!("{}SetOp: {:?} (all={})", indent, s.op, s.all);
                s.left.debug_print_indent(depth + 1);
                s.right.debug_print_indent(depth + 1);
            }
            LogicalPlan::Subquery(sq) => {
                println!("{}Subquery: alias={:?}", indent, sq.alias);
                sq.subquery.debug_print_indent(depth + 1);
            }
            LogicalPlan::Cte(c) => {
                println!("{}Cte: name={}", indent, c.name);
                println!("{} inner:", indent);
                c.inner.debug_print_indent(depth + 1);
                println!("{} outer:", indent);
                c.outer.debug_print_indent(depth + 1);
            }
            LogicalPlan::RecursiveCte(rc) => {
                let op = if rc.union_all { "UNION ALL" } else { "UNION" };
                println!("{}RecursiveCte: name={} ({})", indent, rc.name, op);
                println!("{} non_recursive:", indent);
                rc.non_recursive.debug_print_indent(depth + 1);
                println!("{} recursive:", indent);
                rc.recursive.debug_print_indent(depth + 1);
                println!("{} outer:", indent);
                rc.outer.debug_print_indent(depth + 1);
            }
            LogicalPlan::CteScan(cs) => {
                println!("{}CteScan: {} [{} columns]", indent, cs.name, cs.schema.columns.len());
            }
            LogicalPlan::Insert(inp) => {
                println!("{}Insert into '{}'", indent, inp.table);
                inp.child.debug_print_indent(depth + 1);
            }
        }
    }
}
