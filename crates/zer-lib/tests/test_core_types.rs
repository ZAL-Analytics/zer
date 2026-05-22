use zer_lib::prelude::*;

// ── serde round-trips ────────────────────────────────────────────────────────

#[test]
fn record_serde_json_round_trip() {
    let original = Record::new(42)
        .with_source("test")
        .insert("name", FieldValue::Text("Alice van den Berg".into()))
        .insert("age",  FieldValue::Int(35))
        .insert("score", FieldValue::Float(0.95))
        .insert("active", FieldValue::Bool(true))
        .insert("missing", FieldValue::Null);

    let json     = serde_json::to_string(&original).unwrap();
    let restored: Record = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.id, 42);
    assert_eq!(restored.source.as_deref(), Some("test"));
    assert_eq!(restored.get("name"),    Some(&FieldValue::Text("Alice van den Berg".into())));
    assert_eq!(restored.get("age"),     Some(&FieldValue::Int(35)));
    assert_eq!(restored.get("active"),  Some(&FieldValue::Bool(true)));
    assert_eq!(restored.get("missing"), Some(&FieldValue::Null));
}

#[test]
fn schema_serde_json_round_trip() {
    let original = SchemaBuilder::new()
        .field("first_name", FieldKind::Name)
        .field("last_name",  FieldKind::Name)
        .field("dob",        FieldKind::Date)
        .field("phone",      FieldKind::Phone)
        .build()
        .unwrap();

    let json:     String = serde_json::to_string(&original).unwrap();
    let restored: Schema = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.len(), 4);
    assert_eq!(restored.fields[0].name, "first_name");
    assert_eq!(restored.fields[2].kind, FieldKind::Date);
}

#[test]
fn comparison_vector_serde_json_round_trip() {
    let original = ComparisonVector::new(
        1,
        2,
        vec![ComparisonLevel::Exact, ComparisonLevel::Close, ComparisonLevel::None],
    );

    let json:     String            = serde_json::to_string(&original).unwrap();
    let restored: ComparisonVector  = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.record_a, 1);
    assert_eq!(restored.record_b, 2);
    assert_eq!(restored.levels[0], ComparisonLevel::Exact);
    assert_eq!(restored.levels[2], ComparisonLevel::None);
}

#[test]
fn entity_serde_json_round_trip() {
    let mut e = Entity::new(7);
    e.members.push(EntityMember {
        record_id: 101,
        score:     0.97,
        method:    ResolutionMethod::AutoMatch,
        source:    Some("kvk".into()),
    });
    e.members.push(EntityMember {
        record_id: 102,
        score:     0.88,
        method:    ResolutionMethod::JudgePromoted,
        source:    None,
    });

    let json:     String = serde_json::to_string(&e).unwrap();
    let restored: Entity = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.id, 7);
    assert_eq!(restored.members.len(), 2);
    assert_eq!(restored.members[0].record_id, 101);
    assert_eq!(restored.members[1].method, ResolutionMethod::JudgePromoted);
}

// ── Schema builder ───────────────────────────────────────────────────────────

#[test]
fn schema_builder_rejects_empty() {
    assert!(SchemaBuilder::new().build().is_err());
}

#[test]
fn schema_fields_of_kind_correct() {
    let s = SchemaBuilder::new()
        .field("first_name", FieldKind::Name)
        .field("last_name",  FieldKind::Name)
        .field("dob",        FieldKind::Date)
        .build()
        .unwrap();

    let names: Vec<&str> = s.fields_of_kind(FieldKind::Name).collect();
    assert_eq!(names, vec!["first_name", "last_name"]);

    let addrs: Vec<&str> = s.fields_of_kind(FieldKind::Address).collect();
    assert!(addrs.is_empty());
}

// ── ComparisonLevel ordering ─────────────────────────────────────────────────

#[test]
fn comparison_level_total_order() {
    use std::cmp::Ordering;
    assert_eq!(ComparisonLevel::Exact.cmp(&ComparisonLevel::Close),   Ordering::Greater);
    assert_eq!(ComparisonLevel::None.cmp(&ComparisonLevel::Partial),  Ordering::Less);
    assert_eq!(ComparisonLevel::Close.cmp(&ComparisonLevel::Close),   Ordering::Equal);
}

// ── Trait object safety ──────────────────────────────────────────────────────

#[test]
fn core_traits_are_object_safe() {
    let _: Box<dyn BlockIndex>;
    let _: Box<dyn Blocker>;
    let _: Box<dyn ComparatorTrait>;
    let _: Box<dyn ScorerTrait>;
    let _: Box<dyn Clusterer>;
    let _: Box<dyn EntityStore>;
    let _: Box<dyn Judge>;
}
