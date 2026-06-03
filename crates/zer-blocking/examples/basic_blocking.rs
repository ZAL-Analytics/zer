/// Example: manual and auto-configured blocking for Dutch person, SIS II,
/// and ANPR records.
///
/// Demonstrates:
/// 1. Defining a schema with `SchemaBuilder`
/// 2. Building a `CompositeBlocker` manually with individual keys
/// 3. Building a `CompositeBlocker` automatically with `BlockerFactory`
/// 4. SIS II alias blocking with `AliasPhoneticKey` and `FuzzyYearKey`
/// 5. ANPR license-plate blocking with `LicensePlateNormKey` and `PlateOCRFuzzyKey`
/// 6. Domain-category shortcuts via `BlockerFactory::from_schema_category`
/// 7. User-defined categories via `CustomSchemaCategory` and `BlockerFactory::from_custom_category`
use zer_blocking::{
    keys::{
        AliasPhoneticKey, DateFragmentKey, DateGranularity, FuzzyYearKey, LicensePlateNormKey,
        PhoneticNameDobKey, PlateOCRFuzzyKey, SuffixKey,
    },
    BlockerFactory, CompositeBlocker, CustomSchemaCategory, InvertedIndex, SchemaCategory,
};
use zer_core::{
    record::Record,
    schema::{FieldKind, SchemaBuilder},
    traits::Blocker,
};

fn main() {
    person_blocking_demo();
    sis_blocking_demo();
    anpr_blocking_demo();
    custom_category_demo();
    println!("\nAll demos passed.");
}

// ── Person (KvK) blocking ─────────────────────────────────────────────────────

fn person_blocking_demo() {
    println!("=== Person blocking (KvK / PersonRegistry) ===");

    let schema = SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("geboortedatum", FieldKind::Date)
        .field("woonplaats", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .build()
        .expect("schema must not be empty");

    let r1 = Record::new(1)
        .with_source("kvk")
        .insert("voornamen", "Johannes")
        .insert("achternaam", "van den Berg")
        .insert("geboortedatum", "1978-03-15")
        .insert("woonplaats", "Amsterdam")
        .insert("postcode", "1011AB");

    let r2 = Record::new(2)
        .with_source("kvk")
        .insert("voornamen", "J.")
        .insert("achternaam", "Berg")
        .insert("geboortedatum", "1978-03-15")
        .insert("woonplaats", "Amsterdam")
        .insert("postcode", "1011AB");

    let r3 = Record::new(3)
        .with_source("kvk")
        .insert("voornamen", "Maria")
        .insert("achternaam", "Jansen")
        .insert("geboortedatum", "1990-07-22")
        .insert("woonplaats", "Rotterdam")
        .insert("postcode", "3001XY");

    let manual = CompositeBlocker::new()
        .add(PhoneticNameDobKey::new("achternaam", "geboortedatum"))
        .add(SuffixKey::new("postcode", 4))
        .add(DateFragmentKey::new(
            "geboortedatum",
            DateGranularity::YearMonth,
        ));

    let mut idx = InvertedIndex::new();
    for r in [&r1, &r2, &r3] {
        manual.index_record(r, &schema, &mut idx);
    }

    let cands = manual.candidates(&r1, &schema, &idx);
    assert!(
        cands.contains(&2),
        "r2 (same person) must be a candidate of r1"
    );
    assert!(
        !cands.contains(&3),
        "r3 (unrelated) must NOT be a candidate of r1"
    );
    println!("  Manual: r2 found, r3 excluded. ✓");

    let auto = BlockerFactory::from_schema_category(&schema, SchemaCategory::PersonRegistry);
    let mut idx = InvertedIndex::new();
    for r in [&r1, &r2, &r3] {
        auto.index_record(r, &schema, &mut idx);
    }
    let auto_cands = auto.candidates(&r1, &schema, &idx);
    assert!(auto_cands.contains(&2));
    println!("  Auto:   r2 found via BlockerFactory. ✓");
}

// ── SIS II (WantedPersons) blocking ──────────────────────────────────────────

fn sis_blocking_demo() {
    println!("\n=== SIS II blocking (WantedPersons) ===");

    let schema = SchemaBuilder::new()
        .field("achternaam", FieldKind::Name)
        .field("voornamen", FieldKind::Name)
        .field("alias_namen", FieldKind::Alias)
        .field("geboortedatum", FieldKind::Date)
        .field("document_nummer", FieldKind::Id)
        .build()
        .expect("schema must not be empty");

    let canonical = Record::new(1)
        .with_source("sis")
        .insert("achternaam", "Benabdallah")
        .insert("voornamen", "Fatima")
        .insert("alias_namen", "Benabdallah Fatima|F. Benabdallah")
        .insert("geboortedatum", "1999-06-14")
        .insert("document_nummer", "IR7406812");

    let transposed = Record::new(2)
        .with_source("sis")
        .insert("achternaam", "Fatima")
        .insert("voornamen", "Benabdallah")
        .insert("alias_namen", "Fatima Benabdallah")
        .insert("geboortedatum", "1999-06-14")
        .insert("document_nummer", "IR7406812");

    let unrelated = Record::new(3)
        .with_source("sis")
        .insert("achternaam", "Yilmaz")
        .insert("voornamen", "Mehmet")
        .insert("alias_namen", "")
        .insert("geboortedatum", "1999-03-20")
        .insert("document_nummer", "TK9900001");

    let blocker = CompositeBlocker::new()
        .add(PhoneticNameDobKey::new("achternaam", "geboortedatum"))
        .add(AliasPhoneticKey::new("alias_namen", "geboortedatum"))
        .add(FuzzyYearKey::new("achternaam", "geboortedatum", 1));

    let mut idx = InvertedIndex::new();
    for r in [&canonical, &transposed, &unrelated] {
        blocker.index_record(r, &schema, &mut idx);
    }

    println!("  Blocking keys for canonical record:");
    for key in blocker.blocking_keys(&canonical, &schema) {
        println!("    {key}");
    }

    let cands = blocker.candidates(&canonical, &schema, &idx);
    assert!(
        cands.contains(&2),
        "transposed entry must be a candidate of the canonical entry (alias bridge)"
    );
    println!("  Transposed SIS entry found as candidate. ✓");

    let auto = BlockerFactory::from_schema_category(&schema, SchemaCategory::WantedPersons);
    let mut idx = InvertedIndex::new();
    for r in [&canonical, &transposed, &unrelated] {
        auto.index_record(r, &schema, &mut idx);
    }
    let auto_cands = auto.candidates(&canonical, &schema, &idx);
    assert!(auto_cands.contains(&2));
    println!("  Auto WantedPersons factory also finds transposed entry. ✓");
}

// ── ANPR (ANPRPassages) blocking ──────────────────────────────────────────────

fn anpr_blocking_demo() {
    println!("\n=== ANPR blocking (ANPRPassages) ===");

    let schema = SchemaBuilder::new()
        .field("kenteken", FieldKind::LicensePlate)
        .field("camera_id", FieldKind::Categorical)
        .field("tijdstip", FieldKind::Timestamp)
        .field("lat", FieldKind::GpsCoordinate)
        .field("lon", FieldKind::GpsCoordinate)
        .build()
        .expect("schema must not be empty");

    let true_passage = Record::new(1)
        .with_source("anpr")
        .insert("kenteken", "CX-180-W")
        .insert("camera_id", "CAM-A12-001")
        .insert("tijdstip", "2025-06-01T10:04:00")
        .insert("lat", "52.345")
        .insert("lon", "4.901");

    let ocr_passage = Record::new(2)
        .with_source("anpr")
        .insert("kenteken", "CX-I80-W")
        .insert("camera_id", "CAM-A12-001")
        .insert("tijdstip", "2025-06-01T10:07:00")
        .insert("lat", "52.346")
        .insert("lon", "4.902");

    let other_passage = Record::new(3)
        .with_source("anpr")
        .insert("kenteken", "25-XKL-9")
        .insert("camera_id", "CAM-A20-003")
        .insert("tijdstip", "2025-06-01T14:00:00")
        .insert("lat", "51.922")
        .insert("lon", "4.479");

    let blocker = CompositeBlocker::new()
        .add(LicensePlateNormKey::new("kenteken"))
        .add(PlateOCRFuzzyKey::new("kenteken"));

    let mut idx = InvertedIndex::new();
    for r in [&true_passage, &ocr_passage, &other_passage] {
        blocker.index_record(r, &schema, &mut idx);
    }

    println!("  OCR fuzzy keys for true passage (CX-180-W):");
    for key in blocker.blocking_keys(&true_passage, &schema) {
        println!("    {key}");
    }

    let cands = blocker.candidates(&true_passage, &schema, &idx);
    assert!(
        cands.contains(&2),
        "OCR-confused passage must be a candidate of the true passage"
    );
    assert!(
        !cands.contains(&3),
        "unrelated passage must NOT be a candidate"
    );
    println!("  OCR passage (CX-I80-W) found as candidate of true plate. ✓");
    println!("  Unrelated passage correctly excluded. ✓");

    let auto = BlockerFactory::from_schema_category(&schema, SchemaCategory::ANPRPassages);
    let mut idx = InvertedIndex::new();
    for r in [&true_passage, &ocr_passage, &other_passage] {
        auto.index_record(r, &schema, &mut idx);
    }
    let auto_cands = auto.candidates(&true_passage, &schema, &idx);
    assert!(auto_cands.contains(&2));
    println!("  Auto ANPRPassages factory also finds OCR passage. ✓");
}

// ── Custom category ───────────────────────────────────────────────────────────

fn custom_category_demo() {
    println!("\n=== Custom category (user-defined, KvK schema) ===");

    // Same schema as the KvK person demo.
    let schema = SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("geboortedatum", FieldKind::Date)
        .field("woonplaats", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .build()
        .expect("schema must not be empty");

    let r1 = Record::new(1)
        .with_source("kvk")
        .insert("voornamen", "Johannes")
        .insert("achternaam", "van den Berg")
        .insert("tussenvoegsel", "van den")
        .insert("geboortedatum", "1978-03-15")
        .insert("woonplaats", "Amsterdam")
        .insert("postcode", "1011AB");

    let r2 = Record::new(2)
        .with_source("kvk")
        .insert("voornamen", "J.")
        .insert("achternaam", "Berg")
        .insert("tussenvoegsel", "van den")
        .insert("geboortedatum", "1978-03-15")
        .insert("woonplaats", "Amsterdam")
        .insert("postcode", "1011AB");

    let r3 = Record::new(3)
        .with_source("kvk")
        .insert("voornamen", "Maria")
        .insert("achternaam", "Jansen")
        .insert("tussenvoegsel", "")
        .insert("geboortedatum", "1990-07-22")
        .insert("woonplaats", "Rotterdam")
        .insert("postcode", "3001XY");

    // ── Demo 1: mirror PersonRegistry using only the custom builder ───────────
    //
    // This composes the same keys that BlockerFactory::from_schema would select
    // for this schema, but assembled explicitly by the user.
    let category = CustomSchemaCategory::new()
        .with_phonetic_name_dob() // PhoneticNameDobKey(achternaam, geboortedatum)
        .with_address_initial() // AddressInitialKey(woonplaats, voornamen)
        .with_id_suffix(4) // SuffixKey(postcode, 4)
        .with_exact_categorical(); // ExactFieldKey(tussenvoegsel)

    let blocker = BlockerFactory::from_custom_category(&schema, category);
    let mut idx = InvertedIndex::new();
    for r in [&r1, &r2, &r3] {
        blocker.index_record(r, &schema, &mut idx);
    }

    println!("  Blocking keys for r1 (custom PersonRegistry mirror):");
    for key in blocker.blocking_keys(&r1, &schema) {
        println!("    {key}");
    }

    let cands = blocker.candidates(&r1, &schema, &idx);
    assert!(
        cands.contains(&2),
        "r2 (same person) must be a candidate of r1"
    );
    assert!(
        !cands.contains(&3),
        "r3 (unrelated) must NOT be a candidate of r1"
    );
    println!("  r2 found, r3 excluded. ✓");

    // ── Demo 2: id-suffix-only variant (e.g. postcode deduplication) ─────────
    //
    // A stripped-down category that only blocks on postcode suffix.
    // Useful when names are unreliable but postcodes are authoritative.
    let postcode_only = CustomSchemaCategory::new().with_id_suffix(4);
    let blocker2 = BlockerFactory::from_custom_category(&schema, postcode_only);
    let mut idx2 = InvertedIndex::new();
    for r in [&r1, &r2, &r3] {
        blocker2.index_record(r, &schema, &mut idx2);
    }

    let cands2 = blocker2.candidates(&r1, &schema, &idx2);
    assert!(
        cands2.contains(&2),
        "r2 (same postcode) must be a candidate under postcode-only"
    );
    assert!(
        !cands2.contains(&3),
        "r3 (different postcode) must NOT be a candidate"
    );
    println!("  Postcode-only variant: r2 found, r3 excluded. ✓");

    // ── Demo 3: escape hatch, inject a SuffixKey via with_key ───────────────
    //
    // `with_key` accepts any type that implements `BlockingKey`, so you can
    // plug in keys that the built-in rules don't cover.
    let escape_hatch = CustomSchemaCategory::new().with_key(SuffixKey::new("postcode", 4));
    let blocker3 = BlockerFactory::from_custom_category(&schema, escape_hatch);
    let mut idx3 = InvertedIndex::new();
    for r in [&r1, &r2, &r3] {
        blocker3.index_record(r, &schema, &mut idx3);
    }

    let keys3 = blocker3.blocking_keys(&r1, &schema);
    assert!(
        !keys3.is_empty(),
        "escape-hatch key must produce at least one key"
    );
    println!(
        "  Escape-hatch with_key: {} key(s) produced. ✓",
        keys3.len()
    );
}
