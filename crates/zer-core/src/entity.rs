use crate::record::RecordId;

pub type EntityId = u64;

/// How an entity member was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResolutionMethod {
    AutoMatch,
    JudgePromoted,
    JudgeDemoted,
    Manual,
}

/// A record's membership in an entity, with its resolution score and method.
///
/// `record_key` is the natural key value (e.g. BSN, UUID) as provided by the
/// user via `zer_adapters::DatasetConfig`.  For records created with
/// [`crate::record::Record::new`] it falls back to the numeric ID as a string.
/// This key is stored in the `.zes` file so that the cluster output maps
/// directly to the user's own record identifiers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityMember {
    pub record_id: RecordId,
    pub record_key: String,
    pub score: f32,
    pub method: ResolutionMethod,
    pub source: Option<String>,
}

/// A resolved entity grouping one or more records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub members: Vec<EntityMember>,
}

impl Entity {
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            members: vec![],
        }
    }

    pub fn member_ids(&self) -> impl Iterator<Item = RecordId> + '_ {
        self.members.iter().map(|m| m.record_id)
    }
}
