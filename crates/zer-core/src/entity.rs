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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityMember {
    pub record_id: RecordId,
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
