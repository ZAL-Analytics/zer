use std::collections::{HashMap, HashSet};
use zer_core::{record::RecordId, traits::BlockIndex};

/// Inverted index mapping blocking keys to record IDs.
pub struct InvertedIndex {
    buckets: HashMap<String, Vec<RecordId>>,
    record_keys: HashMap<RecordId, Vec<String>>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            record_keys: HashMap::new(),
        }
    }

    pub fn insert(&mut self, record_id: RecordId, keys: Vec<String>) {
        for key in &keys {
            self.buckets.entry(key.clone()).or_default().push(record_id);
        }
        self.record_keys.insert(record_id, keys);
    }

    /// Returns all record IDs sharing at least one key with the query keys,
    /// excluding `exclude` (the querying record itself). Result is deduplicated.
    pub fn lookup_union(&self, keys: &[String], exclude: RecordId) -> Vec<RecordId> {
        let mut seen: HashSet<RecordId> = HashSet::new();
        for key in keys {
            if let Some(ids) = self.buckets.get(key) {
                for &id in ids {
                    if id != exclude {
                        seen.insert(id);
                    }
                }
            }
        }
        seen.into_iter().collect()
    }

    /// Like [`Self::lookup_union`] but skips any bucket whose size exceeds
    /// `max_bucket_size`.  Overfull buckets have low selectivity and produce
    /// O(n²) spurious pairs; capping them prevents unbounded memory growth.
    ///
    /// Pass `max_bucket_size = 0` to disable the cap (same as `lookup_union`).
    pub fn lookup_union_capped(
        &self,
        keys: &[String],
        exclude: RecordId,
        max_bucket_size: usize,
    ) -> Vec<RecordId> {
        let mut seen: HashSet<RecordId> = HashSet::new();
        for key in keys {
            if let Some(ids) = self.buckets.get(key) {
                if max_bucket_size > 0 && ids.len() > max_bucket_size {
                    continue;
                }
                for &id in ids {
                    if id != exclude {
                        seen.insert(id);
                    }
                }
            }
        }
        seen.into_iter().collect()
    }

    /// Returns the size of a specific bucket, or 0 if not present.
    pub fn bucket_size(&self, key: &str) -> usize {
        self.buckets.get(key).map_or(0, |v| v.len())
    }

    /// Returns the number of buckets exceeding `max_size`.
    pub fn oversized_buckets(&self, max_size: usize) -> usize {
        self.buckets.values().filter(|v| v.len() > max_size).count()
    }

    /// Enumerate all canonical `(i < j)` candidate pairs directly from bucket contents.
    ///
    /// Pass `max_bucket_size = 0` to disable the cap.
    pub fn all_pairs(
        &self,
        id_to_idx: &HashMap<RecordId, usize>,
        max_bucket_size: usize,
    ) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for bucket in self.buckets.values() {
            if max_bucket_size > 0 && bucket.len() > max_bucket_size {
                continue;
            }
            let indices: Vec<usize> = bucket
                .iter()
                .filter_map(|id| id_to_idx.get(id).copied())
                .collect();
            for a in 0..indices.len() {
                for b in (a + 1)..indices.len() {
                    let (i, j) = (indices[a], indices[b]);
                    pairs.push(if i < j { (i, j) } else { (j, i) });
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    pub fn remove(&mut self, record_id: RecordId) {
        if let Some(keys) = self.record_keys.remove(&record_id) {
            for key in keys {
                if let Some(bucket) = self.buckets.get_mut(&key) {
                    bucket.retain(|&id| id != record_id);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn record_count(&self) -> usize {
        self.record_keys.len()
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockIndex for InvertedIndex {
    fn insert(&mut self, record_id: RecordId, keys: Vec<String>) {
        self.insert(record_id, keys);
    }

    fn lookup_union(&self, keys: &[String], exclude: RecordId) -> Vec<RecordId> {
        self.lookup_union(keys, exclude)
    }

    fn remove(&mut self, record_id: RecordId) {
        self.remove(record_id);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> InvertedIndex {
        let mut idx = InvertedIndex::new();
        idx.insert(1, vec!["key_a".into(), "key_b".into()]);
        idx.insert(2, vec!["key_b".into(), "key_c".into()]);
        idx.insert(3, vec!["key_c".into(), "key_d".into()]);
        idx
    }

    #[test]
    fn lookup_union_returns_all_matching() {
        let idx = make_index();
        let mut result = idx.lookup_union(&["key_b".into()], 99);
        result.sort();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn lookup_union_deduplicates() {
        let mut idx = InvertedIndex::new();
        idx.insert(1, vec!["k1".into(), "k2".into()]);
        idx.insert(2, vec!["k1".into(), "k2".into()]);

        let result = idx.lookup_union(&["k1".into(), "k2".into()], 99);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn no_self_candidates() {
        let idx = make_index();
        let result = idx.lookup_union(&["key_a".into(), "key_b".into()], 1);
        assert!(!result.contains(&1));
    }

    #[test]
    fn remove_cleans_up() {
        let mut idx = make_index();
        idx.remove(1);
        let result = idx.lookup_union(&["key_a".into(), "key_b".into()], 99);
        assert!(!result.contains(&1));
    }

    #[test]
    fn block_index_trait_insert_and_lookup() {
        let mut idx: Box<dyn BlockIndex> = Box::new(InvertedIndex::new());
        idx.insert(10, vec!["k".into()]);
        idx.insert(20, vec!["k".into()]);
        let mut result = idx.lookup_union(&["k".into()], 99);
        result.sort();
        assert_eq!(result, vec![10, 20]);
    }

    #[test]
    fn block_index_trait_remove() {
        let mut idx: Box<dyn BlockIndex> = Box::new(InvertedIndex::new());
        idx.insert(1, vec!["x".into()]);
        idx.remove(1);
        let result = idx.lookup_union(&["x".into()], 99);
        assert!(result.is_empty());
    }

    #[test]
    fn lookup_union_capped_skips_oversized_bucket() {
        let mut idx = InvertedIndex::new();
        // "big_key" bucket has 5 records; "small_key" has 2.
        for id in 1u64..=5 {
            idx.insert(id, vec!["big_key".into()]);
        }
        idx.insert(10u64, vec!["small_key".into()]);
        idx.insert(11u64, vec!["small_key".into()]);

        // cap=3: big_key (5 entries) is skipped; small_key (2 entries) is used.
        let result = idx.lookup_union_capped(&["big_key".into(), "small_key".into()], 99, 3);
        assert!(!result.contains(&1), "big_key bucket must be skipped");
        assert!(result.contains(&10), "small_key bucket must be included");
        assert!(result.contains(&11), "small_key bucket must be included");
    }

    #[test]
    fn lookup_union_capped_zero_cap_disables_limit() {
        let mut idx = InvertedIndex::new();
        for id in 1u64..=10 {
            idx.insert(id, vec!["k".into()]);
        }
        // cap=0 means no limit; all 9 non-excluded records returned.
        let result = idx.lookup_union_capped(&["k".into()], 1, 0);
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn oversized_buckets_count_is_correct() {
        let mut idx = InvertedIndex::new();
        for id in 1u64..=5 {
            idx.insert(id, vec!["big".into()]);
        }
        idx.insert(10u64, vec!["small".into()]);
        assert_eq!(idx.oversized_buckets(4), 1);
        assert_eq!(idx.oversized_buckets(5), 0);
        assert_eq!(idx.oversized_buckets(0), 2);
    }
}
