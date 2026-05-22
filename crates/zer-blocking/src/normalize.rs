use unicode_normalization::UnicodeNormalization;

/// Strip hyphens and spaces from a license plate and uppercase it.
/// "25-XKL-9" becomes "25XKL9".
pub fn normalize_plate(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

/// Transliterate non-Latin Unicode to ASCII via any_ascii, then apply
/// standard normalization (NFKD diacritic stripping + uppercase + collapse whitespace).
/// Useful for Arabic/Cyrillic name input before phonetic encoding.
pub fn transliterate_and_normalize(s: &str) -> String {
    let ascii = any_ascii::any_ascii(s);
    normalize_text(&ascii)
}

pub fn normalize_text(s: &str) -> String {
    s.nfkd()
        .filter(|c| c.is_ascii())
        .collect::<String>()
        .to_ascii_uppercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn normalize_digits_only(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_digit()).collect()
}

/// Strip common Dutch tussenvoegsel prefixes so that
/// "VAN DEN BERG" and "BERG" produce the same phonetic key.
pub fn strip_tussenvoegsel(s: &str) -> &str {
    const PREFIXES: &[&str] = &[
        "VAN DER ", "VAN DEN ", "VAN DE ", "VAN HET ", "VAN 'T ",
        "VAN T ", "VAN ", "DEN ", "DER ", "DE ", "TEN ", "TER ",
        "TE ", "IN 'T ", "IN T ", "OP DEN ", "OP DE ", "OP HET ",
        "OP ", "V/D ", "V.D. ",
    ];

    let upper = s.to_ascii_uppercase();
    for prefix in PREFIXES {
        if upper.starts_with(prefix) {
            return &s[prefix.len()..];
        }
    }
    s
}

/// Return the last whitespace-delimited token of a name, stripping
/// tussenvoegsel from single-field full-name representations.
pub fn extract_surname_token(normalized: &str) -> &str {
    let stripped = strip_tussenvoegsel(normalized);
    stripped.split_whitespace().last().unwrap_or(stripped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_diacritics() {
        assert_eq!(normalize_text("Müller"), "MULLER");
        assert_eq!(normalize_text("Çelik"), "CELIK");
        assert_eq!(normalize_text("van den Berg"), "VAN DEN BERG");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_text("  Jan   de  Vries  "), "JAN DE VRIES");
    }

    #[test]
    fn normalize_digits_only_strips_punctuation() {
        assert_eq!(normalize_digits_only("555-123-4567"), "5551234567");
        assert_eq!(normalize_digits_only("+31 (0)20 123 4567"), "310201234567");
        assert_eq!(normalize_digits_only("no digits"), "");
    }

    #[test]
    fn strip_tussenvoegsel_van_der() {
        assert_eq!(strip_tussenvoegsel("VAN DER WAL"), "WAL");
        assert_eq!(strip_tussenvoegsel("VAN DEN BERG"), "BERG");
        assert_eq!(strip_tussenvoegsel("DE VRIES"), "VRIES");
        assert_eq!(strip_tussenvoegsel("JANSEN"), "JANSEN");
    }

    #[test]
    fn extract_surname_token_from_full_name() {
        let n = normalize_text("Saddam Hussein Al-Tikriti");
        assert_eq!(extract_surname_token(&n), "AL-TIKRITI");
        let n2 = normalize_text("Jan de Vries");
        assert_eq!(extract_surname_token(&n2), "VRIES");
    }

    #[test]
    fn normalize_plate_strips_hyphens_and_uppercases() {
        assert_eq!(normalize_plate("25-XKL-9"), "25XKL9");
        assert_eq!(normalize_plate("LD-321-F"), "LD321F");
        assert_eq!(normalize_plate("CX-180-W"), "CX180W");
        assert_eq!(normalize_plate("cx180w"), "CX180W");
    }

    #[test]
    fn transliterate_and_normalize_handles_latin_diacritics() {
        // For Latin input already-ASCII or diacritic-stripped by any_ascii
        assert_eq!(transliterate_and_normalize("Müller"), "MULLER");
        assert_eq!(transliterate_and_normalize("Çelik"), "CELIK");
    }
}
