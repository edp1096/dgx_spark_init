package server

import (
	"sparktalk/internal/workflows"
	"strings"
)

func inspectDocumentLanguageRequest(refs []workflows.DocumentLanguageRecord) documentStageRequest {
	fields := map[string]any{}
	rows := []map[string]any{}
	for _, r := range refs {
		evidence := documentLanguageEvidence(r.Source)
		choices := append([]string{""}, sortedTermKeys(evidence)...)
		mentions := map[string]any{}
		for _, m := range r.Mentions {
			mentions[m.ID] = termObject(map[string]any{"status": termEnum("retain", "translate", "uncertain"), "source_ref": termEnum(choices...), "after": termString(), "reason": termString()})
		}
		fields[r.ID] = termObject(mentions)
		candidates := []map[string]string{}
		for _, m := range r.Mentions {
			left, right := []rune(r.Target[:m.Start]), []rune(r.Target[m.End:])
			candidates = append(candidates, map[string]string{"id": m.ID, "text": m.Text, "left_context": string(left[max(0, len(left)-80):]), "right_context": string(right[:min(80, len(right))])})
		}
		feedback := r.Reason
		if strings.HasPrefix(feedback, "번역 어휘 교정 검증 실패:") {
			feedback = "Independent validation rejected the previous proposal. Re-derive an exact replacement from the original and unchanged neighbors; do not use a reviewer's suggested wording."
		}
		rows = append(rows, map[string]any{"id": r.ID, "source": r.Source, "translation": r.Target, "candidates": candidates, "preserve_lines": r.PreserveLines, "source_evidence": evidence, "previous_validation_error": feedback})
	}
	return termRequest(`Check the listed expressions in their sentence context against the original and the user's requested translation language (Korean by default). Their Latin script alone is NOT an error. Preserve legitimate names, abbreviations, code, URLs, quotations and intentional parenthetical source-language glosses. status=translate only for ordinary running-prose words or connected clauses accidentally emitted in the wrong language (including a third-language pronoun in Korean grammar), when the intended meaning is established by the original. A pronoun can have the correct referent yet still be in the wrong output language; translate that pronoun to the corresponding pronoun in the requested language, preserving person and number. Keep a source pronoun as a pronoun: do not expand it into the antecedent's proper name, even when the referent is obvious. Do not call it an invented character. Intentional parenthetical source-language glosses are protected, including valid synonyms and inflected forms; keep them. Never rewrite surrounding prose, normalize all scripts, or invent a name. after is inserted VERBATIM between left_context and right_context. Return only text that fits those unchanged neighbors: no dictionary definitions, example sentences, tilde placeholders, or repeated surrounding parentheses/punctuation. Do not duplicate meaning already translated in adjacent Korean text. For translate, select the source_evidence id establishing the word's meaning and return after as the complete minimal replacement for the candidate, including its attached Korean inflection. Use standard Korean pronoun/particle contractions (내가, 제가, 네가), not uncontracted 나가, 저가, 너가 for a personal subject. A connected foreign clause is a single candidate: translate it coherently instead of leaving part of it untranslated. Preserve outside text and punctuation. If preserve_lines is true, keep every newline inside that candidate. For retain/uncertain, after must be empty. Use uncertain if the correction cannot be established. Give a short factual Korean reason; do not assert a universal spelling standard.`, map[string]any{"paragraphs": rows}, termObject(map[string]any{"checks": termObject(fields)}))
}

func verifyDocumentLanguageRequest(r workflows.DocumentLanguageRecord, proposed string) documentStageRequest {
	changes := []map[string]string{}
	quotes := documentLanguageEvidence(r.Source)
	for _, m := range r.Mentions {
		c := r.Checks[m.ID]
		if c.Status == "translate" {
			changes = append(changes, map[string]string{"expression": c.After, "original_evidence": quotes[c.SourceRef]})
		}
	}
	return termRequest(`Independently compare the highlighted expressions in the CURRENT translation with the ORIGINAL. Judge their meaning, referent, requested language and attached grammar. The original source is authoritative. valid=true only when every highlighted expression is supported and grammatical. Preserve the original grammatical role: if the aligned source expression is a pronoun, the replacement must remain a corresponding pronoun with the same person and number. Reject expansion of a source pronoun into an antecedent proper name, even if it refers to the same character. Accept legitimately retained names, acronyms, quotations and parenthetical source glosses; do not normalize scripts merely for uniformity. You are not shown the earlier faulty wording or a previous verdict. Check only these scoped expressions and their local syntax, not unrelated stylistic preferences. Give a brief Korean reason. If invalid, state the defect without suggesting alternative wording or placeholder templates.`, map[string]any{"source": r.Source, "current_translation": proposed, "expressions": changes}, termObject(map[string]any{"valid": map[string]any{"type": "boolean"}, "reason": termString()}))
}
