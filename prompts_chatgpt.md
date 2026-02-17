# Prompts ChatGPT - Actions, Outcomes, Effets (Cohen d / g / SMD)

Ce document contient les prompts utilises dans le pipeline, en version copier-coller pour tester directement dans ChatGPT.

## Utilisation rapide

1. Prends des extraits texte de ton PDF (snippets).
2. Formate-les en JSON.
3. Colle le `System prompt` puis le `User prompt` correspondant.
4. Demande une reponse JSON stricte.

Format snippets attendu:

```json
[
  {
    "snippet_id": "s1",
    "text": "..."
  },
  {
    "snippet_id": "s2",
    "text": "..."
  }
]
```

Format candidates effets attendu:

```json
[
  {
    "candidate_id": "c1",
    "text": "..."
  },
  {
    "candidate_id": "c2",
    "text": "..."
  }
]
```

---

## 1) Actions - Router (selection snippets pertinents)

### System prompt

```text
You are a strict relevance router for extracting study actions. Return JSON only.
```

### User prompt (template)

```text
Mark snippet as relevant only if it contains explicit information for at least one of:
- intervention
- comparator/control
- population/participants
- setting/context
- inclusion or exclusion criteria
Schema:
{
  "items": [
    {
      "snippet_id": "string",
      "relevant": true/false,
      "reason": "short text"
    }
  ]
}

Snippets JSON:
{{SNIPPETS_JSON}}
```

---

## 2) Actions - Extractor (structure finale)

### System prompt

```text
You extract structured study actions from snippets. Return strict JSON only.
```

### User prompt (template)

```text
Extract items only when explicitly present in text. Do not infer beyond snippet.
Schema:
{
  "interventions": [
    {
      "snippet_id": "string",
      "label": "string",
      "components": ["string"],
      "dose_duration": "string|unknown",
      "quote": "exact short quote"
    }
  ],
  "comparators": [
    {
      "snippet_id": "string",
      "label": "string",
      "components": ["string"],
      "dose_duration": "string|unknown",
      "quote": "exact short quote"
    }
  ],
  "population": [
    {
      "snippet_id": "string",
      "text": "string",
      "quote": "exact short quote"
    }
  ],
  "setting": [
    {
      "snippet_id": "string",
      "label": "string",
      "quote": "exact short quote"
    }
  ],
  "inclusion_exclusion": [
    {
      "snippet_id": "string",
      "text": "string",
      "quote": "exact short quote"
    }
  ]
}

Snippets JSON:
{{SNIPPETS_JSON}}
```

---

## 3) Outcomes - Router

### System prompt

```text
You are a strict relevance router for extracting study outcomes. Return JSON only.
```

### User prompt (template)

```text
Mark snippet as relevant only if it contains explicit information about outcome labels, instruments, timepoints, grouping, or primary/secondary status.
Schema:
{
  "items": [
    {
      "snippet_id": "string",
      "relevant": true/false,
      "reason": "short text"
    }
  ]
}

Snippets JSON:
{{SNIPPETS_JSON}}
```

---

## 4) Outcomes - Extractor

### System prompt

```text
You extract outcome metadata from snippets. Return strict JSON only.
```

### User prompt (template)

```text
Extract only explicit outcomes from each snippet.
Schema:
{
  "outcomes": [
    {
      "snippet_id": "string",
      "label": "string",
      "instrument": "string|unknown",
      "timepoints": ["string"],
      "grouping": "string|unknown",
      "primary_secondary": "primary|secondary|unknown",
      "quote": "exact short quote"
    }
  ]
}

Snippets JSON:
{{SNIPPETS_JSON}}
```

---

## 5) Effets (Cohen d / g / SMD) - Router

### System prompt

```text
You are a strict relevance router for effect-size extraction. Return JSON only.
```

### User prompt (template)

```text
Mark each snippet as relevant only if it contains an explicit effect size (Cohen d, Hedges g, SMD, effect size) with numeric value.
Schema:
{
  "items": [
    {
      "candidate_id": "string",
      "relevant": true/false,
      "reason": "short text"
    }
  ]
}

Candidates JSON:
{{CANDIDATES_JSON}}
```

---

## 6) Effets (Cohen d / g / SMD) - Extractor

### System prompt

```text
You extract effect sizes from snippets. Return strict JSON only with key 'effects'.
```

### User prompt (template)

```text
For each snippet, extract reported effect sizes when explicitly stated.
Keep only values likely to be Cohen d / Hedges g / SMD (range -5 to 5).
Also classify each extracted effect by role and analysis level.
Provide a short quote copied exactly from snippet that supports the value.
Do not invent data.
Schema:
{
  "effects": [
    {
      "candidate_id": "string",
      "effect_type": "d|g|SMD|unknown",
      "value": number,
      "ci_low": number|null,
      "ci_high": number|null,
      "outcome_hint": "string",
      "effect_role": "pooled_overall|subgroup|followup|sensitivity|individual_study_effect|unclear",
      "analysis_level": "meta_analysis|single_study|unknown",
      "grouping_label": "string",
      "outcome_label_norm": "string",
      "timepoint_label_norm": "string",
      "quote": "exact short quote"
    }
  ]
}

Known outcomes hints: {{KNOWN_OUTCOMES_HINTS}}
Candidates JSON:
{{CANDIDATES_JSON}}
```
