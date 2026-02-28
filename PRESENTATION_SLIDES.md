# Impact — Présentation Projet
## Outil d'analyse automatique d'articles scientifiques pour Boussole

---

## Slide 1 — Contexte & Objectif

**Problème** : Les fournisseurs de contenu de Boussole doivent évaluer la qualité et la fiabilité d'articles scientifiques avant de les intégrer à la base. C'est un processus **manuel, lent et subjectif**.

**Solution proposée** : Un outil automatique qui à partir d'un PDF :
1. Extrait les **tailles d'effet** (d de Cohen, g de Hedges)
2. Évalue la **renommée de l'article** (citations, revue, auteurs)
3. Analyse la **méthodologie** de l'étude (design, randomisation, groupe contrôle…)
4. Génère un **rapport structuré** en français

**Cible utilisateur** : Équipe éditoriale Boussole — outil standalone ou API intégrable

---

## Slide 2 — Architecture du Pipeline

```
PDF ──► Ingestion ──► Routage ──► Extraction ──► Qualité ──► Crédibilité ──► Évaluation ──► Score
         PyMuPDF      Heuristic    4 modes       LLM+        OpenAlex        OpenAlex        Pondéré
         pdfplumber   + LLM        d'extraction  Heuristic   Crossref        SCImago
                                                             DOI             5 dimensions
```

**7 étapes** — chaque étape produit un JSON structuré et versionné.

**Stack** : Python 3 · Streamlit · OpenAI GPT-4.1-mini · OpenAlex API · Crossref · SCImago (26 ans de données)

---

## Slide 3 — Ce qui fonctionne : Extraction des effets

### 4 modes d'extraction complémentaires

| Mode | Source | Méthode |
|------|--------|---------|
| Regex tables | Tableaux structurés | Scan des headers d/g/SMD |
| Regex texte | Passages textuels | Pattern matching `d = 0.77` |
| LLM texte | Passages textuels | GPT-4.1-mini → JSON structuré |
| LLM vision | Images de tableaux | GPT-4.1-mini vision → lecture d'image |

### Démo : Doss et al. (2009) — *Journal of Personality and Social Psychology*

**4 effets extraits correctement :**

| Groupe | Domaine | Prédicteur | d | Source |
|--------|---------|------------|---|--------|
| Mothers | Extra-personal | Positive relationships | -0.71 | Vision p.12 |
| Fathers | Extra-personal | Family strain | 0.61 | Vision p.12 |
| Mothers | Extra-personal | Positive relationships | 0.54 | Vision p.12 |
| Fathers | Extra-personal | Family strain | -0.47 | Vision p.12 |

→ Valeurs **vérifiées manuellement** dans l'article original ✅

### Post-traitement en 4 agents
1. **Extracteur** : combine les 4 modes
2. **Normaliseur** : mapping vers taxonomie canonique (80+ prédicteurs, 6 domaines)
3. **Consolidateur** : déduplique (tolérance ±0.01)
4. **Validateur** : rejette les statistiques non-d/g (β, r, OR…), sélection LLM du meilleur

---

## Slide 4 — Ce qui fonctionne : Score de Renommée

### 5 dimensions, 4 sources de données

| Dimension | Poids | Source | Exemple Doss et al. |
|-----------|-------|--------|---------------------|
| Article (citations) | 35% | OpenAlex | 636 citations → **1.00** |
| Revue | 20% | OpenAlex + SCImago | JPSP, Q1, SJR=5.16 → **0.73** |
| Auteur (h-index) | 15% | OpenAlex | h=74 (Markman) → **1.00** |
| Champ | 20% | OpenAlex | Attachment & Relationships → **1.00** |
| Réseau | 10% | OpenAlex | 2 institutions → **0.63** |

**Score global : 0.91 / 1.00** — cohérent avec un article très cité dans une top-revue

### Résolution automatique de métadonnées
- DOI extrait par LLM ou regex → validé via Crossref
- Si pas de DOI : recherche par titre + premier auteur dans OpenAlex
- Journal identifié → recherche dans 26 ans de données SCImago (1999-2024)

---

## Slide 5 — Ce qui fonctionne : Analyse méthodologique

### 6 indicateurs extraits automatiquement

| Indicateur | Méthode | Exemple RTI Report |
|------------|---------|-------------------|
| Type de design | LLM + regex | Quasi-expérimental ✅ |
| Taille d'échantillon | LLM + regex | N = 977 ✅ |
| Randomisation | LLM | Oui (2 sites sur 4) ✅ |
| Groupe contrôle | LLM | Oui (PSM) ✅ |
| Attrition | LLM | Rapportée (69%) ✅ |
| Aveugle | LLM | Non rapporté ✅ |

→ Chaque indicateur est accompagné d'une **justification** extraite du texte

**Limites assumées** : pas encore RoB2 ni ROBINS-I (analyse de risque de biais formelle)

---

## Slide 6 — Q1 : Limites de la solution

### Domaine de validité actuel

| ✅ Fonctionne | ⚠️ Partiellement | ❌ Hors scope |
|---------------|-------------------|---------------|
| Articles de journal avec d/g/SMD | Rapports (méthodologie OK, score article = 0) | PDF non-scientifiques |
| PDFs en anglais | Articles sans tableaux structurés | Formats non-PDF |
| Articles indexés OpenAlex | Articles récents (<50 citations) | Méta-analyses complexes |
| RCTs et quasi-expérimentaux | Articles en français | Articles avec uniquement OR/RR/β |

### Limites techniques spécifiques

1. **Seuls d, g, SMD sont extraits** — OR, RR, β, r, corrélations non supportés (≈40% des articles en sciences sociales n'utilisent que ces stats)
2. **Seuil |d| > 1.5 = exclusion automatique** — certains effets légitimes en psycho peuvent dépasser ce seuil
3. **Taxonomie fixe de 80 prédicteurs** — un effet mesurant un outcome hors taxonomie est rejeté
4. **Pas de normalisation de champ** — le `FieldNormScore` utilise toujours un fallback heuristique
5. **Score article = 0 pour tout document sans DOI ni indexation OpenAlex** (rapports, littérature grise)
6. **Pipeline mono-document** — pas d'analyse batch ni de méta-analyse sur un corpus

---

## Slide 7 — Q2 : Fiabilité des modèles & mitigation des erreurs

### Risques identifiés

| Risque | Probabilité | Impact |
|--------|-------------|--------|
| LLM hallucine une valeur de d | Moyen | Élevé — fausse donnée |
| LLM classifie mal le design d'étude | Faible | Modéré — score interne biaisé |
| Faux match OpenAlex (titre similaire) | Faible | Élevé — mauvais article évalué |
| Regex rate un effet dans le texte | Moyen | Faible — l'effet est simplement ignoré |

### Garde-fous implémentés

1. **Double validation regex + LLM** : le LLM texte ne retient un effet que si le regex le confirme aussi dans le passage source
2. **Citation à la source** : chaque effet extrait inclut le `quote` (passage source) et la page → vérifiable
3. **Déduplification** : si 2 modes trouvent le même effet (±0.01), un seul est retenu
4. **Seuils de confiance** : OpenAlex title-match ≥ 0.35, Crossref match ≥ 0.40
5. **Température = 0** pour tous les appels LLM → reproductibilité
6. **JSON structuré forcé** (`response_format: json_object`) → pas de texte libre à parser

### Ce qu'on pourrait ajouter (mitigation incrémentale)

- **Human-in-the-loop** : flag les effets extraits pour validation manuelle dans l'UI
- **Score de confiance par effet** : combiner calc_confidence + source_kind + dedup_sources
- **Logging de divergences** : quand regex et LLM trouvent des valeurs différentes → alerte
- **Benchmark sur corpus annoté** : comparer avec les effets réels sur 50-100 articles → precision/recall
- **Test A/B de prompts** : comparer les performances de différentes instructions d'extraction

---

## Slide 8 — Q3 : Mise en pratique

### 3 scénarios de déploiement

| Scénario | Pour qui | Coût | Sécurité |
|----------|----------|------|----------|
| **Standalone local** | 1 évaluateur, son PC | Clé API perso (~0.05€/PDF) | PDF jamais uploadé |
| **Serveur Python** | Équipe Boussole | Hébergement + clé API partagée | HTTPS, auth, logs |
| **API REST intégrée** | App Next.js Boussole | FastAPI wrapper du pipeline | JWT tokens, rate limiting |

### UX actuelle (Streamlit)

- Upload PDF → pipeline automatique → 5 onglets de résultats
- Clés API dans la sidebar (masquées, type password)
- Analyses précédentes listées et re-consultables
- Résumé narratif LLM en français

### Coûts estimés

| Poste | Coût |
|-------|------|
| API OpenAI (gpt-4.1-mini) | ~0.02–0.08 € / PDF |
| OpenAlex API | Gratuit (polite pool avec email) |
| Crossref API | Gratuit |
| Hébergement Streamlit Cloud | Gratuit (public) ou 250$/mois (private) |
| Hébergement VPS (API) | ~10-30€/mois |
| **Total par article** | **~0.05 € avec un VPS** |

### Sécurité

- Clés API **jamais dans le code** → `.env` + sidebar inputs → `.gitignore`
- PDFs traités localement, seuls les extraits textuels vont à OpenAI
- Streamlit Cloud : secrets via l'interface (jamais exposés dans le repo)
- Option API : authentification JWT + CORS restreint

### Intégration Boussole

Le pipeline Python est modulaire — chaque step est importable :
```python
from src.steps.article_evaluation import run as evaluate_article
from src.steps.effects_extract import run as extract_effects
# → renvoient des dicts JSON sérialisables
```
→ Enveloppable dans un **FastAPI** endpoint en quelques heures

---

## Slide 9 — Q4 : Axes d'amélioration incrémentaux

### Court terme (semaines)

| # | Amélioration | Effort | Impact |
|---|-------------|--------|--------|
| 1 | **Extraire OR, RR, β, r** en plus de d/g | 2-3j | Couverture ×2 |
| 2 | **Augmenter |d| max** de 1.5 à 3.0 | 1h | Moins de faux rejets |
| 3 | **Benchmark sur 50 articles** annotés | 3-5j | Mesurer precision/recall |
| 4 | **Score de confiance affiché** par effet | 1j | Transparence pour l'utilisateur |

### Moyen terme (mois)

| # | Amélioration | Effort | Impact |
|---|-------------|--------|--------|
| 5 | **RoB2 / ROBINS-I** automatisé | 2-3 sem | Analyse de biais normalisée |
| 6 | **Analyse batch** (corpus de PDFs) | 1 sem | Passage à l'échelle |
| 7 | **Normalisation de champ** (FieldStats réel) | 1-2 sem | Score article contextualisé |
| 8 | **Support multilingue** (FR, ES) | 1 sem | Couverture mondiale |

### Long terme (trimestres)

| # | Amélioration | Effort | Impact |
|---|-------------|--------|--------|
| 9 | **Méta-analyse** sur un corpus | 1-2 mois | Agrégation d'évidence |
| 10 | **Fine-tuning** d'un modèle spécialisé | 1-2 mois | Meilleure extraction, coût réduit |
| 11 | **Citation graph** (articles citants/cités) | 1 mois | Réseau d'évidence |
| 12 | **Intégration directe** dans Boussole (API) | 2 sem | Workflow automatisé |

---

## Slide 10 — Démo live

1. Upload d'un PDF (article RCT avec d de Cohen)
2. Pipeline en temps réel (≈30-60 secondes)
3. Onglet **Article** : score de renommée, métadonnées enrichies
4. Onglet **Effets** : tableau des d de Cohen extraits avec passages source
5. Onglet **Étude** : design, indicateurs méthodologiques
6. Onglet **Rapport** : résumé narratif LLM + scores

**Article démo** : Doss et al. (2009) — *The Effect of the Transition to Parenthood on Relationship Quality*
- JPSP, 636 citations, DOI auto-résolu
- 4 effets extraits (d = -0.71, 0.61, 0.54, -0.47)
- Score article : 0.91 / 1.00
