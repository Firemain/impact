from __future__ import annotations

import re
import unicodedata

try:
    from .text_normalize import normalize_inline_text
except ImportError:  # pragma: no cover
    from src.text_normalize import normalize_inline_text  # type: ignore


DOMAIN_VALUES = (
    "Intra-personal",
    "Extra-personal",
    "Material environment and education",
    "Socio-political environment",
    "Work and Activities",
    "Health",
    "unknown",
)

PREDICTOR_VALUES = (
    "Criminality and Insecurity",
    "Discrimination",
    "Social cohesion",
    "Traditionalism",
    "Years in state",
    "Academic education",
    "Family education",
    "Access local services",
    "Air and water pollution",
    "Close to green spaces",
    "Housing quality",
    "No access to health services",
    "No access to food",
    "No access to water",
    "Noise pollution",
    "Richness fauna flora and landscape",
    "Relative health",
    "Relative salary",
    "Family support",
    "Family strain",
    "Friends support",
    "Having grandparents",
    "Number of children",
    "Partnered",
    "Positive relationships",
    "Pregnancy",
    "Social closeness",
    "Social integration",
    "Tobacco and alcohol of others",
    "Contribution to others",
    "Emotional management",
    "Managing impact of past experience",
    "Positive habits",
    "Stress management",
    "Active coping",
    "Efforts in life",
    "Hypervigilence",
    "Selective secondary control",
    "Compensatory primary control",
    "Openness to new experiences",
    "Event mastery",
    "Goals in life",
    "Self-control",
    "Time management and anticipation",
    "Need of security",
    "Altruism",
    "Gratitude",
    "Honesty",
    "Agreeableness",
    "Extraversion",
    "Health locus of control",
    "Optimism",
    "Social potency",
    "Sympathy",
    "Conscientiousness",
    "Living the moment",
    "Physical attractiveness",
    "Self protection",
    "Self-efficacy",
    "Alcohol problems",
    "Drug consumption",
    "Food problems",
    "Social networks",
    "Tobacco",
    "Mental health professional",
    "Aches",
    "Diseases",
    "Sleep problems",
    "Games",
    "Intellectual activities",
    "Physical activity",
    "Political participation",
    "Sex life",
    "Spiritual and religious experiences",
    "Assessment financial situation",
    "Assessment professional life",
    "Pro perso balance",
    "Control over professional life",
    "Financial situation effort",
)

PREDICTOR_DOMAIN_MAP: dict[str, str] = {
    "Criminality and Insecurity": "Socio-political environment",
    "Discrimination": "Socio-political environment",
    "Social cohesion": "Socio-political environment",
    "Traditionalism": "Socio-political environment",
    "Years in state": "Socio-political environment",
    "Academic education": "Material environment and education",
    "Family education": "Material environment and education",
    "Access local services": "Material environment and education",
    "Air and water pollution": "Material environment and education",
    "Close to green spaces": "Material environment and education",
    "Housing quality": "Material environment and education",
    "No access to health services": "Material environment and education",
    "No access to food": "Material environment and education",
    "No access to water": "Material environment and education",
    "Noise pollution": "Material environment and education",
    "Richness fauna flora and landscape": "Material environment and education",
    "Relative health": "Extra-personal",
    "Relative salary": "Extra-personal",
    "Family support": "Extra-personal",
    "Family strain": "Extra-personal",
    "Friends support": "Extra-personal",
    "Having grandparents": "Extra-personal",
    "Number of children": "Extra-personal",
    "Partnered": "Extra-personal",
    "Positive relationships": "Extra-personal",
    "Pregnancy": "Extra-personal",
    "Social closeness": "Extra-personal",
    "Social integration": "Extra-personal",
    "Tobacco and alcohol of others": "Extra-personal",
    "Contribution to others": "Extra-personal",
    "Emotional management": "Intra-personal",
    "Managing impact of past experience": "Intra-personal",
    "Positive habits": "Intra-personal",
    "Stress management": "Intra-personal",
    "Active coping": "Intra-personal",
    "Efforts in life": "Intra-personal",
    "Hypervigilence": "Intra-personal",
    "Selective secondary control": "Intra-personal",
    "Compensatory primary control": "Intra-personal",
    "Openness to new experiences": "Intra-personal",
    "Event mastery": "Intra-personal",
    "Goals in life": "Intra-personal",
    "Self-control": "Intra-personal",
    "Time management and anticipation": "Intra-personal",
    "Need of security": "Intra-personal",
    "Altruism": "Intra-personal",
    "Gratitude": "Intra-personal",
    "Honesty": "Intra-personal",
    "Agreeableness": "Intra-personal",
    "Extraversion": "Intra-personal",
    "Health locus of control": "Intra-personal",
    "Optimism": "Intra-personal",
    "Social potency": "Intra-personal",
    "Sympathy": "Intra-personal",
    "Conscientiousness": "Intra-personal",
    "Living the moment": "Intra-personal",
    "Physical attractiveness": "Intra-personal",
    "Self protection": "Intra-personal",
    "Self-efficacy": "Intra-personal",
    "Alcohol problems": "Health",
    "Drug consumption": "Health",
    "Food problems": "Health",
    "Social networks": "Health",
    "Tobacco": "Health",
    "Mental health professional": "Health",
    "Aches": "Health",
    "Diseases": "Health",
    "Sleep problems": "Health",
    "Games": "Work and Activities",
    "Intellectual activities": "Work and Activities",
    "Physical activity": "Work and Activities",
    "Political participation": "Work and Activities",
    "Sex life": "Work and Activities",
    "Spiritual and religious experiences": "Work and Activities",
    "Assessment financial situation": "Work and Activities",
    "Assessment professional life": "Work and Activities",
    "Pro perso balance": "Work and Activities",
    "Control over professional life": "Work and Activities",
    "Financial situation effort": "Work and Activities",
}

PREDICTOR_FR_LABELS: dict[str, str] = {
    "Criminality and Insecurity": "Niveau de criminalite et d'insecurite",
    "Discrimination": "Discrimination",
    "Social cohesion": "Cohesion sociale",
    "Traditionalism": "Traditionnalisme",
    "Years in state": "Stabilite geographique",
    "Academic education": "Niveau d'education",
    "Family education": "Niveau d'education des parents",
    "Access local services": "Acces aux services et commodites",
    "Air and water pollution": "Pollution de l'air et de l'eau",
    "Close to green spaces": "Proximite d'espaces verts",
    "Housing quality": "Qualite de logement",
    "No access to health services": "Manque d'acces aux soins",
    "No access to food": "Manque d'acces a la nourriture",
    "No access to water": "Manque d'acces a l'eau potable",
    "Noise pollution": "Pollution sonore",
    "Richness fauna flora and landscape": "Richesse de la faune et flore",
    "Relative health": "Impression de sante relative",
    "Relative salary": "Salaire relatif",
    "Family support": "Soutien familial",
    "Family strain": "Epreuves familiales",
    "Friends support": "Soutien amical",
    "Having grandparents": "Grands-parents en vie",
    "Number of children": "Nombre d'enfants",
    "Partnered": "En couple",
    "Positive relationships": "Relations positives",
    "Pregnancy": "Grossesse",
    "Social closeness": "Sociabilite",
    "Social integration": "Integration sociale",
    "Tobacco and alcohol of others": "Tabac et alcool dans l'entourage",
    "Contribution to others": "Contribution aux autres",
    "Emotional management": "Gestion des emotions",
    "Managing impact of past experience": "Gestion des experiences vecues",
    "Positive habits": "Habitudes positives",
    "Stress management": "Gestion du stress",
    "Active coping": "Proactivite dans la gestion des epreuves",
    "Efforts in life": "Propension a l'effort",
    "Hypervigilence": "Hypervigilance",
    "Selective secondary control": "Controle secondaire selectif",
    "Compensatory primary control": "Controle primaire compensatoire",
    "Openness to new experiences": "Ouverture a de nouvelles experiences",
    "Event mastery": "Sentiment de maitrise des evenements",
    "Goals in life": "Buts dans la vie",
    "Self-control": "Sentiment d'autonomie et d'agentivite",
    "Time management and anticipation": "Gestion du temps",
    "Need of security": "Besoin de securite",
    "Altruism": "Altruisme",
    "Gratitude": "Gratitude",
    "Honesty": "Honnetete",
    "Agreeableness": "Agreabilite",
    "Extraversion": "Extraversion",
    "Health locus of control": "Sentiment d'avoir prise sur sa sante",
    "Optimism": "Optimisme",
    "Social potency": "Pouvoir d'influence",
    "Sympathy": "Sympathie",
    "Conscientiousness": "Conscienciosite",
    "Living the moment": "Tendance a vivre dans l'instant",
    "Physical attractiveness": "Beaute physique",
    "Self protection": "Tolerance a l'echec",
    "Self-efficacy": "Auto-efficacite",
    "Alcohol problems": "Problemes d'alcool",
    "Drug consumption": "Consommation de drogues",
    "Food problems": "Troubles du comportement alimentaire",
    "Social networks": "Consommation de reseaux sociaux",
    "Tobacco": "Consommation de tabac",
    "Mental health professional": "Diagnostic d'un probleme de sante mentale",
    "Aches": "Douleurs chroniques",
    "Diseases": "Maladies",
    "Sleep problems": "Troubles du sommeil",
    "Games": "Jeux",
    "Intellectual activities": "Activites et loisirs intellectuels",
    "Physical activity": "Activite physique",
    "Political participation": "Participation a la vie politique",
    "Sex life": "Vie sexuelle",
    "Spiritual and religious experiences": "Experiences spirituelles et religieuses",
    "Assessment financial situation": "Satisfaction vis-a-vis de sa situation financiere",
    "Assessment professional life": "Satisfaction vis-a-vis de sa vie professionnelle",
    "Pro perso balance": "Equilibre entre vie professionnelle et vie personnelle",
    "Control over professional life": "Controle sur sa vie professionnelle",
    "Financial situation effort": "Efforts financiers",
}

PREDICTOR_DESCRIPTIONS_FR: dict[str, str] = {
    "Academic education": "Annees d'etudes et qualite de l'education academique recue.",
    "Family education": "Parents investis dans l'education, climat familial stable, revenus du foyer.",
    "Air and water pollution": "Concentration des particules en suspension dans l'air et dans l'eau.",
    "Housing quality": "Domicile ressenti comme agreable a vivre et presentable a l'entourage.",
    "No access to food": "Nourriture insuffisante ou insuffisamment diversifiee.",
    "Richness fauna flora and landscape": "Surface et diversite des espaces naturels accessibles.",
    "Relative health": "Comparaison a l'entourage.",
    "Relative salary": "Comparaison aux autres et a l'entourage.",
    "Family support": "Se sent compris et peut compter sur ses proches.",
    "Family strain": "Frequence et intensite des tensions familiales.",
    "Friends support": "Peut se confier et compter sur ses amis.",
    "Positive relationships": "Relations chaleureuses et de confiance.",
    "Social closeness": "Tendance a rechercher la compagnie.",
    "Social integration": "Sentiment d'appartenance a une communaute.",
    "Managing impact of past experience": "Satisfaction vis-a-vis du passe et capacite a relativiser.",
    "Positive habits": "Hygiene de vie physique et mentale.",
    "Stress management": "Regulation des emotions negatives et resilience.",
    "Active coping": "Capacite a traiter les problemes activement.",
    "Efforts in life": "Investissement et reflexion dans ses activites.",
    "Selective secondary control": "Maintien de l'engagement envers un objectif malgre obstacles ou alternatives.",
    "Compensatory primary control": "Recours a l'aide, aux outils ou strategies compensatoires.",
    "Event mastery": "Sentiment de controler sa vie quotidienne.",
    "Goals in life": "Objectifs futurs et engagement pour les atteindre.",
    "Self-control": "Controle sur ses decisions et actions.",
    "Social potency": "Capacite a diriger et influencer les autres.",
    "Conscientiousness": "Caractere organise et responsable.",
    "Living the moment": "Priorite au present plutot qu'au passe ou futur.",
    "Self protection": "Capacite a voir le positif et eviter l'auto-blame.",
    "Games": "Frequence et duree d'activites ludiques.",
}

_UNKNOWN_TOKENS = {
    "",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "unclear",
    "unspecified",
    "overall",
}

_DOMAIN_ALIASES = {
    "intra personal": "Intra-personal",
    "intrapersonal": "Intra-personal",
    "intra personnel": "Intra-personal",
    "extra personal": "Extra-personal",
    "extrapersonal": "Extra-personal",
    "extra personnel": "Extra-personal",
    "material environment and education": "Material environment and education",
    "material environment": "Material environment and education",
    "environment material education": "Material environment and education",
    "environnement materiel education": "Material environment and education",
    "socio political environment": "Socio-political environment",
    "sociopolitical environment": "Socio-political environment",
    "social political environment": "Socio-political environment",
    "environnement socio politique": "Socio-political environment",
    "work and activities": "Work and Activities",
    "work activities": "Work and Activities",
    "travail activites": "Work and Activities",
    "health": "Health",
    "sante": "Health",
}

_PREDICTOR_ALIASES_RAW = {
    # Socio-political environment
    "level of crime and insecurity": "Criminality and Insecurity",
    "crime and insecurity": "Criminality and Insecurity",
    "criminality": "Criminality and Insecurity",
    "geographic stability": "Years in state",
    "geographical stability": "Years in state",
    # Material environment and education
    "level of education": "Academic education",
    "education level": "Academic education",
    "parents level of education": "Family education",
    "parent level of education": "Family education",
    "parents' level of education": "Family education",
    "access to services and amenities": "Access local services",
    "access to services": "Access local services",
    "access to amenities": "Access local services",
    "proximity to green spaces": "Close to green spaces",
    "richness of wildlife": "Richness fauna flora and landscape",
    "richness of fauna flora and landscape": "Richness fauna flora and landscape",
    "lack of access to food": "No access to food",
    "lack of access to water": "No access to water",
    "lack of access to health services": "No access to health services",
    # Extra-personal
    "feeling relatively healthy": "Relative health",
    "relative healthy feeling": "Relative health",
    "sociability": "Social closeness",
    "social support": "Social integration",
    "support from family": "Family support",
    "support from friends": "Friends support",
    "relationship satisfaction": "Positive relationships",
    "marital satisfaction": "Positive relationships",
    "relationship confidence": "Positive relationships",
    "relationship dedication": "Positive relationships",
    "positive communication": "Positive relationships",
    "negative communication": "Family strain",
    "negative observed communication": "Family strain",
    "poor conflict management": "Family strain",
    "problem intensity": "Family strain",
    "relationship conflict": "Family strain",
    "parental conflict": "Family strain",
    "conflict management": "Family strain",
    # Intra-personal
    "managing emotions": "Emotional management",
    "emotion management": "Emotional management",
    "management of past experiences": "Managing impact of past experience",
    "feeling in control of events": "Event mastery",
    "feeling of control over events": "Event mastery",
    "feeling of agency and autonomy": "Self-control",
    "agency and autonomy": "Self-control",
    "time management": "Time management and anticipation",
    "tendency to live in the moment": "Living the moment",
    "live in the moment": "Living the moment",
    "self-protection": "Self protection",
    "hypervigilance": "Hypervigilence",
    "extroversion": "Extraversion",
    "pleasantness": "Agreeableness",
    # Health
    "drug use": "Drug consumption",
    "drug usage": "Drug consumption",
    "mental illness diagnosis": "Mental health professional",
    "mental health diagnosis": "Mental health professional",
    # Work and Activities
    "intellectual activities and hobbies": "Intellectual activities",
    "hobbies": "Intellectual activities",
    "satisfaction with their financial situation": "Assessment financial situation",
    "satisfaction with financial situation": "Assessment financial situation",
    "satisfaction with their professional life": "Assessment professional life",
    "satisfaction with professional life": "Assessment professional life",
    "balance between professional and personal life": "Pro perso balance",
    "professional personal balance": "Pro perso balance",
    "control over their professional life": "Control over professional life",
    "control over professional life": "Control over professional life",
    "financial efforts": "Financial situation effort",
    "efforts financiers": "Financial situation effort",
    # Parenthesized canonical labels
    "(need of security)": "Need of security",
    "(altruism)": "Altruism",
}


_PREDICTOR_ALIAS_KEYS: dict[str, str] = {}
_LOW_PRIORITY_CONTEXT_PREDICTORS = {
    "Pregnancy",
    "Partnered",
    "Number of children",
    "Having grandparents",
    "Years in state",
    "Traditionalism",
}


def predictor_prompt_catalog() -> str:
    lines: list[str] = []
    for name in PREDICTOR_VALUES:
        domain = PREDICTOR_DOMAIN_MAP.get(name, "unknown")
        fr_label = PREDICTOR_FR_LABELS.get(name, "")
        description = PREDICTOR_DESCRIPTIONS_FR.get(name, "")
        line = f"- {name} | domain={domain}"
        if fr_label:
            line += f" | fr={fr_label}"
        if description:
            line += f" | description={description}"
        lines.append(line)
    return "\n".join(lines)


def derive_group_domain_predictor(
    group_raw: str,
    predictor_raw: str,
    domain_raw: str = "",
    context: str = "",
) -> tuple[str, str, str]:
    group = normalize_group_label(group_raw)
    predictor = normalize_predictor_label(predictor_raw)
    if predictor != "unknown":
        domain = PREDICTOR_DOMAIN_MAP.get(predictor, "unknown")
    else:
        domain = normalize_domain_label(domain_raw)
    return group, domain, predictor


def normalize_group_label(raw: str) -> str:
    text = normalize_inline_text(raw)
    lowered = text.lower()
    if lowered in _UNKNOWN_TOKENS:
        return "unknown"
    text = re.sub(r"\b(vs\.?|versus)\b", "and", text, flags=re.IGNORECASE)
    text = text.replace("/", " and ")
    text = re.sub(r"\s+", " ", text).strip(" ,;:.")
    if not text:
        return "unknown"
    if len(text) > 96:
        text = text[:96].rstrip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def normalize_predictor_label(raw: str) -> str:
    text = normalize_inline_text(raw)
    lowered = text.lower()
    if lowered in _UNKNOWN_TOKENS:
        return "unknown"

    key = _predictor_key(text)
    if not key or key in _UNKNOWN_TOKENS:
        return "unknown"

    exact = _PREDICTOR_ALIAS_KEYS.get(key)
    if exact:
        return exact

    fuzzy = _best_predictor_match(key)
    return fuzzy if fuzzy else "unknown"


def normalize_domain_label(raw: str) -> str:
    key = _plain_key(raw)
    if not key or key in _UNKNOWN_TOKENS:
        return "unknown"
    return _DOMAIN_ALIASES.get(key, "unknown")


def classify_domain(predictor: str, context: str = "") -> str:
    predictor_label = normalize_predictor_label(predictor)
    if predictor_label != "unknown":
        return PREDICTOR_DOMAIN_MAP.get(predictor_label, "unknown")
    context_predictor = infer_predictor_from_text(context)
    if context_predictor != "unknown":
        return PREDICTOR_DOMAIN_MAP.get(context_predictor, "unknown")
    return "unknown"


def infer_predictor_from_text(text: str, anchor_char: int | None = None) -> str:
    context_key = _predictor_key(text)
    if not context_key:
        return "unknown"

    if anchor_char is not None:
        anchor_index = max(0, int(anchor_char))
        anchor_key = len(_predictor_key(text[:anchor_index]))
        best_by_predictor: dict[str, float] = {}
        for alias_key, canonical in _PREDICTOR_ALIAS_KEYS.items():
            if not alias_key:
                continue
            cursor = 0
            while True:
                found = context_key.find(alias_key, cursor)
                if found == -1:
                    break
                token_count = len(alias_key.split())
                alias_center = found + (len(alias_key) / 2.0)
                distance = abs(alias_center - anchor_key)
                score = (token_count * 30.0) - min(distance, 300.0)
                if canonical in _LOW_PRIORITY_CONTEXT_PREDICTORS:
                    score -= 20.0
                previous = best_by_predictor.get(canonical, float("-inf"))
                if score > previous:
                    best_by_predictor[canonical] = score
                cursor = found + 1
        if best_by_predictor:
            ranked = sorted(best_by_predictor.items(), key=lambda item: item[1], reverse=True)
            if len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < 4.0:
                return "unknown"
            if ranked[0][1] >= 25.0:
                return ranked[0][0]

    scores: dict[str, int] = {}
    haystack = f" {context_key} "
    for alias_key, canonical in _PREDICTOR_ALIAS_KEYS.items():
        if not alias_key:
            continue
        needle = f" {alias_key} "
        if needle in haystack:
            current_score = len(alias_key.split())
            previous = scores.get(canonical, 0)
            if current_score > previous:
                scores[canonical] = current_score
    if not scores:
        return "unknown"

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "unknown"
    return ranked[0][0]


def _best_predictor_match(key: str) -> str:
    left_tokens = _tokenize_key(key)
    if not left_tokens:
        return ""

    best_predictor = ""
    best_score = 0.0
    for alias_key, canonical in _PREDICTOR_ALIAS_KEYS.items():
        right_tokens = _tokenize_key(alias_key)
        if not right_tokens:
            continue
        overlap = len(left_tokens & right_tokens)
        if overlap == 0:
            continue

        precision = overlap / max(1, len(left_tokens))
        recall = overlap / max(1, len(right_tokens))
        harmonic = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        if key in alias_key or alias_key in key:
            harmonic += 0.12
        if overlap == 1 and len(left_tokens) > 1 and len(right_tokens) > 1:
            harmonic -= 0.12

        if harmonic > best_score:
            best_score = harmonic
            best_predictor = canonical
    return best_predictor if best_score >= 0.78 else ""


def _tokenize_key(value: str) -> set[str]:
    return {token for token in value.split() if token}


def _predictor_key(value: str) -> str:
    base = normalize_inline_text(value).lower()
    base = _strip_accents(base)
    base = base.replace("&", " and ")
    base = base.replace("/", " ")
    base = base.replace("-", " ")
    base = base.replace("'", "")
    base = re.sub(r"[^a-z0-9\s]+", " ", base)
    return re.sub(r"\s+", " ", base).strip()


def _plain_key(value: str) -> str:
    base = normalize_inline_text(value).lower()
    base = _strip_accents(base)
    base = base.replace("&", " and ")
    base = base.replace("-", " ")
    base = re.sub(r"[^a-z0-9\s]+", " ", base)
    return re.sub(r"\s+", " ", base).strip()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _build_predictor_alias_keys() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical in PREDICTOR_VALUES:
        mapping[_predictor_key(canonical)] = canonical
        fr_label = PREDICTOR_FR_LABELS.get(canonical, "")
        if fr_label:
            mapping[_predictor_key(fr_label)] = canonical
    for alias, canonical in _PREDICTOR_ALIASES_RAW.items():
        mapping[_predictor_key(alias)] = canonical
    return mapping


_PREDICTOR_ALIAS_KEYS.update(_build_predictor_alias_keys())
