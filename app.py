import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- CONFIG ---
st.set_page_config(
    page_title="QA Copilot – 3 Phases",
    page_icon="🧪",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
.phase-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 14px;
    margin-bottom: 12px;
}
.phase-1 { background-color: #1a3a5c; color: #60aaff; border: 1px solid #2255aa; }
.phase-2 { background-color: #1a3a25; color: #60cc88; border: 1px solid #226644; }
.phase-3 { background-color: #3a1a2a; color: #cc6699; border: 1px solid #882255; }
.phase-done { background-color: #2a2a2a; color: #888; border: 1px solid #444; text-decoration: line-through; }
.stepper { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; }
.step { padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.step-active { background: #1e3a5f; color: #60aaff; border: 1px solid #2255aa; }
.step-done { background: #1a3a25; color: #60cc88; border: 1px solid #226644; }
.step-pending { background: #222; color: #555; border: 1px solid #333; }
.step-arrow { color: #444; font-size: 18px; }
.validation-box {
    background: #0e1a0e;
    border: 1px solid #226644;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Clé API Gemini", type="password")
    model_choice = st.selectbox("Modèle", [
        "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"
    ], index=0)
    st.divider()
    st.markdown("### 🗺️ Processus")
    st.markdown("""
**Phase 1 — Analyste**  
L'IA analyse l'US et pose des questions.  
→ Vous répondez et validez.

**Phase 2 — QA Lead**  
L'IA génère le plan de test (titres).  
→ Vous validez ou modifiez.

**Phase 3 — Expert**  
L'IA rédige les cas détaillés.  
→ Vous exportez.
    """)
    st.divider()
    if st.button("🔄 Nouvelle session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# --- SESSION STATE ---
defaults = {
    "phase": 1,
    "phase1_messages": [],
    "phase2_messages": [],
    "phase3_messages": [],
    "phase1_validated": False,
    "phase2_validated": False,
    "phase1_context": "",
    "phase2_draft": "",
    "us_text": "",
    "us_submitted": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- STEPPER ---
def render_stepper():
    p = st.session_state.phase
    steps = [
        ("1", "🔍 Analyse", 1),
        ("2", "📋 Plan de test", 2),
        ("3", "📝 Cas détaillés", 3),
    ]
    html = '<div class="stepper">'
    for i, (num, label, phase_num) in enumerate(steps):
        if phase_num < p:
            css = "step step-done"
            icon = "✅"
        elif phase_num == p:
            css = "step step-active"
            icon = "▶"
        else:
            css = "step step-pending"
            icon = "⏳"
        html += f'<div class="{css}">{icon} {label}</div>'
        if i < len(steps) - 1:
            html += '<span class="step-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- PROMPTS ---
SYSTEM_PHASE1 = """Tu es un Analyste QA expérimenté.
MISSION UNIQUE : Analyser la User Story et poser des questions de clarification.
INTERDIT : générer des cas de test, des titres de scénarios, ou tout plan de test.

Identifie les zones d'ombre : règles métier manquantes, comportements d'erreur non définis,
validations de champs, dépendances, cas limites non précisés.

FORMAT OBLIGATOIRE :

🔍 **PHASE 1 — Analyse & Clarifications**

**Compréhension actuelle :**
[2-3 phrases résumant l'US telle que comprise]

**Questions de clarification :**
1. [Question précise]
2. [Question précise]
...

*Répondez à ces questions pour que je puisse établir un plan de test fiable.*

RAPPEL : Ne jamais proposer de scénarios. Ne jamais inventer une règle métier."""

SYSTEM_PHASE2 = """Tu es un Lead QA.
Tu disposes d'une User Story et de ses clarifications validées.
MISSION : Générer UNIQUEMENT les TITRES des scénarios de test. Pas de détails.

FORMAT OBLIGATOIRE :

📋 **PHASE 2 — Plan de Test (Draft)**

**Contexte :**
[2-3 phrases]

**✅ Cas Nominaux (Happy Path) :**
- [Titre]

**🔄 Cas Alternatifs :**
- [Titre]

**❌ Cas d'Erreur :**
- [Titre]

**⚠️ Edge Cases / Boundary :**
- [Titre]

*Validez ce plan ou demandez des modifications. Cliquez sur "✅ Valider et générer les cas détaillés" quand vous êtes prêt.*

INTERDIT : rédiger les pré-requis, étapes ou résultats attendus."""

SYSTEM_PHASE3 = """Tu es un Expert QA (Senior Test Architect).
Tu as une liste de scénarios VALIDÉS par le testeur.
MISSION : Rédiger les cas de test COMPLETS et DÉTAILLÉS pour chaque scénario.

FORMAT OBLIGATOIRE pour chaque cas :

---

### CAS DE TEST [N] : [Titre]

| Champ | Détail |
|-------|--------|
| **ID** | TC-[N] |
| **Type** | [Nominal / Alternatif / Erreur / Limite] |
| **Priorité** | [Haute / Moyenne / Basse] |

**📌 Pré-requis :**
- [Pré-requis]

**🔢 Étapes de Reproduction :**
1. [Étape concrète et précise]
2. ...

**✅ Résultat Attendu :**
[Description claire et vérifiable]

**🔴 Résultat en cas d'échec :**
[Ce que verrait le testeur si le test échoue]

---

CONTRAINTES :
- Une étape = une action utilisateur ou système précise
- Résultat attendu toujours vérifiable (message, état, valeur…)
- Ne jamais inventer de règle non fournie"""

def call_gemini(messages, system_prompt, user_message, image=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_choice,
        system_instruction=system_prompt
    )
    history = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})
    chat = model.start_chat(history=history)
    parts = [user_message]
    if image:
        parts.append(image)
    resp = chat.send_message(parts)
    return resp.text

def render_chat(messages):
    for m in messages:
        avatar = "🧑‍💻" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

# ============================================================
# PHASE 1
# ============================================================
render_stepper()

if st.session_state.phase == 1:
    st.markdown('<div class="phase-badge phase-1">🔍 Phase 1 — Analyste : Clarifications</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        st.markdown("### Soumettez votre User Story")
        with st.form("us_form"):
            us_text = st.text_area(
                "User Story (+ Critères d'acceptation)",
                height=180,
                placeholder="En tant qu'utilisateur, je veux me connecter avec mon email et mot de passe afin d'accéder à mon espace personnel."
            )
            uploaded = st.file_uploader("📎 Maquette / Capture Figma (optionnel)", type=["png","jpg","jpeg","webp"])
            submitted = st.form_submit_button("🚀 Analyser", use_container_width=True)

        if submitted and us_text.strip():
            st.session_state.us_text = us_text
            st.session_state.us_submitted = True
            image_pil = Image.open(uploaded) if uploaded else None
            prompt = f"Voici la User Story à analyser :\n\n{us_text}"
            if uploaded:
                prompt += "\n\n[Une image de maquette a été fournie, analyse-la également.]"
            with st.spinner("Phase 1 en cours — Analyse de l'US…"):
                try:
                    response = call_gemini([], SYSTEM_PHASE1, prompt, image_pil)
                    st.session_state.phase1_messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                    st.session_state.phase1_context = f"US : {us_text}\n\nRéponse initiale :\n{response}"
                except Exception as e:
                    st.error(f"Erreur Gemini : {e}")
                    st.session_state.us_submitted = False
            st.rerun()
    else:
        render_chat(st.session_state.phase1_messages)

        if not st.session_state.phase1_validated:
            st.chat_input("Répondez aux questions de clarification…", key="p1_input")
            if st.session_state.get("p1_input"):
                user_msg = st.session_state.p1_input
                st.session_state.phase1_messages.append({"role": "user", "content": user_msg})
                with st.spinner("L'IA traite votre réponse…"):
                    try:
                        response = call_gemini(
                            st.session_state.phase1_messages[:-1],
                            SYSTEM_PHASE1,
                            user_msg
                        )
                        st.session_state.phase1_messages.append({"role": "assistant", "content": response})
                        # Mise à jour du contexte cumulé
                        st.session_state.phase1_context += f"\n\nQ: {user_msg}\nR: {response}"
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                st.rerun()

            st.divider()
            st.markdown('<div class="validation-box">', unsafe_allow_html=True)
            st.markdown("#### ✅ Validation Phase 1")
            st.caption("Vous avez répondu à toutes les questions ? Validez pour générer le plan de test.")
            if st.button("➡️ Valider l'analyse et passer à la Phase 2", type="primary", use_container_width=True):
                st.session_state.phase1_validated = True
                st.session_state.phase = 2
                # Générer automatiquement le brouillon Phase 2
                context_msg = f"""Voici le contexte complet validé par le testeur :

{st.session_state.phase1_context}

Génère maintenant le plan de test (titres uniquement)."""
                with st.spinner("Génération du plan de test Phase 2…"):
                    try:
                        response = call_gemini([], SYSTEM_PHASE2, context_msg)
                        st.session_state.phase2_messages = [
                            {"role": "user", "content": context_msg},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.phase2_draft = response
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PHASE 2
# ============================================================
elif st.session_state.phase == 2:
    st.markdown('<div class="phase-badge phase-2">📋 Phase 2 — QA Lead : Plan de Test</div>', unsafe_allow_html=True)

    render_chat(st.session_state.phase2_messages)

    if not st.session_state.phase2_validated:
        st.chat_input("Demandez des modifications au plan…", key="p2_input")
        if st.session_state.get("p2_input"):
            user_msg = st.session_state.p2_input
            st.session_state.phase2_messages.append({"role": "user", "content": user_msg})
            with st.spinner("Mise à jour du plan…"):
                try:
                    response = call_gemini(
                        st.session_state.phase2_messages[:-1],
                        SYSTEM_PHASE2,
                        user_msg
                    )
                    st.session_state.phase2_messages.append({"role": "assistant", "content": response})
                    st.session_state.phase2_draft = response
                except Exception as e:
                    st.error(f"Erreur : {e}")
            st.rerun()

        st.divider()
        st.markdown('<div class="validation-box">', unsafe_allow_html=True)
        st.markdown("#### ✅ Validation Phase 2")
        st.caption("Le plan de test vous convient ? Validez pour générer les cas de tests détaillés.")
        if st.button("➡️ Valider le plan et générer les cas détaillés (Phase 3)", type="primary", use_container_width=True):
            st.session_state.phase2_validated = True
            st.session_state.phase = 3
            plan_msg = f"""Voici le plan de test VALIDÉ par le testeur :

{st.session_state.phase2_draft}

Contexte de l'US :
{st.session_state.phase1_context}

Génère maintenant les cas de tests COMPLETS et DÉTAILLÉS pour chaque scénario."""
            with st.spinner("Génération des cas de tests détaillés Phase 3…"):
                try:
                    response = call_gemini([], SYSTEM_PHASE3, plan_msg)
                    st.session_state.phase3_messages = [
                        {"role": "user", "content": plan_msg},
                        {"role": "assistant", "content": response},
                    ]
                except Exception as e:
                    st.error(f"Erreur : {e}")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PHASE 3
# ============================================================
elif st.session_state.phase == 3:
    st.markdown('<div class="phase-badge phase-3">📝 Phase 3 — Expert : Cas de Tests Détaillés</div>', unsafe_allow_html=True)

    render_chat(st.session_state.phase3_messages)

    if st.session_state.phase3_messages:
        st.divider()
        col1, col2 = st.columns(2)
        all_content = "\n\n".join([m["content"] for m in st.session_state.phase3_messages if m["role"] == "assistant"])

        with col1:
            st.download_button(
                label="📥 Exporter en Markdown (.md)",
                data=all_content,
                file_name="cas_de_tests_QA.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            us_title = st.session_state.us_text[:40].replace(" ", "_") if st.session_state.us_text else "QA"
            st.download_button(
                label="📄 Exporter en Texte (.txt)",
                data=all_content,
                file_name=f"cas_tests_{us_title}.txt",
                mime="text/plain",
                use_container_width=True
            )

        st.chat_input("Demandez des ajustements ou des cas supplémentaires…", key="p3_input")
        if st.session_state.get("p3_input"):
            user_msg = st.session_state.p3_input
            st.session_state.phase3_messages.append({"role": "user", "content": user_msg})
            with st.spinner("Mise à jour des cas de tests…"):
                try:
                    response = call_gemini(
                        st.session_state.phase3_messages[:-1],
                        SYSTEM_PHASE3,
                        user_msg
                    )
                    st.session_state.phase3_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Erreur : {e}")
            st.rerun()
