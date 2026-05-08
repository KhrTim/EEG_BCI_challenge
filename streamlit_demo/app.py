# streamlit_demo/app.py
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_DEMO = os.path.dirname(os.path.abspath(__file__))
_EEGCONFORMER = os.path.join(_DEMO, "..", "eegconformer")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)
if _EEGCONFORMER not in sys.path:
    sys.path.insert(0, _EEGCONFORMER)

from inference import PATCH_CENTERS_MS  # noqa: E402

_DATA_DIR = os.path.join(_DEMO, "data")
_DEFAULT_CKPT = os.path.join(_DATA_DIR, "fold_rep1_fold1.pt")
_DEFAULT_CSVS = sorted(
    os.path.join(_DATA_DIR, f)
    for f in os.listdir(_DATA_DIR)
    if f.startswith("Data_") and f.endswith(".csv")
) if os.path.isdir(_DATA_DIR) else []
_DEFAULT_LABELS = os.path.join(_DATA_DIR, "TrainLabels.csv")

st.set_page_config(
    page_title="Детектор ошибок ЭЭГ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _init_state():
    defaults = {
        "sessions": {},
        "subjects": [],
        "ch_names": [f"CH{i:02d}" for i in range(1, 57)],
        "model": None,
        "label_dict": None,
        "epoch_idx": 0,
        "nav_subject": None,
        "nav_session": None,
        "last_result": None,
        "ch_select": [],
        "_sal_epoch_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _eeg_figure(epoch: np.ndarray, ch_names: list, selected_ch: list,
                peak_i: int | None = None) -> go.Figure:
    """epoch: (56, 260), selected_ch: list of channel name strings"""
    t_ms = np.linspace(0, 1300, 260)
    fig = go.Figure()
    for ch in selected_ch:
        idx = ch_names.index(ch)
        fig.add_trace(go.Scatter(
            x=t_ms, y=epoch[idx], mode="lines", name=ch, line=dict(width=1)
        ))
    if peak_i is not None:
        x0 = peak_i * 15 / 200 * 1000
        x1 = (peak_i * 15 + 75) / 200 * 1000
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(243,139,168,0.15)", line_width=0,
            annotation_text="пик внимания",
            annotation_position="top left",
            annotation_font_color="#f38ba8",
            annotation_font_size=9,
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        height=220,
        xaxis_title="мс",
        yaxis_title="мкВ",
        legend=dict(orientation="h", y=-0.3),
        template="plotly_dark",
        showlegend=True,
        transition=dict(duration=300, easing="cubic-in-out"),
    )
    fig.update_xaxes(range=[0, 1300])
    return fig


def _channel_importance_figure(importance: np.ndarray, ch_names: list) -> go.Figure:
    """importance: (56,) normalized channel saliency.
    Channels rendered in FIXED original order so Plotly can animate bar lengths."""
    n = len(ch_names)
    max_val = float(importance.max()) + 1e-8
    colors = [f"rgba(137,180,250,{0.35 + 0.65 * float(v) / max_val:.2f})" for v in importance]
    fig = go.Figure(go.Bar(
        x=importance.tolist(),
        y=list(ch_names),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in importance],
        textposition="outside",
        textfont=dict(size=7),
    ))
    fig.update_layout(
        margin=dict(l=0, r=40, t=4, b=0),
        height=n * 12 + 20,  # fixed by channel count, not by importance rank
        xaxis=dict(showticklabels=False, range=[0, max_val * 1.25], fixedrange=True),
        yaxis=dict(autorange="reversed", fixedrange=True),
        template="plotly_dark",
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    return fig


def _attn_figure(attn: np.ndarray) -> go.Figure:
    """attn: (13,) normalized attention weights"""
    peak_i = int(np.argmax(attn))
    colors = ["#f38ba8" if i == peak_i else "#89b4fa" for i in range(13)]
    tick_ms = [f"{int(ms)}" for ms in PATCH_CENTERS_MS]

    fig = go.Figure(go.Bar(
        x=list(range(13)),
        y=attn.tolist(),
        marker_color=colors,
        text=[f"{v:.2f}" for v in attn],
        textposition="outside",
        textfont=dict(size=8),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=40),
        height=200,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(13)),
            ticktext=tick_ms,
            title="мс",
        ),
        yaxis=dict(showticklabels=False),
        template="plotly_dark",
        transition=dict(duration=300, easing="cubic-in-out"),
    )
    return fig


# ── page ─────────────────────────────────────────────────────────────────────

_init_state()

st.markdown("""
<style>
@keyframes _fadeIn {
    from { opacity: 0.25; }
    to   { opacity: 1;    }
}
@keyframes _fadeSlide {
    from { opacity: 0.2; transform: translateY(4px); }
    to   { opacity: 1;   transform: translateY(0);   }
}
/* plotly chart containers — fade in on update */
[data-testid="stPlotlyChart"]          { animation: _fadeIn    0.35s ease-out; }
/* text elements that change every epoch */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stCaptionContainer"]  p  { animation: _fadeSlide 0.3s  ease-out; }
/* progress bar fill */
[data-testid="stProgressBar"] > div    { transition: width 0.4s cubic-bezier(.4,0,.2,1); }
/* alert / success / error badges */
[data-testid="stAlert"]                { animation: _fadeSlide 0.3s  ease-out; }
</style>
""", unsafe_allow_html=True)

# ── auto-load default data on first run ──────────────────────────────────────
if not st.session_state.sessions and os.path.isfile(_DEFAULT_CKPT) and _DEFAULT_CSVS:
    from processing import epoch_csv_files, apply_ea_to_sessions, build_label_dict
    from inference import load_model

    class _PathFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self._path = path
        def read(self):
            with open(self._path, "rb") as f:
                return f.read()

    with st.spinner("Загрузка данных по умолчанию..."):
        sessions, subjects, ch_names = epoch_csv_files([_PathFile(p) for p in _DEFAULT_CSVS])
        sessions = apply_ea_to_sessions(sessions)
        model = load_model(_DEFAULT_CKPT)
        label_dict = build_label_dict(_PathFile(_DEFAULT_LABELS) if os.path.isfile(_DEFAULT_LABELS) else None)

    st.session_state.sessions = sessions
    st.session_state.subjects = subjects
    st.session_state.ch_names = ch_names
    st.session_state.model = model
    st.session_state.label_dict = label_dict
    st.session_state.epoch_idx = 0
    st.session_state.nav_subject = subjects[0] if subjects else None
    st.session_state.nav_session = (
        sorted(k[1] for k in sessions if k[0] == subjects[0])[0] if subjects else None
    )
    st.session_state.last_result = None
    st.session_state._sal_epoch_key = None
    st.rerun()

# Top banner
st.info(
    "ℹ️ **Что это такое?** Человек управляет компьютером через нейроинтерфейс (BCI): "
    "на экране мигают буквы, система угадывает нужную. Когда система выбирает "
    "**неправильную букву**, мозг генерирует характерный сигнал — потенциал, "
    "связанный с ошибкой (ErrP). Эта модель обнаруживает ErrP по сигналу ЭЭГ."
)
col_a, col_b = st.columns(2)
col_a.success("✓  ВЕРНО — система выбрала правильную букву")
col_b.error("✗  ОШИБКА — система выбрала неправильную букву → ErrP")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Детектор ErrP")

    st.subheader("Модель")
    model_file = st.file_uploader(
        "Checkpoint (.pt)", type=["pt", "pth"], label_visibility="collapsed"
    )

    st.subheader("Данные (CSV)")
    csv_files = st.file_uploader(
        "CSV файлы (Data_S##_Sess##.csv)",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.subheader("Метки (необязательно)")
    labels_file = st.file_uploader(
        "TrainLabels.csv", type=["csv"], label_visibility="collapsed"
    )

    # Navigation — only when data is loaded
    if st.session_state.sessions:
        st.divider()
        st.subheader("Навигация")
        subjects = st.session_state.subjects
        st.selectbox(
            "Испытуемый",
            subjects,
            format_func=lambda s: f"S{s:02d}",
            key="nav_subject",
        )
        avail_sessions = sorted(
            k[1] for k in st.session_state.sessions if k[0] == st.session_state.nav_subject
        )
        # Sync stale nav_session when subject changes
        if avail_sessions and st.session_state.nav_session not in avail_sessions:
            st.session_state.nav_session = avail_sessions[0]
            st.session_state.epoch_idx = 0
            st.session_state.last_result = None
            st.rerun()
        st.selectbox(
            "Сессия",
            avail_sessions,
            format_func=lambda s: f"{s:02d}",
            key="nav_session",
        )

        key = (st.session_state.nav_subject, st.session_state.nav_session)
        if key in st.session_state.sessions:
            n_ep = len(st.session_state.sessions[key]["epochs"])
            ep_idx = max(0, min(st.session_state.epoch_idx, n_ep - 1))

            st.write(f"Эпоха: **{ep_idx + 1:03d}** / {n_ep:03d}")
            col_prev, col_next = st.columns(2)
            if col_prev.button("◀ Пред.", use_container_width=True):
                st.session_state.epoch_idx = max(0, ep_idx - 1)
                st.session_state.last_result = None
                st.rerun()
            if col_next.button("След. ▶", use_container_width=True):
                st.session_state.epoch_idx = min(n_ep - 1, ep_idx + 1)
                st.session_state.last_result = None
                st.rerun()

    st.divider()
    run_clicked = st.button(
        "▶ Запустить",
        type="primary",
        use_container_width=True,
        disabled=not (model_file and csv_files) and not st.session_state.sessions,
    )

# ── process on Run ────────────────────────────────────────────────────────────
if run_clicked:
    from processing import epoch_csv_files, apply_ea_to_sessions, build_label_dict
    from inference import load_model

    with st.spinner("Загрузка и обработка данных..."):
        sessions, subjects, ch_names = epoch_csv_files(csv_files)
        sessions = apply_ea_to_sessions(sessions)
        model = load_model(model_file)
        label_dict = build_label_dict(labels_file)

    st.session_state.sessions = sessions
    st.session_state.subjects = subjects
    st.session_state.ch_names = ch_names
    st.session_state.model = model
    st.session_state.label_dict = label_dict
    st.session_state.epoch_idx = 0
    st.session_state.nav_subject = subjects[0] if subjects else None
    st.session_state.nav_session = (
        sorted(k[1] for k in sessions if k[0] == subjects[0])[0]
        if subjects else None
    )
    st.session_state.last_result = None
    st.session_state._sal_epoch_key = None
    st.rerun()

# ── main canvas ───────────────────────────────────────────────────────────────
if not st.session_state.sessions:
    st.markdown(
        "### Загрузите модель и CSV файлы в боковой панели, затем нажмите **▶ Запустить**"
    )
    st.stop()

subj = st.session_state.nav_subject
sess = st.session_state.nav_session
key = (subj, sess)

if key not in st.session_state.sessions:
    st.warning("Нет данных для выбранной сессии.")
    st.stop()

# Reset epoch and cached result when subject/session changes
if st.session_state.get("_last_nav_key") != key:
    st.session_state.epoch_idx = 0
    st.session_state.last_result = None
    st.session_state["_last_nav_key"] = key
    st.rerun()

ep_data = st.session_state.sessions[key]
n_ep = len(ep_data["epochs"])
ep_idx = max(0, min(st.session_state.epoch_idx, n_ep - 1))
epoch = ep_data["epochs"][ep_idx]      # (56, 260)
fb_id = ep_data["fb_ids"][ep_idx]
ch_names = st.session_state.ch_names

# Run inference first so peak_i and saliency are available for the EEG chart
if st.session_state.last_result is None and st.session_state.model is not None:
    from inference import run_inference, run_saliency
    with st.spinner("Инференс..."):
        prob, attn = run_inference(st.session_state.model, epoch)
        sal = run_saliency(st.session_state.model, epoch)
    st.session_state.last_result = (prob, attn, sal)

_peak_i = int(st.session_state.last_result[1].argmax()) if st.session_state.last_result is not None else None

# Auto-select top-4 channels by saliency each time the epoch changes
_sal_key = (subj, sess, ep_idx)
if st.session_state.last_result is not None and st.session_state._sal_epoch_key != _sal_key:
    _sal = st.session_state.last_result[2]
    top4 = [ch_names[i] for i in _sal.argsort()[::-1][:4]]
    st.session_state.ch_select = top4
    st.session_state._sal_epoch_key = _sal_key
elif not st.session_state.ch_select and ch_names:
    st.session_state.ch_select = ch_names[:4]

# EEG waveform
st.subheader(f"Сигнал ЭЭГ — Эпоха {ep_idx + 1:03d}  ({fb_id})")
selected_ch = st.multiselect(
    "Каналы",
    ch_names,
    key="ch_select",
    label_visibility="collapsed",
)
if selected_ch:
    st.plotly_chart(
        _eeg_figure(epoch, ch_names, selected_ch, peak_i=_peak_i),
        use_container_width=True,
        key="eeg_chart",
    )
    st.caption(f"полосовой фильтр 1–40 Гц · 200 Гц · {len(selected_ch)} из {len(ch_names)} каналов")

st.divider()

if st.session_state.last_result is None:
    st.info("Загрузите модель и нажмите ▶ Запустить для отображения предсказания.")
    st.stop()

prob, attn, sal = st.session_state.last_result
pred_label = "ОШИБКА" if prob >= 0.5 else "ВЕРНО"
peak_ms = int(PATCH_CENTERS_MS[int(attn.argmax())])

col_attn, col_pred = st.columns([1, 1])

with col_attn:
    st.subheader("Карта внимания")
    st.plotly_chart(_attn_figure(attn), use_container_width=True, key="attn_chart")
    st.caption(
        f"13 временных патчей · пик внимания ~{peak_ms} мс после обратной связи"
    )

with col_pred:
    st.subheader("Предсказание модели")

    if pred_label == "ОШИБКА":
        st.markdown(
            "<h1 style='color:#f38ba8;text-align:center'>ОШИБКА</h1>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h1 style='color:#a6e3a1;text-align:center'>ВЕРНО</h1>",
            unsafe_allow_html=True,
        )

    prob_pct = prob if prob >= 0.5 else 1.0 - prob
    st.markdown(
        f"<h2 style='text-align:center'>{prob_pct * 100:.0f}%</h2>",
        unsafe_allow_html=True,
    )
    st.progress(float(prob_pct))

    if pred_label == "ОШИБКА":
        st.caption("неправильная буква → мозг сгенерировал ErrP")
    else:
        st.caption("правильная буква, ErrP отсутствует")

    # Ground truth
    label_dict = st.session_state.label_dict
    if label_dict is not None:
        true_label = label_dict.get(fb_id)
        if true_label is not None:
            st.divider()
            true_sym = "✗" if true_label == 1 else "✓"
            true_name = "ОШИБКА" if true_label == 1 else "ВЕРНО"
            correct = (true_label == 1) == (prob >= 0.5)
            verdict = "← верно" if correct else "← ошибка модели"
            if correct:
                st.success(f"Истинная метка: {true_sym} {true_name} {verdict}")
            else:
                st.error(f"Истинная метка: {true_sym} {true_name} {verdict}")
        else:
            st.caption("метка недоступна")

st.divider()
st.subheader("Важность каналов")
st.plotly_chart(
    _channel_importance_figure(sal, ch_names),
    use_container_width=True,
    key="ch_chart",
)
st.caption("градиентная значимость · top-20 каналов по вкладу в предсказание")
