import gradio as gr
import torch
from models import UNet
from datasets import tophat
import torchvision.transforms as T
from torchvision.utils import make_grid
import numpy as np
from PIL import Image, ImageChops, ImageOps
from utils import fusion_predict, make_mask, clear_mask
import cv2
from skimage import morphology

from latency import BUDGET_MS
from realtime import build_default_pipeline

SIZE = 512

tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img,50)),
    T.ToTensor(),
    T.Normalize((0.5),(0.5))
])

tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5),(0.5))
])


ckpts = ['ckpt/fscad_36249.ckpt'] # 828 FS-Model
#  ckpts = ['ckpt/xcad_4afe3.ckpt'] # XCAD-Model
#  ckpts = ['ckpt/pt_bc62a.ckpt'] # PT-Model

netE = UNet(1, 1, 32, bilinear=True)
checkpoint = torch.load(ckpts[0], map_location="cpu")
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
netE.load_state_dict(new_state_dict)
netE.to('cpu')

def predict(img, auto_tresh=True, options=None):
    if img is None:
        return None, None

    options = options or []

    if auto_tresh:
        if "Multiangle" in options:
            multiangle = True
        else:
            multiangle = False
        if "Pad margin" in options:
            pad = 50
        else:
            pad = 0

        img =  img.convert('L')
        x1 = tfmc1(img)
        x2 = tfmc2(img)
        _, out1 = fusion_predict(netE, ["none"], x1, multiangle=multiangle, denoise=4, size=SIZE, cutoff=0.4, pad=pad, netE=True)
        _, out2 = fusion_predict(netE, ["none"], x2, multiangle=False, denoise=4, size=SIZE, cutoff=0.4, pad=pad, netE=True)

        out_merge = Image.fromarray(np.expand_dims(np.max(np.concatenate((np.array(out1),np.array(out2)),axis=2),axis=2),2).repeat(3,2))

        mask_merge = make_mask(out_merge,remove_size=2000, local_kernel=21, hole_max_size=100)
        out_merge = T.functional.adjust_gamma(out_merge, 2)
        seg_img = clear_mask(mask_merge)

        sub_img = ImageChops.invert(out_merge)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2**(-0.5))
        sub_img = T.ToPILImage()(sub_img)
    else:
        img =  img.convert('L')
        x = tfmc1(img)
        input = x.unsqueeze(0)
        with torch.no_grad():
            pred_y = netE(input)
        
        # 处理减影图片
        sub_img = make_grid(pred_y,normalize=True)
        sub_img = (sub_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        sub_img = cv2.fastNlMeansDenoising(sub_img, None, 3, 7, 21)

        sub_img = T.ToPILImage()(sub_img)
        sub_img = ImageOps.autocontrast(sub_img, cutoff=1)
        sub_img = T.functional.adjust_gamma(sub_img, 2)

        sub_img = ImageChops.invert(sub_img)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2**(-0.5))
        sub_img = T.ToPILImage()(sub_img)
        
        
        # 处理分割图片
        seg_img = torch.sign(pred_y)
        seg_img = ((seg_img.cpu().detach() + 1)/2).numpy().astype(bool)
        seg_img = morphology.remove_small_objects(seg_img, 500)
        seg_img = (seg_img * 255).astype('uint8')
        seg_img = torch.from_numpy(seg_img/255)
        seg_img = T.ToPILImage()(seg_img[0])

    return sub_img, seg_img


# ----------------------------------------------------------------------------
# real-time engine (additive: predict() above is left exactly as it was)
# ----------------------------------------------------------------------------

ENGINE_DEEP = "DeepSA (original)"
ENGINE_REALTIME = f"Real-time (<= {BUDGET_MS:.0f} ms/frame)"

realtime_pipeline = build_default_pipeline()


def predict_realtime(img):
    """Background suppression + coronary enhancement within the latency budget.

    Returns the panels in the same order the UI expects: enhanced frame, vessel
    mask, comparison strip and a status line with the measured latency.
    """
    if img is None:
        return None, None, None, ""
    result = realtime_pipeline.process(img)
    status = (
        f"<div class='app-status'><b>{result.latency_ms:.1f} ms</b> / frame &middot; "
        f"{1000.0 / max(result.latency_ms, 1e-6):.0f} FPS &middot; "
        f"budget {result.budget_ms:.0f} ms &middot; "
        f"{'within budget' if result.within_budget else 'over budget'}"
        f"<br><span class='app-status__sub'>"
        f"{' &middot; '.join(f'{k} {v:.1f} ms' for k, v in sorted(result.stages_ms.items(), key=lambda kv: -kv[1]))}"
        f" &middot; {result.input_bit_depth}-bit in &rarr; {result.output_bit_depth}-bit out"
        f"</span></div>"
    )
    return result.enhanced, result.segmentation, result.comparison, status


def process_frame(img, engine=ENGINE_DEEP):
    """UI dispatcher: the original model, or the real-time pipeline."""
    if engine == ENGINE_REALTIME:
        enhanced, mask, comparison, status = predict_realtime(img)
        return enhanced, mask, comparison, status
    sub_img, seg_img = predict(img)
    return sub_img, seg_img, None, ""



# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

PANEL_HEIGHT = 380

LOGO_SVG = """
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M12 20.5s-7.2-4.4-7.2-9.6A4.4 4.4 0 0 1 12 7.6a4.4 4.4 0 0 1 7.2 3.3c0 5.2-7.2 9.6-7.2 9.6Z"/>
  <path d="M2.6 13.2h4.1l1.8-3.6 2.9 7.2 2.3-4.6 1.4 2.6h3.3"/>
</svg>
"""


def _panel_head(index, title, subtitle):
    return f"""
<div class="app-head">
  <span class="app-head__idx">{index}</span>
  <span class="app-head__text">
    <span class="app-head__title">{title}</span>
    <span class="app-head__sub">{subtitle}</span>
  </span>
</div>
"""

HERO_HTML = f"""
<div class="app-hero">
  <span class="app-hero__orb app-hero__orb--a"></span>
  <span class="app-hero__orb app-hero__orb--b"></span>

  <div class="app-hero__brand">
    <span class="app-logo">{LOGO_SVG}</span>
    <span class="app-hero__names">
      <span class="app-hero__title">Cardiology Coronary Artery Image Processing</span>
    </span>
  </div>


</div>
"""

CUSTOM_CSS = """
:root {
  --app-bg: #060a13;
  --app-panel: #0d1424;
  --app-panel-2: #0a1120;
  --app-line: rgba(148, 163, 184, 0.14);
  --app-line-strong: rgba(148, 163, 184, 0.28);
  --app-text: #e2e8f0;
  --app-muted: #94a3b8;
  --app-accent: #22d3ee;
  --app-accent-2: #818cf8;
}

.gradio-container {
  max-width: 1440px !important;
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif !important;
  background:
    radial-gradient(1100px 620px at 8% -12%, rgba(34, 211, 238, 0.13), transparent 62%),
    radial-gradient(900px 540px at 102% -6%, rgba(129, 140, 248, 0.14), transparent 58%),
    radial-gradient(760px 520px at 50% 118%, rgba(45, 212, 191, 0.08), transparent 62%),
    var(--app-bg) !important;
  color: var(--app-text);
}

.gradio-container h1, .gradio-container h2, .gradio-container h3 { letter-spacing: -0.02em; }
.gradio-container p, .gradio-container span, .gradio-container label { color: var(--app-text); }
.gradio-container code {
  font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  background: rgba(34, 211, 238, 0.09);
  border: 1px solid var(--app-line);
  border-radius: 6px;
  padding: 1px 6px;
  font-size: 0.82em;
  color: #7dd3fc;
}

/* ------------------------------------------------------------------ hero */
#hero { padding: 0 !important; background: transparent !important; border: 0 !important; }

.app-hero {
  position: relative;
  overflow: hidden;
  padding: 34px 38px 32px;
  border: 1px solid var(--app-line);
  border-radius: 20px;
  background:
    linear-gradient(135deg, rgba(13, 20, 36, 0.94), rgba(6, 10, 19, 0.96));
  box-shadow: 0 24px 60px -30px rgba(0, 0, 0, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.app-hero__orb { position: absolute; border-radius: 50%; filter: blur(58px); pointer-events: none; }
.app-hero__orb--a { width: 300px; height: 300px; top: -150px; left: -70px; background: rgba(34, 211, 238, 0.22); }
.app-hero__orb--b { width: 340px; height: 340px; top: -170px; right: -90px; background: rgba(129, 140, 248, 0.22); }

.app-hero__brand { position: relative; display: flex; align-items: center; gap: 16px; }

.app-logo {
  display: grid; place-items: center;
  width: 54px; height: 54px; flex: 0 0 54px;
  border-radius: 16px;
  color: #04202a;
  background: linear-gradient(140deg, var(--app-accent), var(--app-accent-2));
  box-shadow: 0 10px 26px -10px rgba(34, 211, 238, 0.75);
}
.app-logo svg { width: 30px; height: 30px; }

.app-hero__names { display: flex; flex-direction: column; }
.app-hero__title {
  font-size: 2.1rem; font-weight: 700; line-height: 1.1;
  background: linear-gradient(96deg, #f8fafc 8%, var(--app-accent) 96%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* ----------------------------------------------------------------- cards */
.app-card {
  background: linear-gradient(180deg, var(--app-panel), var(--app-panel-2)) !important;
  border: 1px solid var(--app-line) !important;
  border-radius: 18px !important;
  box-shadow: 0 18px 44px -28px rgba(0, 0, 0, 0.9), inset 0 1px 0 rgba(255, 255, 255, 0.035);
  padding: 16px 16px 14px !important;
  gap: 10px !important;
}
.app-card--quiet { background: rgba(13, 20, 36, 0.6) !important; }

.app-head { display: flex; align-items: center; gap: 12px; }
.app-head__idx {
  display: grid; place-items: center;
  width: 30px; height: 30px; flex: 0 0 30px;
  border-radius: 9px; font-size: 0.74rem; font-weight: 700;
  font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  color: var(--app-accent);
  background: rgba(34, 211, 238, 0.1);
  border: 1px solid rgba(34, 211, 238, 0.26);
}
.app-head__text { display: flex; flex-direction: column; gap: 1px; }
.app-head__title { font-size: 0.98rem; font-weight: 650; color: #f1f5f9; }
.app-head__sub { font-size: 0.76rem; color: var(--app-muted); }

.app-hint { font-size: 0.76rem; color: var(--app-muted); letter-spacing: 0.02em; }

.app-status {
  display: flex; flex-direction: column; gap: 3px;
  padding: 9px 12px; border-radius: 10px;
  font-size: 0.86rem;
  background: rgba(34, 211, 238, 0.07);
  border: 1px solid rgba(34, 211, 238, 0.22);
  color: var(--app-text);
}
.app-status b { color: var(--app-accent); font-size: 0.98rem; }
.app-status__sub { font-size: 0.72rem; color: var(--app-muted); }

/* ---------------------------------------------------------------- images */
.app-card .image-container,
.app-card .gr-image,
.app-card .image-upload-container {
  border-radius: 12px !important;
  border: 1px solid var(--app-line) !important;
  background: #050810 !important;
  overflow: hidden;
}
.app-card .gr-image img, .app-card .image-container img { background: #050810 !important; }
.app-card label > span { font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--app-muted); }

/* ----------------------------------------------------- responsive tweaks */
@media (max-width: 860px) {
  .app-hero { padding: 26px 22px 24px; border-radius: 16px; }
  .app-hero__title { font-size: 1.7rem; }
  .app-card { padding: 14px 12px 12px !important; }
}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
).set(
    body_background_fill="#060a13",
    body_background_fill_dark="#060a13",
    block_background_fill="#0d1424",
    block_background_fill_dark="#0d1424",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    body_text_color="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    input_background_fill="#050810",
    input_background_fill_dark="#050810",
    input_border_color="#1e293b",
    input_border_color_dark="#1e293b",
    input_shadow="0 0 0 1px rgba(148,163,184,0.14)",
    input_shadow_dark="0 0 0 1px rgba(148,163,184,0.14)",
)

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Cardiology Coronary Artery Image Processing") as demo:
    gr.HTML(HERO_HTML, elem_id="hero")

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, min_width=320):
            with gr.Column(variant="panel", elem_classes="app-card", elem_id="input-card"):
                gr.HTML(_panel_head("01", "Input angiogram", "Upload a single coronary angiogram frame"))
                input_img = gr.Image(
                    type="pil", source="upload", show_label=False, height=PANEL_HEIGHT, elem_id="input-img"
                )
                engine = gr.Radio(
                    [ENGINE_DEEP, ENGINE_REALTIME],
                    value=ENGINE_REALTIME,
                    label="Engine",
                    elem_id="engine",
                )
                gr.HTML(
                    "<span class='app-hint'>PNG &middot; JPG &middot; 8- or 16-bit &middot; "
                    "single coronary frame</span>"
                )
                out_status = gr.HTML(elem_id="status")

        with gr.Column(scale=8, min_width=460):
            with gr.Row(equal_height=True):
                with gr.Column(variant="panel", elem_classes="app-card", elem_id="sub-card"):
                    gr.HTML(_panel_head("02", "Deep subtraction", "Background removed, vessels enhanced"))
                    out_sub = gr.Image(
                        type="pil", show_label=False, height=PANEL_HEIGHT, elem_id="sub-out"
                    )

                with gr.Column(variant="panel", elem_classes="app-card", elem_id="seg-card"):
                    gr.HTML(_panel_head("03", "Vessel segmentation", "Binary vessel mask"))
                    out_seg = gr.Image(
                        type="pil", show_label=False, height=PANEL_HEIGHT, elem_id="seg-out"
                    )

            with gr.Column(variant="panel", elem_classes="app-card", elem_id="cmp-card"):
                gr.HTML(_panel_head("04", "Comparison", "Original | background suppressed | coronary enhanced"))
                out_cmp = gr.Image(
                    type="pil", show_label=False, elem_id="cmp-out"
                )

    input_img.change(
        fn=process_frame,
        inputs=[input_img, engine],
        outputs=[out_sub, out_seg, out_cmp, out_status],
        api_name="process",
    )
    engine.change(
        fn=process_frame,
        inputs=[input_img, engine],
        outputs=[out_sub, out_seg, out_cmp, out_status],
        api_name="switch_engine",
    )


if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
