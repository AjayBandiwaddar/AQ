import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.models.sr_unet import UNetSR, set_seed
from src.train import load_sr_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def plot_map(data, title, vmin=None, vmax=None, colorbar=True):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)

def compute_metrics(sr, hr):
    psnr = peak_signal_noise_ratio(hr, sr, data_range=hr.max() - hr.min())
    ssim = structural_similarity(hr, sr, data_range=hr.max() - hr.min())
    return psnr, ssim

def main():
    st.title("Super-Resolution Air Quality Map Demo")
    st.write("Upload a low-res air quality map or generate synthetic, run U-Net SR model, and visualize enhancement.")

    # Model loading
    ckpt_path = st.text_input("Checkpoint path", value="models/sr_unet_checkpoint.pth")
    in_channels = st.number_input("Input channels", min_value=1, value=3)
    device = "cpu"
    set_seed(42)
    try:
        model = load_sr_model(ckpt_path, in_channels=in_channels, out_channels=1)
        model.to(device)
        model.eval()
        st.success("SR U-Net model loaded.")
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return

    # Data input
    st.header("Step 1: Upload or Generate Low-Resolution Map")
    uploaded = st.file_uploader("Upload low-res .npy file (shape: [C,H,W]):", type=["npy"])
    gen_synth = st.button("Generate synthetic low-res/hi-res pair")
    if uploaded:
        low_res = np.load(uploaded)
        st.write(f"Loaded low-res shape: {low_res.shape}")
    elif gen_synth:
        # Generate synthetic data
        low_res = np.random.rand(in_channels, 16, 16)
        hi_res = np.random.rand(1, 32, 32)
        st.write("Synthetic maps generated.")
        st.session_state["hi_res"] = hi_res
    else:
        st.info("Upload a low-res map or generate synthetic example to proceed.")
        return

    # Model run
    st.header("Step 2: Run SR U-Net Model")
    if st.button("Run Super-Resolution"):
        with st.spinner("Enhancing map..."):
            inp = torch.tensor(low_res[None,...], dtype=torch.float32).to(device)
            sr_out = model(inp).detach().cpu().numpy()[0,0]
            st.success("SR output generated.")

            # Visualization
            st.header("Step 3: Visualization")
            plot_map(low_res[0], "Input Low-Res Map", vmin=low_res.min(), vmax=low_res.max())
            plot_map(sr_out, "SR Output Map", vmin=sr_out.min(), vmax=sr_out.max())

            # If hi_res available
            hi_res = st.session_state.get("hi_res")
            if hi_res is not None:
                plot_map(hi_res[0], "Ground Truth High-Res Map", vmin=hi_res.min(), vmax=hi_res.max())
                psnr, ssim = compute_metrics(sr_out, hi_res[0])
                st.write(f"**PSNR:** {psnr:.2f}")
                st.write(f"**SSIM:** {ssim:.3f}")
            else:
                st.info("Upload or generate ground truth high-res for metrics.")
                st.write("To compute metrics, upload high-res .npy file below:")
                hr_up = st.file_uploader("Upload high-res .npy file (shape: [1,h,w]):", type=["npy"], key="hr_up")
                if hr_up:
                    hi_res = np.load(hr_up)
                    plot_map(hi_res[0], "Ground Truth High-Res Map", vmin=hi_res.min(), vmax=hi_res.max())
                    psnr, ssim = compute_metrics(sr_out, hi_res[0])
                    st.write(f"**PSNR:** {psnr:.2f}")
                    st.write(f"**SSIM:** {ssim:.3f}")

if __name__ == "__main__":
    main()