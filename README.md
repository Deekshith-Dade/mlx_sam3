<div align="center">

# 🎯 MLX SAM3

**Segment Anything Model 3 — Native Apple Silicon Implementation**

[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-SAM%20License-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-MLX%20Community-yellow)](https://huggingface.co/mlx-community/sam3-image)

*A high-performance MLX port of Meta's SAM3 for interactive image segmentation on Mac*

<br>

[![Read the Blog](https://img.shields.io/badge/📖_Read_the_Blog-Understanding_SAM3-FF6B6B?style=for-the-badge)](https://deekshith.me/blog/mlx-sam3)

</div>

---

<table>
<tr>
<td>
<strong>📖 New to SAM3?</strong> Check out the accompanying blog post where I explain the SAM3 architecture, how it works, and what makes it special: <a href="https://deekshith.me/blog/mlx-sam3"><strong>Understanding SAM3 →</strong></a>
</td>
</tr>
</table>

---

## ✨ Features

- **🚀 Native Apple Silicon** — Optimized for M1/M2/M3/M4 chips using MLX
- **📝 Text Prompts** — Segment objects by describing them ("car", "person", "dog")
- **📦 Box Prompts** — Draw bounding boxes to include or exclude regions
- **🎨 Interactive Studio** — Beautiful web interface for real-time segmentation
- **🐍 Python API** — Simple programmatic access for scripting and integration
- **⬇️ Auto Model Download** — Weights automatically fetched from HuggingFace

---

## 🖼️ Demo

<div align="center">
<table>
<tr>
<td><img src="assets/images/appdemo.png" alt="SAM3 Demo - Car Detection" width="100%"></td>
<td><img src="assets/images/appdemo1.png" alt="SAM3 Demo - Coat Segmentation" width="100%"></td>
</tr>
<tr>
<td align="center"><em>Object detection with "car" prompt</em></td>
<td align="center"><em>Semantic segmentation with "coat" prompt</em></td>
</tr>
</table>

*SAM3 Studio — Interactive segmentation with text and box prompts*

</div>

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **macOS** | 13.0+ | Apple Silicon required (M1/M2/M3/M4) |
| **Python** | 3.13+ | Required for MLX compatibility |
| **Node.js** | 18+ | For the web interface |
| **uv** | Latest | *Optional but recommended* — [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |

> ⚠️ **Apple Silicon Only**: This project uses [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework optimized exclusively for Apple Silicon.

---

## 🚀 Quick Start

### Option 1: One-Command Launch (Recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed:

```bash
# Clone the repository
git clone https://github.com/Deekshith-Dade/mlx_sam3.git
cd mlx_sam3

# Install project dependencies
uv sync

# Launch the app (backend + frontend)
cd app && ./run.sh
```

The first run will automatically download MLX weights from [mlx-community/sam3-image](https://huggingface.co/mlx-community/sam3-image) (~3.5GB).

**Access the app:**
- 🌐 **Frontend**: http://localhost:3000
- 🔌 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

Press `Ctrl+C` to stop all servers.

---

### Option 2: Manual Setup (Standard pip)

<details>
<summary><strong>Click to expand manual setup instructions</strong></summary>

#### 1. Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/mlx_sam3.git
cd mlx_sam3

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

#### 2. Start the Backend

```bash
cd app/backend
pip install -r requirements.txt
python main.py
```

The backend will start on http://localhost:8000

#### 3. Start the Frontend (new terminal)

```bash
cd app/frontend
npm install
npm run dev
```

The frontend will start on http://localhost:3000

</details>

---

## 🐍 Python API

Use SAM3 directly in your Python scripts:

```python
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model (auto-downloads MLX weights from mlx-community/sam3-image)
model = build_sam3_image_model()
processor = Sam3Processor(model, confidence_threshold=0.5)

# Load and process an image
image = Image.open("your_image.jpg")
state = processor.set_image(image)

# Segment with text prompt
state = processor.set_text_prompt("person", state)

# Access results
masks = state["masks"]       # Binary segmentation masks
boxes = state["boxes"]       # Bounding boxes [x0, y0, x1, y1]
scores = state["scores"]     # Confidence scores

print(f"Found {len(scores)} objects")
```

### Adding Box Prompts

```python
# Add a box prompt (normalized coordinates: center_x, center_y, width, height)
# label=True for inclusion, label=False for exclusion
state = processor.add_geometric_prompt(
    box=[0.5, 0.5, 0.3, 0.3],  # Center of image, 30% width/height
    label=True,
    state=state
)
```

### Reset and Try New Prompts

```python
# Clear all prompts while keeping the image
processor.reset_all_prompts(state)

# Try a different prompt
state = processor.set_text_prompt("car", state)
```

---

## 🏗️ Project Structure

```
mlx_sam3/
├── sam3/                    # Core MLX SAM3 implementation
│   ├── model/               # Model components
│   │   ├── sam3_image.py    # Main model architecture
│   │   ├── vitdet.py        # Vision Transformer backbone
│   │   ├── text_encoder_ve.py # Text encoder
│   │   └── ...
│   ├── model_builder.py     # Model construction utilities
│   ├── convert.py           # Weight conversion from PyTorch
│   └── utils.py             # Helper utilities
├── app/                     # Web application
│   ├── backend/             # FastAPI server
│   ├── frontend/            # Next.js React app
│   └── run.sh               # One-command launcher
├── assets/                  # Static assets & test images
├── examples/                # Jupyter notebook examples
└── pyproject.toml           # Project configuration
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check if the model is loaded and ready |
| `/upload` | POST | Upload an image and create a session |
| `/segment/text` | POST | Segment using a text prompt |
| `/segment/box` | POST | Add a box prompt (include/exclude) |
| `/reset` | POST | Clear all prompts for a session |
| `/session/{id}` | DELETE | Delete a session and free memory |

### Example API Call

```bash
# Upload an image
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_image.jpg"

# Response: {"session_id": "abc-123", "width": 1920, "height": 1080, ...}

# Segment with text
curl -X POST "http://localhost:8000/segment/text" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "prompt": "car"}'
```

---

## 📓 Examples

Jupyter notebooks are available in the `examples/` directory:

- **`sam3_image_predictor_example.ipynb`** — Basic image segmentation
- **`sam3_image_interactive.ipynb`** — Interactive prompting workflows

Run them with:

```bash
cd examples
jupyter notebook
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | [MLX](https://github.com/ml-explore/mlx) |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4 |
| **Model** | [SAM3 MLX](https://huggingface.co/mlx-community/sam3-image) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is a derivative work of Meta's [SAM 3](https://github.com/facebookresearch/sam3) (both the ported code and the converted weights) and is therefore distributed under Meta's SAM License, as its terms require — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Meta AI](https://ai.meta.com/) for the original SAM3 model
- [MLX Team](https://github.com/ml-explore/mlx) at Apple for the incredible ML framework
- The open-source community for continuous inspiration

---

<div align="center">

**Built with ❤️ for Apple Silicon**

[Report Bug](https://github.com/Deekshith-Dade/mlx_sam3/issues) · [Request Feature](https://github.com/Deekshith-Dade/mlx_sam3/issues)

</div>

