🌱 PlotPheno

An integrated UAV-based software system for automated wheat breeding plot
segmentation and multi-temporal dynamic phenotyping.

Python PyTorch License

PlotPheno is a lightweight, desktop-level graphical software designed for
agricultural breeders and remote sensing researchers. It provides an end-to-end,
GIS-free solution from raw UAV imagery to variety-level growth kinetics. Powered
by the high-performance ST-PRNet instance segmentation network, PlotPheno
seamlessly handles dense plot arrangements, severe canopy overlapping, and
complex field backgrounds.

✨ Key Features

  - 🎯 AI-Driven Plot Segmentation: Utilizes ST-PRNet (Swin Transformer +
    PointRend) for pixel-level boundary refinement. Automatically clusters and
    assigns standard Row-Col indices to each plot.
  - 🌍 Multi-Sensor Phenotyping: Automatically crops and aligns multi-source
    remote sensing data (RGB, Multispectral, LiDAR CHM) using geometric plot
    boundaries.
  - 🌿 Pure Canopy Extraction: Implements an adaptive Excess Green (EXG)
    background filtering mechanism to eliminate soil and weed interference,
    ensuring pure vegetation-derived traits.
  - 📊 Dynamic Spectral Calculation: Supports real-time, multi-band mathematical
    matrix calculations (e.g., automatically computing SAVI/EVI from raw NIR,
    Red, and Blue reflectance layers).
  - 📈 Time-Series Kinetics: Merges multi-epoch phenotypic data with 2D planting
    grid matrices to reconstruct variety-level growth kinetics (e.g., height
    growth velocity V_{PH}, senescence slope V_{NDRE}).
  - 🗺️ Automated Heatmap Rendering: Automatically generates 4x high-resolution
    spatial heatmaps and 2D grid Excel reports synchronized with the base image
    name (e.g., Filling_PH.png, Filling_PH.xlsx).

🛠️ Installation & Setup

1. Environment Requirements

  - OS: Windows 10 / Windows 11 (64-bit)
  - Hardware: Intel Core i7 or higher, 16 GB RAM minimum. An NVIDIA GPU with
    CUDA support (e.g., RTX 3060 8GB+) is strongly recommended for fast
    deep-learning inference.
  - Dependencies: Python 3.8, CUDA 11.7.

2. Clone Repository & Install Packages

git clone https://github.com/maikanao666/PlotPheno.git
cd PlotPheno
pip install -r requirements.txt

3. Download Model Weights

Due to GitHub's file size limits, the pre-trained ST-PRNet weights are not
included in this repository. Please download the weight file (weight.pth) from
the link below and place it in the models/ directory:

- **Official Link (For Global Reviewers):** https://github.com/maikanao666/PlotPheno/releases/tag/V1.0

Project structure should look like this:

PlotPheno/
├── models/
│   ├── st_prnet.py
│   └── weight.pth      <--- Put the downloaded weight here!
├── core_engine.py
├── run_app.py
├── ui_main.py
...

🚀 How to Use

Run the following command in your terminal to start the GUI:

python run_app.py

Note: You can switch the interface language between English and Chinese via the
Language menu.

Step 1: Plot Segmentation

1.  Go to File -> Import Image to load your base UAV RGB orthomosaic (e.g.,
    Jointing.tif).
2.  Adjust Processing Parameters in the Settings menu (e.g., Slice Size,
    Confidence Threshold, Rectification Angle).
3.  Double-click Plot Segmentation (Seg) in the right panel to bind the base
    image, then click the 1. Plot Segmentation button. The system will generate
    masks and an overlay image with automated Row-Col indexing.

Step 2: Phenotype Extraction

1.  Check the desired phenotypic traits (e.g., FVC, Area, NDVI, SAVI, PH,
    Volume).
2.  Double-click the checked traits to bind them to their corresponding
    multispectral or LiDAR layers. (Tip: For dynamic indices like SAVI, the
    system allows you to bind multiple raw bands like NIR and Red
    simultaneously).
3.  Click the 2. Phenotype Extraction button. The system will prompt you to
    select the RGB base map for adaptive background removal. Once finished,
    spatial heatmaps and structured Excel grids will be automatically exported.

Step 3: Time-Series Dynamics Analysis

1.  Click the 3. TimeSeries button to open the kinetics analysis panel.
2.  Import a 2D Excel planting grid matrix containing the variety (pedigree)
    information.
3.  Import the previously generated phenotypic Excel sheets for multiple growth
    stages (e.g., Jointing, Booting, Filling) and input the interval days.
4.  Click Run Kinetics Solver. The system will aggregate multi-replicate data,
    calculate dynamic physiological velocities, and plot trajectory curves for
    each variety.

📅 Roadmap / Future Works

- [√] V1.0: End-to-end segmentation, multi-modal extraction, and dynamic
  kinetics.
- [ ] V2.0: Interactive Edge Modification Module (Currently under development in
  the Tools menu). Will allow users to manually drag polygon vertices and draw
  split lines to decouple severely adhered plots without model retraining.
- [ ] V2.0: Extension to Thermal Infrared (TIR) and Hyperspectral imagery for
  Crop Water Stress Index (CWSI) and biochemical trait retrievals.

📜 Citation

If you find this software or our dataset useful for your research, please
consider citing our paper (Currently under review at Computers and Electronics
in Agriculture):

@article{PlotPheno2025,
  title={PlotPheno: An integrated system for wheat breeding plot segmentation and phenotyping from UAV imagery},
  author={},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  note={Under Review}
}

📧 Contact

For any questions, bug reports, or collaborations, please open an issue or
contact: [Yijing Liang] - [maikanao@163.com]
