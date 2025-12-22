# Project Diagrams & Figures

This folder contains comprehensive visual documentation of the MDLM-ATAT project architecture, data flow, and component interactions.

## 📊 Available Diagrams

### 1. System Architecture (`01_system_architecture.drawio`)
**Overview**: High-level system architecture showing how ATAT extends base MDLM

**Contents**:
- Base MDLM Framework (blue)
  - Diffusion processes (SUBS/D3PM/SEDD)
  - Noise schedules
  - DiT architecture
  - DataLoader
  - Samplers
  - Training loop
- ATAT Extension (green)
  - Importance Estimator
  - Adaptive Masking
  - Curriculum Learning
  - Uncertainty Sampler
  - ATAT-DiT integration
- External Systems (yellow/gray)
  - Data storage (/media/scratch/)
  - WandB logging
  - Checkpoints
- Relationships and data flow

**Use Case**: Understanding overall project structure, explaining to collaborators

---

### 2. Training Flow (`02_training_flow.drawio`)
**Overview**: Complete training pipeline from start to checkpoint

**Contents**:
- Initialization phase
  - Load dataset
  - Initialize models
- Training loop
  - Batch processing
  - Component execution order:
    1. Importance Estimation
    2. Curriculum Stage determination
    3. Adaptive Masking
    4. Forward Diffusion
    5. Model Forward Pass
    6. Loss Computation
    7. Backward Pass
    8. Metrics Logging
- Checkpointing logic
- Loop continuation/termination

**Use Case**: Debugging training, understanding execution order, onboarding new developers

---

### 3. Component Details (`03_component_details.drawio`)
**Overview**: Deep dive into each of the 4 ATAT components

**Contents**:

#### Component 1: Importance Estimator
- Input: Token embeddings + timestep
- Architecture:
  - Time embedding layer
  - 2 Transformer layers (4 heads, 256 dim)
  - Importance head (Linear + Sigmoid)
- Output: Importance scores [0,1]
- Parameters: ~2M

#### Component 2: Adaptive Masking
- Input: Importance scores + timestep + stage
- Process:
  - Temperature scaling based on curriculum stage
  - Softmax over importance
  - Noise schedule application
  - Bernoulli sampling
- Output: Binary masks

#### Component 3: Curriculum Learning
- Input: Training step
- Process:
  - Stage determination (easy/medium/hard)
  - Weight computation (2.0 for in-range, 1.0 otherwise)
  - Weighted batch sampling
  - Loss weight adjustment
- Output: Stage, weights

#### Component 4: Uncertainty-Guided Sampler
- Input: Masked sequence
- Process:
  - Model prediction
  - Entropy calculation
  - Top-k uncertain token selection
  - Selective denoising
  - Convergence check
  - Loop or terminate
- Output: Generated text
- Benefit: 30% faster inference

**Use Case**: Implementation details, understanding component internals, paper writing

---

### 4. Data Flow (`04_data_flow.drawio`)
**Overview**: End-to-end data flow from storage through training to inference

**Contents**:

#### Data Storage Section
- `/media/scratch/adele/` breakdown:
  - mdlm_data_cache/ (47GB)
  - mdlm_fresh/ (321GB)
  - datasets/ (50GB)
- WandB cloud sync

#### Data Processing
- Download → Tokenize → Cache → Batch → Augment → Ready

#### Training
- Batch → 4 ATAT components → DiT → Loss → Backward → Log/Save

#### Inference
- Load checkpoint → Initialize → Predict → Uncertainty → Denoise → Output

#### Results
- Checkpoints (.ckpt)
- Training logs (CSV, TensorBoard)
- WandB dashboard
- Generated samples
- Evaluation metrics (JSON)
- Visualizations (GIFs, plots)

**Use Case**: Understanding data lifecycle, debugging I/O issues, storage planning

---

### 5. File Structure (`05_file_structure.drawio`)
**Overview**: Complete project directory tree and file organization

**Contents**:

#### Main Folders
- **mdlm/** (Base MDLM - blue)
  - Core files: main.py, diffusion.py, etc.
  - models/ subfolder
  - configs/ subfolder
  - scripts/ subfolder

- **mdlm_atat/** (ATAT Extension - green)
  - atat/ (4 core components)
  - models/ (atat_dit.py)
  - configs/ (tiny.yaml, small.yaml)
  - scripts/ (train, eval, ablation)
  - tests/

- **docs/** (Documentation - yellow)
  - Top-level guides
  - reports/ (9 reports)
  - figures/ (5 diagrams)
  - research_proposal.tex

#### External Storage
- Cylinder icons showing scratch storage
- Size breakdown (418GB total)

#### Summary Statistics
- Module count: 4
- Script count: 8
- Config count: 10+
- Doc count: 15+
- Total LOC: ~3,500

**Use Case**: Project navigation, understanding organization, newcomer orientation

---

## 🎨 How to View/Edit

### Viewing
1. **Draw.io Desktop App** (Recommended)
   ```bash
   # Install draw.io desktop
   # Ubuntu/Debian:
   wget https://github.com/jgraph/drawio-desktop/releases/download/v21.6.5/drawio-amd64-21.6.5.deb
   sudo dpkg -i drawio-amd64-21.6.5.deb
   
   # Open diagram
   drawio 01_system_architecture.drawio
   ```

2. **Online** (draw.io web)
   - Go to https://app.diagrams.net/
   - File → Open from → Device
   - Select any `.drawio` file

3. **VS Code Extension**
   ```bash
   # Install Draw.io Integration extension
   code --install-extension hediet.vscode-drawio
   
   # Then open .drawio files directly in VS Code
   ```

### Editing
1. Open file in draw.io (desktop or web)
2. Make changes
3. File → Save
4. Commit changes to git

### Exporting
To generate PNG/SVG/PDF exports:

```bash
# Using draw.io desktop
drawio --export --format png --output 01_system_architecture.png 01_system_architecture.drawio

# Or in the app:
# File → Export as → PNG/SVG/PDF
```

## 📝 Diagram Conventions

### Color Coding
- **Blue (#dae8fc)**: Base MDLM components
- **Green (#d5e8d4)**: ATAT extension/outputs
- **Purple (#e1d5e7)**: ATAT core components (4 modules)
- **Red/Pink (#f8cecc)**: Final outputs/important results
- **Yellow (#fff2cc)**: Configuration/data/temporary
- **Orange (#ffe6cc)**: External storage/systems
- **Gray (#f5f5f5)**: Neutral/shared components

### Shape Meanings
- **Rectangle**: Process/Component
- **Rounded Rectangle**: Container/Group
- **Cylinder**: Database/Storage
- **Diamond**: Decision point
- **Cloud**: External service (WandB)
- **Note**: Documentation/File
- **Parallelogram**: Input/Output data
- **Document**: Generated output

### Arrow Types
- **Solid thick (3px)**: Primary data flow
- **Solid thin (2px)**: Secondary flow
- **Dashed**: Optional/conditional flow
- **Dotted**: Reference/connection

## 🔄 Updating Diagrams

When to update:
1. **Architecture changes**: New components, removed modules
2. **Process changes**: Training flow modifications
3. **File structure changes**: New folders, reorganization
4. **Data flow changes**: New storage locations, pipeline updates

Update checklist:
- [ ] Update relevant diagram(s)
- [ ] Export new PNG versions (for presentations)
- [ ] Update this README if diagram purpose changed
- [ ] Commit with descriptive message: "docs: update [diagram] for [reason]"

## 📊 Quick Reference

| Diagram | Primary Use | Detail Level | Best For |
|---------|-------------|--------------|----------|
| 01_System_Architecture | Overview | High-level | Presentations, proposals |
| 02_Training_Flow | Training | Medium | Debugging, implementation |
| 03_Component_Details | Components | Deep-dive | Paper writing, understanding |
| 04_Data_Flow | Data | Medium | I/O debugging, optimization |
| 05_File_Structure | Organization | File-level | Navigation, newcomers |

## 🎯 Common Use Cases

### For Presentations
Best diagrams: #1 (System Architecture), #4 (Data Flow)
- Export as PNG/SVG
- High-level overview
- Clear visual impact

### For Paper Writing
Best diagrams: #3 (Component Details), #2 (Training Flow)
- Include in research proposal
- Detailed technical content
- Shows innovation clearly

### For Onboarding
Best diagrams: #5 (File Structure), #1 (System Architecture)
- Start with overview
- Then dive into file organization
- Helps newcomers navigate

### For Debugging
Best diagrams: #2 (Training Flow), #4 (Data Flow)
- Trace execution path
- Identify bottlenecks
- Understand data pipeline

## 🔗 Related Documentation

- [RESTRUCTURING_GUIDE.md](../RESTRUCTURING_GUIDE.md) - Project organization text guide
- [TECHNICAL_REPORT.md](../reports/TECHNICAL_REPORT.md) - Detailed technical docs
- [DATA_PATHS.md](../DATA_PATHS.md) - Storage and dataset information
- [research_proposal.tex](../research_proposal.tex) - Full CVPR paper

## 📌 Notes

- All diagrams created December 3, 2024
- Diagrams reflect post-restructuring organization
- Keep diagrams in sync with code changes
- Use draw.io for consistency
- Export PNGs for presentations

---

**Last Updated**: December 3, 2024  
**Maintainer**: Adele Chinda  
**Format**: Draw.io (.drawio XML format)  
**Count**: 5 comprehensive diagrams
