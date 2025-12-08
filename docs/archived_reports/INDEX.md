# Documentation Index

Welcome to the MDLM-ATAT documentation hub. All project documentation is organized here.

## 📋 Quick Links

### Getting Started
- [**GETTING_STARTED.md**](reports/GETTING_STARTED.md) - First-time setup and running your first experiment
- [**QUICK_REFERENCE.md**](reports/QUICK_REFERENCE.md) - Command cheat sheet and common operations
- [**RESTRUCTURING_GUIDE.md**](RESTRUCTURING_GUIDE.md) - **NEW!** Project organization and file structure

### Project Overview
- [**INDEX.md**](reports/INDEX.md) - Original documentation index
- [**PROJECT_SUMMARY.md**](reports/PROJECT_SUMMARY.md) - High-level project overview
- [**EXECUTIVE_SUMMARY.md**](reports/EXECUTIVE_SUMMARY.md) - Executive summary for stakeholders

### Technical Documentation
- [**TECHNICAL_REPORT.md**](reports/TECHNICAL_REPORT.md) - Detailed technical implementation
- [**research_proposal.tex**](research_proposal.tex) - CVPR 2026 paper draft (LaTeX)

### Visualization & Demos
- [**GIF_QUICK_START.md**](reports/GIF_QUICK_START.md) - Creating sampling animations
- [**GIF_VISUALIZATION_README.md**](reports/GIF_VISUALIZATION_README.md) - Visualization tools guide

### Presentations
- [**PRESENTATION_SLIDES.md**](reports/PRESENTATION_SLIDES.md) - Presentation materials

## 🗂️ Documentation Structure

```
docs/
├── INDEX.md                        # This file
├── RESTRUCTURING_GUIDE.md          # Project organization (READ THIS FIRST!)
├── research_proposal.tex           # CVPR paper draft
│
└── reports/                        # Research reports
    ├── INDEX.md                    # Original report index
    ├── EXECUTIVE_SUMMARY.md        # High-level overview
    ├── PROJECT_SUMMARY.md          # Project summary
    ├── TECHNICAL_REPORT.md         # Technical details
    ├── GETTING_STARTED.md          # Setup guide
    ├── QUICK_REFERENCE.md          # Command reference
    ├── GIF_QUICK_START.md          # GIF creation guide
    ├── GIF_VISUALIZATION_README.md # Visualization tools
    └── PRESENTATION_SLIDES.md      # Presentation materials
```

## 🚀 Recommended Reading Order

### For New Contributors
1. [RESTRUCTURING_GUIDE.md](RESTRUCTURING_GUIDE.md) - Understand project structure
2. [GETTING_STARTED.md](reports/GETTING_STARTED.md) - Set up environment
3. [QUICK_REFERENCE.md](reports/QUICK_REFERENCE.md) - Learn key commands
4. [TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md) - Deep dive into implementation

### For Researchers
1. [EXECUTIVE_SUMMARY.md](reports/EXECUTIVE_SUMMARY.md) - Research overview
2. [research_proposal.tex](research_proposal.tex) - Full paper
3. [TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md) - Implementation details
4. [PROJECT_SUMMARY.md](reports/PROJECT_SUMMARY.md) - Project context

### For Quick Tasks
1. [QUICK_REFERENCE.md](reports/QUICK_REFERENCE.md) - Command cheat sheet
2. [GETTING_STARTED.md](reports/GETTING_STARTED.md) - Setup instructions
3. [GIF_QUICK_START.md](reports/GIF_QUICK_START.md) - Create visualizations

## 📊 Project Status (December 2024)

- ✅ **Codebase restructured** - Clean organization, ready for production
- ✅ **ATAT implementation complete** - All 4 components implemented
- ✅ **Research proposal written** - CVPR paper draft ready
- ✅ **Testing infrastructure** - Unit tests and ablation scripts
- ⏳ **Training runs pending** - Ready to start experiments
- ⏳ **Results collection** - Awaiting experimental data

## 🎯 Next Steps

See [RESTRUCTURING_GUIDE.md](RESTRUCTURING_GUIDE.md) for detailed next steps. Summary:

1. **Immediate**: Start ATAT-Tiny training (10k steps test run)
2. **This Week**: Full ATAT-Tiny training (100k steps)
3. **This Month**: ATAT-Small on OpenWebText, collect results
4. **Next Month**: Finalize paper, prepare submission

## 📁 Related Files

- Main project README: `../README.md` (MDLM baseline)
- ATAT README: `../mdlm_atat/README.md` (ATAT extension)
- Configuration examples: `../mdlm_atat/configs/atat/`
- Training scripts: `../mdlm_atat/scripts/`

## 🔄 Recent Updates

**December 3, 2024**: Major restructuring
- Consolidated all reports into `docs/reports/`
- Created unified documentation index
- Added comprehensive restructuring guide
- Cleaned `__pycache__` and temporary files
- Updated all data paths to `/media/scratch/adele/`

---

**Maintained by**: Adele Chinda  
**Last Updated**: December 3, 2024
