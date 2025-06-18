# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **TabPFN Frontend Application** - a web-based interface for the TabPFN (Tabular Prior-Fitted Network) machine learning model. The application enables users to upload tabular datasets, perform classification, view predictions, and analyze model interpretability through SHAP values.

**Current Status**: Project is in planning phase with comprehensive specification in PLAN.md. No implementation code exists yet.

## Technical Stack

**Frontend** (Not yet implemented):
- React.js with functional components and hooks
- Tailwind CSS for styling
- Shadcn/ui for UI components
- Recharts/Nivo for data visualizations
- React Context API or Zustand for state management

**Backend** (Planned):
- Flask or FastAPI (Python)
- Libraries: tabpfn, pandas, scikit-learn, shap
- Docker containerization

## Development Commands

**Note**: No package.json, requirements.txt, or build configuration exists yet. These commands will be relevant once implementation begins:

```bash
# Frontend (when React project is created)
npm install          # Install dependencies
npm run dev          # Development server
npm run build        # Production build
npm run lint         # Code linting
npm run test         # Run tests

# Backend (when Python backend is created)
pip install -r requirements.txt  # Install Python dependencies
python app.py                     # Run Flask/FastAPI server
pytest                           # Run Python tests
```

## Architecture Overview

**User Flow** (from PLAN.md):
1. **Data Upload**: Users upload CSV/Excel files via drag-and-drop interface
2. **Data Preview**: Display sample data with target column selection
3. **Model Processing**: Send data to backend API for TabPFN training/prediction
4. **Results Display**: Show predictions, metrics, confusion matrix, ROC curve
5. **SHAP Analysis**: Provide model interpretability through SHAP visualizations
6. **Download**: Export predictions and SHAP values

**Component Structure** (Planned):
```
App.jsx
├── Header.jsx
├── FileUpload.jsx
├── DataTablePreview.jsx
├── LoadingSpinner.jsx
├── ResultsDashboard.jsx
│   ├── MetricsSummary.jsx
│   ├── ConfusionMatrixChart.jsx
│   ├── RocCurveChart.jsx
│   ├── PredictionsTable.jsx
│   └── ShapExplanation.jsx
│       ├── ShapSummaryPlot.jsx
│       ├── ShapDependencePlot.jsx
│       └── ShapIndividualPlot.jsx
├── ErrorMessage.jsx
└── DownloadButton.jsx
```

**API Endpoints** (Planned):
- `POST /predict` - Dataset processing and predictions
- `POST /shap` - SHAP value computation

## Development Phases

1. **Phase 1**: Foundation (1-2 weeks) - React setup, file upload, data preview
2. **Phase 2**: Core Functionality (2-3 weeks) - TabPFN integration, results display
3. **Phase 3**: Interpretability & Refinement (2-3 weeks) - SHAP implementation, UI polish

## Key Implementation Notes

- Use Inter font and rounded corners (rounded-md/rounded-lg) throughout
- Ensure responsive design with Tailwind utilities (sm:, md:, lg:)
- Implement comprehensive error handling with actionable user messages
- Support CSV and XLSX file formats with proper validation
- All visualizations must be interactive and clearly labeled
- Maintain clean, minimalist design with consistent color palette

## Important Files

- `PLAN.md` - Comprehensive 374-line project specification with detailed requirements
- `.claude/settings.local.json` - Claude Code permissions configuration

## Next Steps

Project needs initial setup:
1. Initialize React project with Vite or Create React App
2. Configure Tailwind CSS and Shadcn/ui
3. Set up Python backend with Flask/FastAPI
4. Create basic project structure following the planned component hierarchy