# TabPFN Web Application

A modern web interface for TabPFN (Tabular Prior-Fitted Network), enabling easy-to-use machine learning predictions on tabular data with built-in model interpretability through SHAP values.

## 🚀 Features

- **Drag-and-Drop File Upload**: Support for CSV and Excel files
- **Interactive Data Preview**: View and select target columns for prediction
- **TabPFN Predictions**: State-of-the-art tabular machine learning without hyperparameter tuning
- **Comprehensive Results Dashboard**:
  - Performance metrics (accuracy, precision, recall, F1)
  - Interactive confusion matrix
  - ROC curve visualization
  - Detailed predictions table
- **Model Interpretability**: SHAP-based explanations including:
  - Summary plots showing feature importance
  - Dependency plots for feature interactions
  - Individual prediction explanations
- **Export Functionality**: Download predictions and SHAP values

## 🛠️ Tech Stack

### Frontend
- **React** - UI framework
- **Tailwind CSS** - Styling
- **Shadcn/ui** - Component library
- **Recharts** - Data visualizations
- **Vite** - Build tool

### Backend
- **FastAPI** - Python web framework
- **TabPFN** - Machine learning model
- **SHAP** - Model interpretability
- **Pandas** - Data processing
- **scikit-learn** - ML utilities

## 📦 Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd tabpfn
```

2. **Install Frontend Dependencies**
```bash
cd frontend
npm install
```

3. **Install Backend Dependencies**
```bash
cd ../backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Running the Application

### Start the Backend Server
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```
The API will be available at `http://localhost:8000`

### Start the Frontend Development Server
```bash
cd frontend
npm run dev
```
The application will be available at `http://localhost:5173`

## 📊 Usage

1. **Upload Data**: Drag and drop or click to upload a CSV/Excel file
2. **Select Target**: Choose the column you want to predict
3. **View Results**: Explore predictions, metrics, and visualizations
4. **Analyze Interpretability**: Understand model decisions through SHAP explanations
5. **Export**: Download results for further analysis

## 🏗️ Project Structure

```
tabpfn/
├── frontend/               # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── lib/          # Utilities and helpers
│   │   └── App.jsx       # Main application
│   └── package.json
│
├── backend/               # Python API
│   ├── app.py           # FastAPI application
│   ├── requirements.txt  # Python dependencies
│   └── uploads/         # Temporary file storage
│
└── README.md
```

## 🧪 API Endpoints

- `POST /predict` - Process dataset and return predictions
  - Body: `multipart/form-data` with file and target column
  - Returns: Predictions, metrics, and visualizations data

- `POST /shap` - Generate SHAP explanations
  - Body: Model data and predictions
  - Returns: SHAP values and visualization data

## 🔧 Development

### Frontend Development
```bash
cd frontend
npm run dev      # Development server with hot reload
npm run build    # Production build
npm run preview  # Preview production build
```

### Backend Development
```bash
cd backend
python app.py    # Run with auto-reload enabled
```

## 📈 Performance Notes

- TabPFN works best with datasets up to 1000 rows and 100 features
- For larger datasets, consider sampling or feature selection
- The model provides uncertainty estimates along with predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [TabPFN](https://github.com/automl/TabPFN) - The core machine learning model
- [SHAP](https://github.com/slundberg/shap) - Model interpretability framework
- [Shadcn/ui](https://ui.shadcn.com/) - UI component library

