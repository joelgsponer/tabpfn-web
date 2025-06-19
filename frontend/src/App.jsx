import { useState } from 'react'
import { Header } from './components/Header'
import { FileUpload } from './components/FileUpload'
import { DataTablePreview } from './components/DataTablePreview'
import { LoadingSpinner } from './components/LoadingSpinner'
import { ErrorMessage } from './components/ErrorMessage'
import { ResultsDashboard } from './components/ResultsDashboard'

function App() {
  const [uploadedData, setUploadedData] = useState(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [columnTypes, setColumnTypes] = useState({})
  const [excludedColumns, setExcludedColumns] = useState([])
  const [modelConfig, setModelConfig] = useState({
    test_size: 0.2,
    N_ensemble_configurations: 16,
    device: 'auto',
    max_samples: 1000,
    random_state: 42,
    n_estimators: 4,
    n_jobs: -1
  })
  const [modelResults, setModelResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const detectColumnType = (fileData, header) => {
    const sample = fileData.data.slice(0, 10).map(row => row[header]).filter(val => val !== '')
    
    // Check if all values are numeric
    const numericValues = sample.filter(val => !isNaN(val) && val !== '')
    if (numericValues.length === sample.length && sample.length > 0) {
      // Check if all are integers
      const integerValues = numericValues.filter(val => Number.isInteger(parseFloat(val)))
      if (integerValues.length === numericValues.length) {
        return 'integer'
      }
      return 'numeric'
    }
    
    // Check if it looks like a categorical variable
    const uniqueValues = [...new Set(sample)]
    if (uniqueValues.length <= Math.max(2, sample.length * 0.5)) {
      return 'categorical'
    }
    
    return 'text'
  }

  const handleFileUpload = (fileData) => {
    setUploadedData(fileData)
    setTargetColumn('')
    setExcludedColumns([])
    
    // Auto-detect column types
    const detectedTypes = {}
    fileData.headers.forEach(header => {
      detectedTypes[header] = detectColumnType(fileData, header)
    })
    setColumnTypes(detectedTypes)
    
    setModelResults(null)
    setError(null)
  }

  const handleColumnTypeChange = (column, type) => {
    setColumnTypes(prev => ({ ...prev, [column]: type }))
  }

  const handleExcludedColumnsChange = (excludedCols) => {
    setExcludedColumns(excludedCols)
    // Remove target column from excluded if it was selected
    if (excludedCols.includes(targetColumn)) {
      setTargetColumn('')
    }
  }

  const handleModelConfigChange = (config) => {
    setModelConfig(config)
  }

  const handleReset = () => {
    setUploadedData(null)
    setTargetColumn('')
    setColumnTypes({})
    setExcludedColumns([])
    setModelConfig({
      test_size: 0.2,
      N_ensemble_configurations: 16,
      device: 'auto',
      max_samples: 1000,
      random_state: 42,
      n_estimators: 4,
      n_jobs: -1
    })
    setModelResults(null)
    setError(null)
    setIsLoading(false)
  }

  const handleRunModel = async () => {
    if (!uploadedData || !targetColumn) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: uploadedData.data,
          target_column: targetColumn,
          column_types: columnTypes,
          excluded_columns: excludedColumns,
          model_config_params: modelConfig
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const results = await response.json()
      setModelResults(results)
    } catch (err) {
      setError(err.message || 'An error occurred while processing your data')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {!uploadedData && (
            <>
              <h2 className="text-3xl font-bold mb-8">Upload Your Dataset</h2>
              <FileUpload onFileUpload={handleFileUpload} />
            </>
          )}
          
          {uploadedData && !modelResults && (
            <DataTablePreview
              fileData={uploadedData}
              targetColumn={targetColumn}
              onTargetColumnChange={setTargetColumn}
              onRunModel={handleRunModel}
              isLoading={isLoading}
              columnTypes={columnTypes}
              onColumnTypeChange={handleColumnTypeChange}
              excludedColumns={excludedColumns}
              onExcludedColumnsChange={handleExcludedColumnsChange}
              modelConfig={modelConfig}
              onModelConfigChange={handleModelConfigChange}
            />
          )}
          
          {isLoading && <LoadingSpinner />}
          
          {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}
          
          {modelResults && (
            <ResultsDashboard
              results={modelResults}
              uploadedData={uploadedData}
              targetColumn={targetColumn}
              onReset={handleReset}
            />
          )}
        </div>
      </main>
    </div>
  )
}

export default App
