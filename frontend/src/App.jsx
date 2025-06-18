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
  const [modelResults, setModelResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileUpload = (fileData) => {
    setUploadedData(fileData)
    setTargetColumn('')
    setModelResults(null)
    setError(null)
  }

  const handleReset = () => {
    setUploadedData(null)
    setTargetColumn('')
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
          target_column: targetColumn
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
