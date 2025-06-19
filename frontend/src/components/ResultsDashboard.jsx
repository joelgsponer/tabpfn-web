import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Download, BarChart, Activity, Brain, AlertCircle } from 'lucide-react'

export function ResultsDashboard({ results, uploadedData, targetColumn, onReset }) {
  const [shapData, setShapData] = useState(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapError, setShapError] = useState(null)
  const [selectedInstance, setSelectedInstance] = useState(0)
  const [shapOverviewData, setShapOverviewData] = useState(null)
  const [shapOverviewLoading, setShapOverviewLoading] = useState(false)
  const [shapOverviewError, setShapOverviewError] = useState(null)
  const [shapOverviewInstances, setShapOverviewInstances] = useState(20)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(25)
  const [shapProgress, setShapProgress] = useState(0)
  const [shapOverviewProgress, setShapOverviewProgress] = useState(0)

  if (!results) return null

  const { accuracy, precision, recall, f1_score, classes, confusion_matrix, predictions, model_data, model_metadata } = results
  
  // Detect if this is a regression task
  const isRegression = classes.length === 0 || model_metadata?.model_type === 'TabPFNRegressor'

  const computeShap = async (instanceIndex = selectedInstance) => {
    if (!model_data) {
      setShapError('Model data not available for SHAP computation')
      return
    }

    setShapLoading(true)
    setShapError(null)
    setSelectedInstance(instanceIndex)
    setShapProgress(0)

    // Progress simulation for individual SHAP (typically takes 5-15 seconds)
    const progressInterval = setInterval(() => {
      setShapProgress(prev => {
        if (prev >= 95) return prev
        // Slower progress towards the end
        const increment = prev < 50 ? 8 : prev < 80 ? 4 : 1
        return Math.min(prev + increment, 95)
      })
    }, 500)

    try {
      const response = await fetch('http://localhost:5000/shap', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_data: model_data,
          data: uploadedData.data,
          instance_index: instanceIndex
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'SHAP computation failed')
      }

      const shapResponse = await response.json()
      setShapProgress(100)
      setTimeout(() => setShapData(shapResponse), 200) // Small delay to show 100%
    } catch (error) {
      console.error('SHAP computation error:', error)
      setShapError(error.message)
    } finally {
      clearInterval(progressInterval)
      setTimeout(() => {
        setShapLoading(false)
        setShapProgress(0)
      }, 300)
    }
  }

  const computeShapOverview = async () => {
    if (!model_data) {
      setShapOverviewError('Model data not available for SHAP overview computation')
      return
    }

    setShapOverviewLoading(true)
    setShapOverviewError(null)
    setShapOverviewProgress(0)

    // Progress simulation for SHAP overview (scales with number of instances)
    const estimatedTime = shapOverviewInstances * 2000 // ~2 seconds per instance
    const progressInterval = setInterval(() => {
      setShapOverviewProgress(prev => {
        if (prev >= 95) return prev
        // Faster initial progress, slower towards end
        const increment = prev < 30 ? 5 : prev < 70 ? 3 : 0.5
        return Math.min(prev + increment, 95)
      })
    }, Math.max(200, estimatedTime / 100)) // Adjust interval based on estimated time

    try {
      const response = await fetch('http://localhost:5000/shap-overview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_data: model_data,
          data: uploadedData.data,
          num_instances: shapOverviewInstances
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'SHAP overview computation failed')
      }

      const shapOverviewResponse = await response.json()
      setShapOverviewProgress(100)
      setTimeout(() => setShapOverviewData(shapOverviewResponse), 200) // Small delay to show 100%
    } catch (error) {
      console.error('SHAP overview computation error:', error)
      setShapOverviewError(error.message)
    } finally {
      clearInterval(progressInterval)
      setTimeout(() => {
        setShapOverviewLoading(false)
        setShapOverviewProgress(0)
      }, 300)
    }
  }

  const downloadModel = () => {
    if (!model_data || !model_metadata) {
      alert('Model data not available for download')
      return
    }

    // Convert base64 to blob
    const byteCharacters = atob(model_data)
    const byteNumbers = new Array(byteCharacters.length)
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }
    const byteArray = new Uint8Array(byteNumbers)
    const blob = new Blob([byteArray], { type: 'application/octet-stream' })
    
    // Create download link
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `tabpfn_model_${model_metadata.target_column}_${new Date().toISOString().split('T')[0]}.pkl`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const downloadResults = () => {
    // Prepare CSV data with predictions
    const csvData = uploadedData.data.map((row, index) => ({
      ...row,
      predicted: predictions[index],
      correct: row[targetColumn] === predictions[index] ? 'Yes' : 'No'
    }))
    
    // Convert to CSV
    const headers = Object.keys(csvData[0])
    const csvContent = [
      headers.join(','),
      ...csvData.map(row => headers.map(header => `"${row[header]}"`).join(','))
    ].join('\n')
    
    // Download CSV
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `predictions_${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold">Model Results</h2>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onReset}>
            New Analysis
          </Button>
          <Button variant="outline" onClick={downloadResults}>
            <Download className="h-4 w-4 mr-2" />
            Download Results CSV
          </Button>
          <Button onClick={downloadModel}>
            <Download className="h-4 w-4 mr-2" />
            Download Model
          </Button>
        </div>
      </div>

      {/* 1. Overall Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {isRegression ? (
          <>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">R² Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(accuracy * 100).toFixed(2)}%</div>
                <p className="text-xs text-muted-foreground">Variance explained</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">MAE</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{precision[0]?.toFixed(3)}</div>
                <p className="text-xs text-muted-foreground">Mean Absolute Error</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">MSE</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{recall[0]?.toFixed(3)}</div>
                <p className="text-xs text-muted-foreground">Mean Squared Error</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">RMSE</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{Math.sqrt(recall[0]).toFixed(3)}</div>
                <p className="text-xs text-muted-foreground">Root Mean Squared Error</p>
              </CardContent>
            </Card>
          </>
        ) : (
          <>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{(accuracy * 100).toFixed(2)}%</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg Precision</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(precision.reduce((a, b) => a + b, 0) / precision.length * 100).toFixed(2)}%
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg Recall</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(recall.reduce((a, b) => a + b, 0) / recall.length * 100).toFixed(2)}%
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Avg F1 Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(f1_score.reduce((a, b) => a + b, 0) / f1_score.length * 100).toFixed(2)}%
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>

      {/* Confusion Matrix for Classification */}
      {!isRegression && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart className="h-5 w-5" />
              Confusion Matrix
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Actual \ Predicted</TableHead>
                    {classes.map((cls) => (
                      <TableHead key={cls} className="text-center">{cls}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {confusion_matrix.map((row, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-medium">{classes[i]}</TableCell>
                      {row.map((value, j) => (
                        <TableCell key={j} className="text-center">
                          {value}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 2. Predictions Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Predictions
            </div>
            <div className="text-sm text-muted-foreground">
              {predictions.length} total predictions
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Pagination Controls */}
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {Math.min((currentPage - 1) * itemsPerPage + 1, predictions.length)} to {Math.min(currentPage * itemsPerPage, predictions.length)} of {predictions.length} predictions
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>
                <span className="text-sm">
                  Page {currentPage} of {Math.ceil(predictions.length / itemsPerPage)}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(Math.min(Math.ceil(predictions.length / itemsPerPage), currentPage + 1))}
                  disabled={currentPage === Math.ceil(predictions.length / itemsPerPage)}
                >
                  Next
                </Button>
              </div>
            </div>

            {/* Predictions Table */}
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Index</TableHead>
                    {uploadedData.headers.filter(h => h !== targetColumn).slice(0, 3).map((header) => (
                      <TableHead key={header}>{header}</TableHead>
                    ))}
                    <TableHead>Actual</TableHead>
                    <TableHead>Predicted</TableHead>
                    {isRegression && <TableHead>Error</TableHead>}
                    <TableHead className="text-center">SHAP</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(() => {
                    const startIndex = (currentPage - 1) * itemsPerPage
                    const endIndex = Math.min(startIndex + itemsPerPage, predictions.length)
                    const currentPageData = uploadedData.data.slice(startIndex, endIndex)
                    
                    return currentPageData.map((row, pageIndex) => {
                      const actualIndex = startIndex + pageIndex
                      return (
                        <TableRow key={actualIndex}>
                          <TableCell className="font-medium">{actualIndex + 1}</TableCell>
                          {uploadedData.headers.filter(h => h !== targetColumn).slice(0, 3).map((header) => (
                            <TableCell key={header}>
                              {typeof row[header] === 'number' ? 
                                (Number.isInteger(row[header]) ? row[header] : row[header].toFixed(3)) : 
                                row[header]
                              }
                            </TableCell>
                          ))}
                          <TableCell>{row[targetColumn]}</TableCell>
                          <TableCell className={isRegression ? '' : (row[targetColumn] === predictions[actualIndex] ? 'text-green-600' : 'text-red-600')}>
                            {isRegression ? predictions[actualIndex].toFixed(3) : predictions[actualIndex]}
                          </TableCell>
                          {isRegression && (
                            <TableCell className="text-muted-foreground">
                              {(parseFloat(row[targetColumn]) - predictions[actualIndex]).toFixed(3)}
                            </TableCell>
                          )}
                          <TableCell className="text-center">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => computeShap(actualIndex)}
                              disabled={shapLoading}
                              className="h-7 px-2 text-xs"
                            >
                              <Brain className="h-3 w-3 mr-1" />
                              {shapLoading && selectedInstance === actualIndex ? 'Loading...' : 'Explain'}
                            </Button>
                          </TableCell>
                        </TableRow>
                      )
                    })
                  })()}
                </TableBody>
              </Table>
            </div>

            {/* Bottom Pagination */}
            <div className="flex items-center justify-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(1)}
                disabled={currentPage === 1}
              >
                First
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              
              {/* Page Numbers */}
              {(() => {
                const totalPages = Math.ceil(predictions.length / itemsPerPage)
                const maxVisiblePages = 5
                let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2))
                let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1)
                
                if (endPage - startPage + 1 < maxVisiblePages) {
                  startPage = Math.max(1, endPage - maxVisiblePages + 1)
                }
                
                const pageNumbers = []
                for (let i = startPage; i <= endPage; i++) {
                  pageNumbers.push(i)
                }
                
                return pageNumbers.map(pageNum => (
                  <Button
                    key={pageNum}
                    variant={currentPage === pageNum ? "default" : "outline"}
                    size="sm"
                    onClick={() => setCurrentPage(pageNum)}
                    className="w-8 h-8"
                  >
                    {pageNum}
                  </Button>
                ))
              })()}
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.min(Math.ceil(predictions.length / itemsPerPage), currentPage + 1))}
                disabled={currentPage === Math.ceil(predictions.length / itemsPerPage)}
              >
                Next
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(Math.ceil(predictions.length / itemsPerPage))}
                disabled={currentPage === Math.ceil(predictions.length / itemsPerPage)}
              >
                Last
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3. Individual SHAP Analysis */}
      {(shapData || shapError || shapLoading) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              SHAP Analysis - Model Interpretability
            </CardTitle>
          </CardHeader>
          <CardContent>
            {shapLoading && (
              <div className="flex items-center justify-center py-8">
                <div className="text-center w-full max-w-md">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto mb-4"></div>
                  <p className="text-muted-foreground mb-4">Computing SHAP values for instance #{selectedInstance + 1}...</p>
                  
                  {/* Progress Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${shapProgress}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {shapProgress}% complete {shapProgress > 0 && shapProgress < 100 && '• Analyzing feature contributions...'}
                    {shapProgress >= 100 && '• Finalizing results...'}
                  </p>
                </div>
              </div>
            )}

            {shapError && (
              <div className="flex items-center gap-2 p-4 bg-red-50 text-red-700 rounded-lg">
                <AlertCircle className="h-5 w-5" />
                <div>
                  <p className="font-medium">SHAP Computation Failed</p>
                  <p className="text-sm">{shapError}</p>
                </div>
              </div>
            )}

            {shapData && (
              <div className="space-y-6">
                {/* Current instance info */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="text-sm font-medium mb-1">Explaining Instance #{selectedInstance + 1}</h4>
                  <p className="text-sm text-muted-foreground">
                    {(() => {
                      const row = uploadedData.data[selectedInstance]
                      const previewFeatures = uploadedData.headers
                        .filter(h => h !== targetColumn)
                        .slice(0, 2)
                        .map(feature => {
                          const value = row[feature]
                          if (typeof value === 'number') {
                            return `${feature}=${value.toFixed(2)}`
                          }
                          return `${feature}=${value}`
                        })
                        .join(', ')
                      
                      const targetValue = row[targetColumn]
                      const targetDisplay = typeof targetValue === 'number' ? 
                        targetValue.toFixed(2) : targetValue
                      
                      return `${previewFeatures} → ${targetColumn}=${targetDisplay}`
                    })()}
                  </p>
                </div>

                {/* SHAP Interpretation */}
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Interpretation</h4>
                  <pre className="text-sm text-blue-800 whitespace-pre-wrap">{shapData.interpretation}</pre>
                </div>

                {/* Feature Values Table */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Instance Values</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Feature</TableHead>
                            <TableHead>Value</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(shapData.instance_values).map(([feature, value]) => (
                            <TableRow key={feature}>
                              <TableCell className="font-medium">{feature}</TableCell>
                              <TableCell>{typeof value === 'number' ? value.toFixed(3) : value}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>

                  {/* SHAP Values */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">SHAP Feature Contributions</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {Object.entries(shapData.shap_values)
                          .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
                          .map(([feature, value]) => (
                          <div key={feature} className="flex items-center gap-2">
                            <div className="w-24 text-sm font-medium truncate" title={feature}>
                              {feature}
                            </div>
                            <div className="flex-1 bg-gray-200 rounded-full h-4 relative">
                              <div 
                                className={`h-4 rounded-full ${value >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{ 
                                  width: `${Math.abs(value) / Math.max(...Object.values(shapData.shap_values).map(Math.abs)) * 100}%`,
                                  marginLeft: value < 0 ? `${100 - Math.abs(value) / Math.max(...Object.values(shapData.shap_values).map(Math.abs)) * 100}%` : '0'
                                }}
                              />
                            </div>
                            <div className={`w-16 text-sm font-mono ${value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {value >= 0 ? '+' : ''}{value.toFixed(3)}
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      <div className="mt-4 pt-4 border-t text-sm text-muted-foreground">
                        <div className="flex justify-between">
                          <span>Baseline:</span>
                          <span>{shapData.baseline_value.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Prediction:</span>
                          <span>{shapData.predicted_value.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between font-medium">
                          <span>Sum Check:</span>
                          <span>
                            {(Object.values(shapData.shap_values).reduce((a, b) => a + b, 0) + shapData.baseline_value).toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 4. Summary SHAP Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart className="h-5 w-5" />
              SHAP Overview - Global Feature Importance
            </div>
            <div className="flex items-center gap-2">
              <select 
                value={shapOverviewInstances} 
                onChange={(e) => setShapOverviewInstances(parseInt(e.target.value))}
                className="px-3 py-1 border rounded-md text-sm"
                title="Number of instances for overview"
              >
                <option value={10}>10 instances</option>
                <option value={20}>20 instances</option>
                <option value={50}>50 instances</option>
                <option value={100}>100 instances</option>
                <option value={200}>200 instances</option>
              </select>
              <Button 
                onClick={computeShapOverview} 
                disabled={shapOverviewLoading}
                className="bg-indigo-600 hover:bg-indigo-700"
              >
                <BarChart className="h-4 w-4 mr-2" />
                {shapOverviewLoading ? 'Computing Overview...' : 'SHAP Overview'}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!shapOverviewData && !shapOverviewLoading && !shapOverviewError && (
            <div className="text-center py-8">
              <BarChart className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">Click "SHAP Overview" to analyze global feature importance across multiple instances</p>
            </div>
          )}
          
          {shapOverviewLoading && (
            <div className="flex items-center justify-center py-8">
              <div className="text-center w-full max-w-md">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-4"></div>
                <p className="text-muted-foreground mb-4">Computing SHAP overview for {shapOverviewInstances} instances...</p>
                
                {/* Progress Bar */}
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div 
                    className="bg-indigo-600 h-2 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${shapOverviewProgress}%` }}
                  ></div>
                </div>
                <p className="text-xs text-muted-foreground">
                  {shapOverviewProgress}% complete {shapOverviewProgress > 0 && shapOverviewProgress < 100 && '• Processing instances...'}
                  {shapOverviewProgress >= 100 && '• Generating visualizations...'}
                </p>
                
                {/* Estimated time remaining */}
                {shapOverviewProgress > 10 && shapOverviewProgress < 95 && (
                  <p className="text-xs text-muted-foreground mt-2 opacity-75">
                    Est. time: ~{Math.max(1, Math.round((100 - shapOverviewProgress) / 100 * shapOverviewInstances * 2))}s remaining
                  </p>
                )}
              </div>
            </div>
          )}

          {shapOverviewError && (
            <div className="flex items-center gap-2 p-4 bg-red-50 text-red-700 rounded-lg">
              <AlertCircle className="h-5 w-5" />
              <div>
                <p className="font-medium">SHAP Overview Computation Failed</p>
                <p className="text-sm">{shapOverviewError}</p>
              </div>
            </div>
          )}

          {shapOverviewData && (
              <div className="space-y-6">
                {/* Summary */}
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Global Feature Importance</h4>
                  <p className="text-sm text-blue-800">
                    Based on SHAP values computed for {shapOverviewData.shap_values_matrix.length} instances.
                    Features are ranked by mean absolute SHAP value.
                  </p>
                </div>

                {/* SHAP Summary Plot (Beeswarm-style) */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Feature Impact Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {(() => {
                        // Find global min/max for consistent scale across all features
                        let globalMinShap = Infinity
                        let globalMaxShap = -Infinity
                        
                        shapOverviewData.summary_plot_data.feature_importance_order.slice(0, 15).forEach(feature => {
                          const featureIndex = shapOverviewData.feature_names.indexOf(feature)
                          const shapValues = shapOverviewData.shap_values_matrix.map(row => row[featureIndex])
                          globalMinShap = Math.min(globalMinShap, ...shapValues)
                          globalMaxShap = Math.max(globalMaxShap, ...shapValues)
                        })
                        
                        // Ensure the scale includes zero
                        globalMinShap = Math.min(globalMinShap, 0)
                        globalMaxShap = Math.max(globalMaxShap, 0)
                        const globalRange = globalMaxShap - globalMinShap
                        
                        return shapOverviewData.summary_plot_data.feature_importance_order.slice(0, 15).map((feature, idx) => {
                          const featureIndex = shapOverviewData.feature_names.indexOf(feature)
                          const shapValues = shapOverviewData.shap_values_matrix.map(row => row[featureIndex])
                          const featureValues = shapOverviewData.feature_values_matrix.map(row => row[featureIndex])
                          const meanAbsShap = shapOverviewData.mean_abs_shap[feature]
                          
                          // Calculate min/max for gradient color coding
                          const numericFeatureValues = featureValues.filter(v => typeof v === 'number')
                          let minFeatureValue, maxFeatureValue
                          if (numericFeatureValues.length > 0) {
                            minFeatureValue = Math.min(...numericFeatureValues)
                            maxFeatureValue = Math.max(...numericFeatureValues)
                          } else {
                            minFeatureValue = 0
                            maxFeatureValue = 1
                          }
                          
                          // Function to get gradient color from sky blue → purple → tomato
                          const getGradientColor = (value) => {
                            if (typeof value !== 'number' || minFeatureValue === maxFeatureValue) {
                              return 'rgb(135, 206, 235)' // Default sky blue for non-numeric
                            }
                            
                            // Normalize value to 0-1 range
                            const normalized = (value - minFeatureValue) / (maxFeatureValue - minFeatureValue)
                            
                            // Define color points: sky blue (0) → purple (0.5) → tomato (1)
                            if (normalized <= 0.5) {
                              // Interpolate between sky blue and purple
                              const t = normalized * 2 // Scale to 0-1 for this segment
                              const r = Math.round(135 + (128 - 135) * t) // 135 → 128
                              const g = Math.round(206 + (0 - 206) * t)   // 206 → 0  
                              const b = Math.round(235 + (128 - 235) * t) // 235 → 128
                              return `rgb(${r}, ${g}, ${b})`
                            } else {
                              // Interpolate between purple and tomato
                              const t = (normalized - 0.5) * 2 // Scale to 0-1 for this segment
                              const r = Math.round(128 + (255 - 128) * t) // 128 → 255
                              const g = Math.round(0 + (99 - 0) * t)      // 0 → 99
                              const b = Math.round(128 + (71 - 128) * t)  // 128 → 71
                              return `rgb(${r}, ${g}, ${b})`
                            }
                          }
                          
                          return (
                            <div key={feature} className="flex items-center gap-4">
                              <div className="w-32 text-sm font-medium truncate" title={feature}>
                                {feature}
                              </div>
                              <div className="flex-1 relative h-10 bg-gray-100 rounded overflow-visible">
                                {/* Zero line - consistent position across all features */}
                                <div 
                                  className="absolute top-0 bottom-0 w-px bg-gray-600"
                                  style={{ left: `${((0 - globalMinShap) / globalRange) * 100}%` }}
                                />
                                {/* SHAP value dots with violin effect */}
                                {(() => {
                                  // Create bins for density calculation
                                  const binWidth = 2 // Percentage width of each bin
                                  const bins = {}
                                  
                                  // Sort values by SHAP value for better stacking
                                  const sortedIndices = shapValues
                                    .map((val, idx) => ({ val, idx }))
                                    .sort((a, b) => a.val - b.val)
                                    .map(item => item.idx)
                                  
                                  // Calculate positions and bin assignments
                                  const dots = sortedIndices.map(i => {
                                    const shapVal = shapValues[i]
                                    const featureVal = featureValues[i]
                                    const position = ((shapVal - globalMinShap) / globalRange) * 100
                                    const binIndex = Math.floor(position / binWidth)
                                    
                                    // Initialize bin if needed
                                    if (!bins[binIndex]) bins[binIndex] = []
                                    bins[binIndex].push({ i, shapVal, featureVal, position })
                                    
                                    return { i, shapVal, featureVal, position, binIndex }
                                  })
                                  
                                  // Calculate vertical positions with violin effect
                                  return dots.map(({ i, shapVal, featureVal, position, binIndex }) => {
                                    const binDots = bins[binIndex]
                                    const dotIndex = binDots.findIndex(d => d.i === i)
                                    const binSize = binDots.length
                                    
                                    // Create violin shape by spreading dots vertically based on density
                                    const maxSpread = 20 // Maximum vertical spread in pixels
                                    const spread = Math.min(binSize * 3, maxSpread)
                                    
                                    // Calculate vertical position to create violin shape
                                    let yOffset
                                    if (binSize === 1) {
                                      yOffset = 0
                                    } else {
                                      // Distribute dots symmetrically around center
                                      const normalizedIndex = (dotIndex - (binSize - 1) / 2) / ((binSize - 1) / 2)
                                      // Add some randomness for natural look
                                      const jitter = (Math.random() - 0.5) * 1.5
                                      yOffset = normalizedIndex * spread / 2 + jitter
                                    }
                                    
                                    // Get gradient color for this feature value
                                    const dotColor = getGradientColor(featureVal)
                                    
                                    return (
                                      <div
                                        key={i}
                                        className="absolute w-1.5 h-1.5 rounded-full opacity-70 hover:opacity-100 transition-opacity"
                                        style={{ 
                                          left: `${Math.max(0, Math.min(98, position))}%`,
                                          top: `${20 + yOffset}px`,
                                          transform: 'translateX(-50%)',
                                          backgroundColor: dotColor
                                        }}
                                        title={`SHAP: ${shapVal.toFixed(3)}, Value: ${featureVal}`}
                                      />
                                    )
                                  })
                                })()}
                              </div>
                              <div className="w-20 text-xs text-muted-foreground text-right">
                                Avg: {meanAbsShap.toFixed(3)}
                              </div>
                            </div>
                          )
                        })
                      })()}
                    </div>
                    
                    <div className="mt-4 pt-4 border-t">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <div className="flex items-center gap-4">
                          {/* Gradient Color Legend */}
                          <div className="flex items-center gap-2">
                            <span>Feature values:</span>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 rounded-full opacity-70" style={{ backgroundColor: 'rgb(135, 206, 235)' }}></div>
                              <span>Low</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 rounded-full opacity-70" style={{ backgroundColor: 'rgb(128, 0, 128)' }}></div>
                              <span>Mid</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <div className="w-3 h-3 rounded-full opacity-70" style={{ backgroundColor: 'rgb(255, 99, 71)' }}></div>
                              <span>High</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span>← Negative impact</span>
                          <div className="w-px h-4 bg-gray-600"></div>
                          <span>Positive impact →</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Feature Importance Ranking */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Feature Importance Ranking</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {shapOverviewData.summary_plot_data.feature_importance_order.slice(0, 10).map((feature, idx) => {
                        const meanAbsShap = shapOverviewData.mean_abs_shap[feature]
                        const maxImportance = Math.max(...Object.values(shapOverviewData.mean_abs_shap))
                        
                        return (
                          <div key={feature} className="flex items-center gap-3">
                            <div className="w-6 text-sm font-bold text-gray-600">
                              {idx + 1}
                            </div>
                            <div className="w-32 text-sm font-medium truncate" title={feature}>
                              {feature}
                            </div>
                            <div className="flex-1 bg-gray-200 rounded-full h-3">
                              <div 
                                className="bg-indigo-500 h-3 rounded-full"
                                style={{ width: `${(meanAbsShap / maxImportance) * 100}%` }}
                              />
                            </div>
                            <div className="w-16 text-sm font-mono text-right">
                              {meanAbsShap.toFixed(3)}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </CardContent>
        </Card>
    </div>
  )
}
