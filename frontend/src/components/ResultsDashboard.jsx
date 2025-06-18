import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Download, BarChart, Activity } from 'lucide-react'

export function ResultsDashboard({ results, uploadedData, targetColumn, onReset }) {
  if (!results) return null

  const { accuracy, precision, recall, f1_score, classes, confusion_matrix, predictions } = results

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold">Model Results</h2>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onReset}>
            New Analysis
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Download Results
          </Button>
        </div>
      </div>

      {/* Metrics Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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
      </div>

      {/* Confusion Matrix */}
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

      {/* Predictions Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto max-h-96">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Index</TableHead>
                  {uploadedData.headers.filter(h => h !== targetColumn).slice(0, 3).map((header) => (
                    <TableHead key={header}>{header}</TableHead>
                  ))}
                  <TableHead>Actual</TableHead>
                  <TableHead>Predicted</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {uploadedData.data.slice(0, 50).map((row, index) => (
                  <TableRow key={index}>
                    <TableCell>{index + 1}</TableCell>
                    {uploadedData.headers.filter(h => h !== targetColumn).slice(0, 3).map((header) => (
                      <TableCell key={header}>{row[header]}</TableCell>
                    ))}
                    <TableCell>{row[targetColumn]}</TableCell>
                    <TableCell className={row[targetColumn] === predictions[index] ? 'text-green-600' : 'text-red-600'}>
                      {predictions[index]}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          {uploadedData.data.length > 50 && (
            <p className="text-sm text-muted-foreground mt-2 text-center">
              Showing first 50 predictions of {predictions.length} total
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}