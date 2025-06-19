import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { Button } from './ui/button'
import { Checkbox } from './ui/checkbox'
import { Settings, Eye, EyeOff } from 'lucide-react'

export function ColumnConfiguration({ fileData, columnTypes, onColumnTypeChange, excludedColumns, onExcludedColumnsChange }) {
  if (!fileData) return null

  const detectColumnType = (header) => {
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

  const handleColumnToggle = (column, included) => {
    if (included) {
      onExcludedColumnsChange(excludedColumns.filter(col => col !== column))
    } else {
      onExcludedColumnsChange([...excludedColumns, column])
    }
  }

  const handleSelectAll = () => {
    onExcludedColumnsChange([])
  }

  const handleDeselectAll = () => {
    onExcludedColumnsChange([...fileData.headers])
  }

  const includedColumns = fileData.headers.filter(header => !excludedColumns.includes(header))

  return (
    <Card className="mt-4">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Column Configuration
          </CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">
              {includedColumns.length} of {fileData.headers.length} columns included
            </span>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleSelectAll}
              className="h-8"
            >
              <Eye className="h-3 w-3 mr-1" />
              All
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleDeselectAll}
              className="h-8"
            >
              <EyeOff className="h-3 w-3 mr-1" />
              None
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">Include</TableHead>
                <TableHead>Column Name</TableHead>
                <TableHead>Sample Values</TableHead>
                <TableHead>Detected Type</TableHead>
                <TableHead>Data Type</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {fileData.headers.map((header) => {
                const detectedType = detectColumnType(header)
                const currentType = columnTypes[header] || detectedType
                const sampleValues = fileData.data.slice(0, 3).map(row => row[header]).join(', ')
                const isIncluded = !excludedColumns.includes(header)
                
                return (
                  <TableRow key={header} className={!isIncluded ? 'opacity-50' : ''}>
                    <TableCell>
                      <Checkbox
                        checked={isIncluded}
                        onCheckedChange={(checked) => handleColumnToggle(header, checked)}
                      />
                    </TableCell>
                    <TableCell className={`font-medium ${!isIncluded ? 'line-through text-muted-foreground' : ''}`}>
                      {header}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground max-w-32 truncate">
                      {sampleValues}
                    </TableCell>
                    <TableCell className="text-sm">
                      <span className={`px-2 py-1 rounded text-xs ${
                        detectedType === 'numeric' ? 'bg-blue-100 text-blue-800' :
                        detectedType === 'integer' ? 'bg-green-100 text-green-800' :
                        detectedType === 'categorical' ? 'bg-purple-100 text-purple-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {detectedType}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Select
                        value={currentType}
                        onValueChange={(value) => onColumnTypeChange(header, value)}
                        disabled={!isIncluded}
                      >
                        <SelectTrigger className="w-32">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="numeric">Numeric</SelectItem>
                          <SelectItem value="integer">Integer</SelectItem>
                          <SelectItem value="categorical">Categorical</SelectItem>
                          <SelectItem value="text">Text</SelectItem>
                        </SelectContent>
                      </Select>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}