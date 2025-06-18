import React, { useState, useCallback } from 'react'
import { Upload, FileSpreadsheet, AlertCircle } from 'lucide-react'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'
import * as XLSX from 'xlsx'

export function FileUpload({ onFileUpload }) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)

  const validateFile = (file) => {
    const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
    const fileExtension = file.name.split('.').pop().toLowerCase()
    
    if (!validTypes.includes(file.type) && !['csv', 'xlsx', 'xls'].includes(fileExtension)) {
      return 'Please upload a CSV or Excel file (.csv, .xlsx, .xls)'
    }
    
    // 10MB limit
    if (file.size > 10 * 1024 * 1024) {
      return 'File size must be less than 10MB'
    }
    
    return null
  }

  const handleFile = useCallback((file) => {
    setError(null)
    
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    const reader = new FileReader()
    
    if (file.name.endsWith('.csv')) {
      reader.onload = (e) => {
        try {
          const text = e.target.result
          const rows = text.split('\n').filter(row => row.trim())
          
          // Detect delimiter
          const firstRow = rows[0]
          const delimiters = [',', '\t', ';', '|', ' ']
          let delimiter = ','
          let maxCount = 0
          
          for (const delim of delimiters) {
            const count = (firstRow.match(new RegExp(`\\${delim}`, 'g')) || []).length
            if (count > maxCount) {
              maxCount = count
              delimiter = delim
            }
          }
          
          // For space delimiter, use regex to handle multiple spaces
          const splitRow = (row) => {
            if (delimiter === ' ') {
              return row.trim().split(/\s+/)
            }
            return row.split(delimiter)
          }
          
          const headers = splitRow(rows[0]).map(h => h.trim())
          
          const data = rows.slice(1).map(row => {
            const values = splitRow(row)
            const obj = {}
            headers.forEach((header, index) => {
              obj[header] = values[index]?.trim() || ''
            })
            return obj
          })
          
          onFileUpload({
            fileName: file.name,
            headers,
            data,
            totalRows: data.length
          })
        } catch (err) {
          setError('Error parsing CSV file. Please ensure it is properly formatted.')
        }
      }
      reader.readAsText(file)
    } else {
      // Handle Excel files
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target.result)
          const workbook = XLSX.read(data, { type: 'array' })
          
          // Get the first sheet
          const firstSheetName = workbook.SheetNames[0]
          const worksheet = workbook.Sheets[firstSheetName]
          
          // Convert to JSON
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 })
          
          if (jsonData.length === 0) {
            setError('The Excel file appears to be empty')
            return
          }
          
          const headers = jsonData[0]
          const rows = jsonData.slice(1).map(row => {
            const obj = {}
            headers.forEach((header, index) => {
              obj[header] = row[index] !== undefined ? String(row[index]) : ''
            })
            return obj
          })
          
          onFileUpload({
            fileName: file.name,
            headers,
            data: rows,
            totalRows: rows.length
          })
        } catch (err) {
          setError('Error parsing Excel file. Please ensure it is properly formatted.')
        }
      }
      reader.readAsArrayBuffer(file)
    }
  }, [onFileUpload])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback((e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      handleFile(files[0])
    }
  }, [handleFile])

  return (
    <Card className={`border-2 border-dashed transition-colors ${isDragging ? 'border-primary bg-primary/5' : 'border'}`}>
      <CardContent className="p-8">
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className="flex flex-col items-center justify-center space-y-4"
        >
          <div className="rounded-full bg-primary/10 p-4">
            <Upload className="h-8 w-8 text-primary" />
          </div>
          
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold">Drop your dataset here</h3>
            <p className="text-sm text-muted-foreground">
              or click to browse
            </p>
            <p className="text-xs text-muted-foreground flex items-center gap-2 justify-center">
              <FileSpreadsheet className="h-4 w-4" />
              Supports CSV and Excel files (.csv, .xlsx) up to 10MB
            </p>
          </div>

          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleInputChange}
            className="hidden"
            id="file-upload"
          />
          
          <Button
            variant="outline"
            onClick={() => document.getElementById('file-upload').click()}
            className="rounded-md"
          >
            Select File
          </Button>

          {error && (
            <div className="flex items-center gap-2 text-destructive text-sm">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}