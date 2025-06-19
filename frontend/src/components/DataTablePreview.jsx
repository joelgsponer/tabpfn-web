import React from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card'
import { Button } from './ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table'
import { PlayCircle, FileSpreadsheet } from 'lucide-react'
import { ColumnConfiguration } from './ColumnConfiguration'
import { ModelConfiguration } from './ModelConfiguration'

export function DataTablePreview({ 
  fileData, 
  targetColumn, 
  onTargetColumnChange, 
  onRunModel,
  isLoading,
  columnTypes,
  onColumnTypeChange,
  excludedColumns,
  onExcludedColumnsChange,
  modelConfig,
  onModelConfigChange
}) {
  if (!fileData) return null

  const previewRows = fileData.data.slice(0, 10)
  const hasMoreRows = fileData.totalRows > 10

  return (
    <Card className="mt-8">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FileSpreadsheet className="h-5 w-5" />
              Data Preview
            </CardTitle>
            <CardDescription>
              {fileData.fileName} • {fileData.totalRows} rows • {fileData.headers.length} columns
            </CardDescription>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label htmlFor="target-column" className="text-sm font-medium">
                Target Column:
              </label>
              <Select
                value={targetColumn}
                onValueChange={onTargetColumnChange}
                disabled={isLoading}
              >
                <SelectTrigger id="target-column" className="w-[200px] rounded-md">
                  <SelectValue placeholder="Select target column..." />
                </SelectTrigger>
                <SelectContent>
                  {fileData.headers.filter(header => !excludedColumns.includes(header)).map((header) => (
                    <SelectItem key={header} value={header}>
                      {header}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <Button 
              onClick={onRunModel}
              disabled={!targetColumn || isLoading}
              className="rounded-md"
            >
              <PlayCircle className="h-4 w-4 mr-2" />
              Run TabPFN
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="rounded-md border overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                {fileData.headers.map((header) => (
                  <TableHead 
                    key={header}
                    className={header === targetColumn ? 'bg-primary/10 font-bold' : ''}
                  >
                    {header}
                    {header === targetColumn && (
                      <span className="ml-1 text-xs text-primary">(Target)</span>
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {previewRows.map((row, index) => (
                <TableRow key={index}>
                  {fileData.headers.map((header) => (
                    <TableCell 
                      key={header}
                      className={header === targetColumn ? 'bg-primary/5' : ''}
                    >
                      {row[header]}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        
        {hasMoreRows && (
          <p className="text-sm text-muted-foreground mt-2 text-center">
            Showing first 10 rows of {fileData.totalRows} total rows
          </p>
        )}
      </CardContent>
      
      <ColumnConfiguration
        fileData={fileData}
        columnTypes={columnTypes}
        onColumnTypeChange={onColumnTypeChange}
        excludedColumns={excludedColumns}
        onExcludedColumnsChange={onExcludedColumnsChange}
      />
      
      <ModelConfiguration
        fileData={fileData}
        excludedColumns={excludedColumns}
        modelConfig={modelConfig}
        onModelConfigChange={onModelConfigChange}
      />
    </Card>
  )
}