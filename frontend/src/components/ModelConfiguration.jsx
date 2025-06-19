import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { Label } from './ui/label'
import { Input } from './ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Brain } from 'lucide-react'

export function ModelConfiguration({ 
  fileData, 
  excludedColumns, 
  modelConfig, 
  onModelConfigChange 
}) {
  if (!fileData) return null

  const handleConfigChange = (key, value) => {
    onModelConfigChange({ ...modelConfig, [key]: value })
  }

  const featureColumns = fileData.headers.filter(header => 
    !excludedColumns.includes(header)
  )

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Model Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        
        {/* Model Hyperparameters */}
        <div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            
            {/* Train/Test Split */}
            <div className="space-y-2">
              <Label htmlFor="test-size">Test Split Ratio</Label>
              <Input
                id="test-size"
                type="number"
                min="0.1"
                max="0.5"
                step="0.05"
                value={modelConfig.test_size}
                onChange={(e) => handleConfigChange('test_size', parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Fraction of data for testing (0.1 - 0.5)
              </p>
            </div>

            {/* TabPFN: Ensemble Configurations */}
            <div className="space-y-2">
              <Label htmlFor="n-ensemble">Ensemble Configurations</Label>
              <Select
                value={modelConfig.N_ensemble_configurations?.toString() || '16'}
                onValueChange={(value) => handleConfigChange('N_ensemble_configurations', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="4">4 (fastest)</SelectItem>
                  <SelectItem value="8">8</SelectItem>
                  <SelectItem value="16">16 (default)</SelectItem>
                  <SelectItem value="32">32 (better accuracy)</SelectItem>
                  <SelectItem value="64">64 (best accuracy)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Number of ensemble configurations (more = slower but better)
              </p>
            </div>

            {/* TabPFN: Device */}
            <div className="space-y-2">
              <Label htmlFor="device">Computing Device</Label>
              <Select
                value={modelConfig.device || 'auto'}
                onValueChange={(value) => handleConfigChange('device', value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect (recommended)</SelectItem>
                  <SelectItem value="cpu">CPU (compatible)</SelectItem>
                  <SelectItem value="cuda">GPU (CUDA)</SelectItem>
                  <SelectItem value="mps">GPU (Apple Silicon)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Auto-detect will use GPU if available
              </p>
            </div>

            {/* TabPFN: Max Samples */}
            <div className="space-y-2">
              <Label htmlFor="max-samples">Max Training Samples</Label>
              <Select
                value={modelConfig.max_samples?.toString() || '1000'}
                onValueChange={(value) => handleConfigChange('max_samples', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="500">500 (very fast)</SelectItem>
                  <SelectItem value="1000">1000 (recommended)</SelectItem>
                  <SelectItem value="2000">2000 (slower, may fail)</SelectItem>
                  <SelectItem value="5000">5000 (experimental)</SelectItem>
                  <SelectItem value="10000">10000 (high memory)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Higher values may cause out-of-memory errors
              </p>
            </div>

            {/* Random State */}
            <div className="space-y-2">
              <Label htmlFor="random-state">Random Seed</Label>
              <Input
                id="random-state"
                type="number"
                min="0"
                max="9999"
                step="1"
                value={modelConfig.random_state}
                onChange={(e) => handleConfigChange('random_state', parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Seed for reproducible results
              </p>
            </div>

            {/* N Estimators */}
            <div className="space-y-2">
              <Label htmlFor="n-estimators">N Estimators</Label>
              <Select
                value={modelConfig.n_estimators?.toString() || '4'}
                onValueChange={(value) => handleConfigChange('n_estimators', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 (fastest)</SelectItem>
                  <SelectItem value="2">2</SelectItem>
                  <SelectItem value="4">4 (default)</SelectItem>
                  <SelectItem value="8">8</SelectItem>
                  <SelectItem value="16">16 (slower)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Number of estimators for ensemble
              </p>
            </div>

            {/* N Jobs */}
            <div className="space-y-2">
              <Label htmlFor="n-jobs">Parallel Jobs</Label>
              <Select
                value={modelConfig.n_jobs?.toString() || '-1'}
                onValueChange={(value) => handleConfigChange('n_jobs', parseInt(value))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="-1">All CPUs (-1)</SelectItem>
                  <SelectItem value="1">Single CPU (1)</SelectItem>
                  <SelectItem value="2">2 CPUs</SelectItem>
                  <SelectItem value="4">4 CPUs</SelectItem>
                  <SelectItem value="8">8 CPUs</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Number of parallel jobs (-1 = use all available)
              </p>
            </div>

          </div>
        </div>

        {/* Model Summary */}
        <div className="bg-muted/50 p-3 rounded-md">
          <h5 className="font-medium text-sm mb-2">TabPFN Configuration Summary:</h5>
          <div className="text-xs text-muted-foreground space-y-1">
            <p>• Features: {featureColumns.length} columns (max: 100)</p>
            <p>• Test split: {(modelConfig.test_size * 100).toFixed(0)}% for testing</p>
            <p>• Ensemble: {modelConfig.N_ensemble_configurations || 16} configurations</p>
            <p>• Estimators: {modelConfig.n_estimators || 4}, Jobs: {modelConfig.n_jobs || -1}</p>
            <p>• Device: {modelConfig.device || 'cpu'}</p>
          </div>
          <div className="text-xs text-orange-600 mt-2">
            <p>⚠️ TabPFN limits: ≤{modelConfig.max_samples || 1000} samples, ≤100 features, ≤10 classes</p>
          </div>
        </div>

      </CardContent>
    </Card>
  )
}