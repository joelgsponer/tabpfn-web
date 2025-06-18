import React from 'react'
import { Loader2 } from 'lucide-react'

export function LoadingSpinner() {
  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-card p-8 rounded-lg shadow-lg flex flex-col items-center space-y-4">
        <Loader2 className="h-12 w-12 text-primary animate-spin" />
        <div className="text-center">
          <h3 className="text-lg font-semibold">Processing your data</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Training model and generating predictions...
          </p>
        </div>
      </div>
    </div>
  )
}