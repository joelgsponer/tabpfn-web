import React from 'react'
import { AlertCircle, X } from 'lucide-react'
import { Card, CardContent } from './ui/card'
import { Button } from './ui/button'

export function ErrorMessage({ message, onDismiss }) {
  return (
    <Card className="mt-4 border-destructive/50 bg-destructive/10">
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
            <div>
              <h4 className="font-semibold text-destructive">Error</h4>
              <p className="text-sm text-destructive/90 mt-1">{message}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onDismiss}
            className="h-8 w-8 rounded-md"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}