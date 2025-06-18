import React from 'react'
import { Brain } from 'lucide-react'

export function Header() {
  return (
    <header className="border-b">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          <h1 className="text-2xl font-bold">TabPFN</h1>
          <span className="text-muted-foreground">Tabular Classification Made Simple</span>
        </div>
      </div>
    </header>
  )
}