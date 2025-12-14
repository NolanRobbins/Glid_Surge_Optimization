'use client'

import React from 'react'

export interface RouteOptionCardProps {
  title: string
  description: string
  distance: number // in miles
  estimatedTime: number // in hours
  cost: number // in dollars
  optimizationLevel?: 'optimal' | 'good' | 'standard'
  optimizationLabel?: string
  departureTime?: string
  arrivalTime?: string
  onClick?: () => void
}

export default function RouteOptionCard({
  title,
  description,
  distance,
  estimatedTime,
  cost,
  optimizationLevel = 'standard',
  optimizationLabel,
  departureTime,
  arrivalTime,
  onClick,
}: RouteOptionCardProps) {
  const getOptimizationBadgeClass = (level: string) => {
    switch (level) {
      case 'optimal':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'good':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const displayOptimizationLabel = optimizationLabel || 
    (optimizationLevel === 'optimal' ? 'Optimal' : 
     optimizationLevel === 'good' ? 'Good' : 'Standard')

  return (
    <div
      onClick={onClick}
      className={`rounded-lg p-4 border border-gray-200 cursor-pointer transition-all hover:border-blue-400 hover:shadow-md ${
        onClick ? '' : 'cursor-default'
      }`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-gray-900">{title}</h4>
            <span
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${getOptimizationBadgeClass(optimizationLevel)}`}
              title={`${displayOptimizationLabel} optimization level`}
            >
              {displayOptimizationLabel}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <div className="text-xs text-gray-500 mb-1">Mileage</div>
          <div className="text-base font-semibold text-gray-900">{distance.toFixed(1)} mi</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Time</div>
          <div className="text-base font-semibold text-gray-900">{estimatedTime.toFixed(1)} hrs</div>
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1">Cost</div>
          <div className="text-base font-semibold text-gray-900">${cost.toFixed(2)}</div>
        </div>
      </div>
      {(departureTime || arrivalTime) && (
        <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-500">
          {departureTime && (
            <div className="flex justify-between">
              <span>Departure:</span>
              <span>{departureTime}</span>
            </div>
          )}
          {arrivalTime && (
            <div className={`flex justify-between ${departureTime ? 'mt-1' : ''}`}>
              <span>Arrival:</span>
              <span>{arrivalTime}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

