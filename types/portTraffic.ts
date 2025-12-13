export interface PortTrafficData {
  date: number // Unix timestamp in milliseconds
  year: number
  month: number
  day: number
  portid: string
  portname: string
  country: string
  ISO3: string
  portcalls_container: number
  portcalls_dry_bulk: number
  portcalls_general_cargo: number
  portcalls_roro: number
  portcalls_tanker: number
  portcalls_cargo: number
  portcalls: number
  import_container: number
  import_dry_bulk: number
  import_general_cargo: number
  import_roro: number
  import_tanker: number
  import_cargo: number
  import: number
  export_container: number
  export_dry_bulk: number
  export_general_cargo: number
  export_roro: number
  export_tanker: number
  export_cargo: number
  export: number
  ObjectId: number
}

export interface ArcGISResponse {
  objectIdFieldName: string
  uniqueIdField: {
    name: string
    isSystemMaintained: boolean
  }
  globalIdFieldName: string
  fields: Array<{
    name: string
    type: string
    alias: string
    sqlType: string
    length?: number
  }>
  exceededTransferLimit: boolean
  features: Array<{
    attributes: PortTrafficData
  }>
}

export interface PortSummary {
  portid: string
  portname: string
  country: string
  ISO3: string
  totalPortcalls: number
  totalImport: number
  totalExport: number
  avgDailyPortcalls: number
  avgDailyImport: number
  avgDailyExport: number
}

export type CargoType = 
  | 'container' 
  | 'dry_bulk' 
  | 'general_cargo' 
  | 'roro' 
  | 'tanker' 
  | 'cargo' 
  | 'all'

export interface DateRange {
  start: Date
  end: Date
}

