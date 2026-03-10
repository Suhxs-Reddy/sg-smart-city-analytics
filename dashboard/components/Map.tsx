'use client'

import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { Icon } from 'leaflet'
import { useEffect, useState } from 'react'

interface Camera {
  camera_id: string
  latitude: number
  longitude: number
  vehicle_count: number
  congestion_level: string
  last_updated: string
}

interface MapProps {
  cameras: Camera[]
}

// Fix for default marker icon in Next.js
const cameraIcon = new Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})

const getCongestionColor = (level: string): string => {
  switch (level.toLowerCase()) {
    case 'low':
      return '#10b981' // green
    case 'medium':
      return '#f59e0b' // yellow
    case 'high':
      return '#f97316' // orange
    case 'critical':
      return '#ef4444' // red
    default:
      return '#6b7280' // gray
  }
}

const getCongestionRadius = (vehicleCount: number): number => {
  if (vehicleCount < 10) return 5
  if (vehicleCount < 30) return 8
  if (vehicleCount < 50) return 12
  return 15
}

export default function Map({ cameras }: MapProps) {
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  if (!isMounted) {
    return null
  }

  // Singapore center coordinates
  const singaporeCenter: [number, number] = [1.3521, 103.8198]

  return (
    <MapContainer
      center={singaporeCenter}
      zoom={12}
      scrollWheelZoom={true}
      style={{ height: '100%', width: '100%' }}
      className="z-0"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {cameras.length > 0 ? (
        cameras.map((camera) => (
          <div key={camera.camera_id}>
            {/* Congestion heatmap circle */}
            <CircleMarker
              center={[camera.latitude, camera.longitude]}
              radius={getCongestionRadius(camera.vehicle_count)}
              fillColor={getCongestionColor(camera.congestion_level)}
              fillOpacity={0.3}
              color={getCongestionColor(camera.congestion_level)}
              weight={2}
            />

            {/* Camera marker */}
            <Marker
              position={[camera.latitude, camera.longitude]}
              icon={cameraIcon}
            >
              <Popup className="camera-popup">
                <div>
                  <h3>Camera {camera.camera_id}</h3>
                  <div className="mt-2 space-y-1">
                    <div className="metric">
                      <span className="font-semibold">Vehicles:</span>
                      <span>{camera.vehicle_count}</span>
                    </div>
                    <div className="metric">
                      <span className="font-semibold">Congestion:</span>
                      <span
                        className="px-2 py-1 rounded text-white text-xs"
                        style={{
                          backgroundColor: getCongestionColor(camera.congestion_level)
                        }}
                      >
                        {camera.congestion_level.toUpperCase()}
                      </span>
                    </div>
                    <div className="metric">
                      <span className="font-semibold">Location:</span>
                      <span className="text-xs">
                        {camera.latitude.toFixed(4)}, {camera.longitude.toFixed(4)}
                      </span>
                    </div>
                    <div className="metric">
                      <span className="font-semibold">Updated:</span>
                      <span className="text-xs">
                        {new Date(camera.last_updated).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                  <button
                    className="mt-3 w-full bg-primary text-white px-4 py-2 rounded hover:bg-blue-600 transition text-sm"
                    onClick={() => {
                      window.open(
                        `${process.env.NEXT_PUBLIC_API_URL}/api/cameras/${camera.camera_id}/image`,
                        '_blank'
                      )
                    }}
                  >
                    View Detection Image
                  </button>
                </div>
              </Popup>
            </Marker>
          </div>
        ))
      ) : (
        // Demo markers if no API data available
        Array.from({ length: 90 }, (_, i) => {
          const lat = 1.3521 + (Math.random() - 0.5) * 0.1
          const lng = 103.8198 + (Math.random() - 0.5) * 0.1
          const vehicleCount = Math.floor(Math.random() * 60)
          const congestionLevel =
            vehicleCount < 15
              ? 'low'
              : vehicleCount < 30
              ? 'medium'
              : vehicleCount < 45
              ? 'high'
              : 'critical'

          return (
            <div key={`demo-${i}`}>
              <CircleMarker
                center={[lat, lng]}
                radius={getCongestionRadius(vehicleCount)}
                fillColor={getCongestionColor(congestionLevel)}
                fillOpacity={0.3}
                color={getCongestionColor(congestionLevel)}
                weight={2}
              />
              <Marker position={[lat, lng]} icon={cameraIcon}>
                <Popup className="camera-popup">
                  <div>
                    <h3>Camera {1001 + i}</h3>
                    <div className="mt-2 space-y-1">
                      <div className="metric">
                        <span className="font-semibold">Vehicles:</span>
                        <span>{vehicleCount}</span>
                      </div>
                      <div className="metric">
                        <span className="font-semibold">Congestion:</span>
                        <span
                          className="px-2 py-1 rounded text-white text-xs"
                          style={{
                            backgroundColor: getCongestionColor(congestionLevel)
                          }}
                        >
                          {congestionLevel.toUpperCase()}
                        </span>
                      </div>
                      <div className="metric">
                        <span className="font-semibold">Location:</span>
                        <span className="text-xs">
                          {lat.toFixed(4)}, {lng.toFixed(4)}
                        </span>
                      </div>
                      <div className="metric">
                        <span className="font-semibold">Status:</span>
                        <span className="text-xs text-yellow-600">Demo Mode</span>
                      </div>
                    </div>
                  </div>
                </Popup>
              </Marker>
            </div>
          )
        })
      )}
    </MapContainer>
  )
}
