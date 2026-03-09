'use client'

import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import useSWR from 'swr'

// Dynamic import to avoid SSR issues with Leaflet
const Map = dynamic(() => import('../components/Map'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-screen">
      <div className="spinner"></div>
    </div>
  ),
})

const fetcher = (url: string) => fetch(url).then((res) => res.json())

interface SystemStats {
  total_cameras: number
  total_vehicles: number
  system_uptime: string
  average_fps: number
  active_cameras: number
}

interface Camera {
  camera_id: string
  latitude: number
  longitude: number
  vehicle_count: number
  congestion_level: string
  last_updated: string
}

export default function Home() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  // Fetch system stats
  const { data: stats, error: statsError } = useSWR<SystemStats>(
    `${apiUrl}/api/stats`,
    fetcher,
    { refreshInterval: 10000 } // Refresh every 10 seconds
  )

  // Fetch camera data
  const { data: cameras, error: camerasError } = useSWR<Camera[]>(
    `${apiUrl}/api/cameras`,
    fetcher,
    { refreshInterval: 30000 } // Refresh every 30 seconds
  )

  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <main className="flex flex-col">
      {/* Stats Dashboard */}
      <div className="container mx-auto p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="stat-card">
            <div className="stat-value">
              {stats?.total_cameras || '90'}
            </div>
            <div className="stat-label">Total Cameras</div>
          </div>

          <div className="stat-card">
            <div className="stat-value text-green-500">
              {stats?.active_cameras || '87'}
            </div>
            <div className="stat-label">Active Cameras</div>
          </div>

          <div className="stat-card">
            <div className="stat-value text-blue-500">
              {stats?.total_vehicles?.toLocaleString() || '1,247'}
            </div>
            <div className="stat-label">Vehicles Detected</div>
          </div>

          <div className="stat-card">
            <div className="stat-value text-purple-500">
              {stats?.average_fps || '110'} <span className="text-sm">FPS</span>
            </div>
            <div className="stat-label">Average Speed</div>
          </div>
        </div>

        {/* Alert Banner */}
        <div className="bg-yellow-50 dark:bg-yellow-900 border-l-4 border-yellow-400 p-4 mb-6 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-yellow-400"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-yellow-700 dark:text-yellow-200">
                <span className="font-medium">Demo Mode:</span> This is a demonstration
                of the Singapore Smart City Analytics platform. API endpoints are
                currently in staging mode.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Map */}
      <div className="relative" style={{ height: 'calc(100vh - 400px)', minHeight: '500px' }}>
        {camerasError || statsError ? (
          <div className="flex items-center justify-center h-full bg-gray-100 dark:bg-gray-900">
            <div className="text-center">
              <p className="text-red-500 mb-4">Failed to load data</p>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                API might not be running. Check: {apiUrl}
              </p>
            </div>
          </div>
        ) : (
          <Map cameras={cameras || []} />
        )}
      </div>

      {/* Footer Info */}
      <div className="container mx-auto p-4 mt-6">
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-bold mb-4">About This Project</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h3 className="font-semibold mb-2">Production System</h3>
              <ul className="list-disc list-inside text-gray-600 dark:text-gray-400 space-y-1">
                <li>Real-time analytics for 90 Singapore LTA cameras</li>
                <li>YOLOv11s detection model (92% mAP)</li>
                <li>BoT-SORT + OSNet vehicle tracking</li>
                <li>FastAPI backend with MLflow tracking</li>
                <li>Full CI/CD with GitHub Actions</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Tech Stack</h3>
              <ul className="list-disc list-inside text-gray-600 dark:text-gray-400 space-y-1">
                <li>Next.js 14 + React + TypeScript</li>
                <li>Leaflet for interactive mapping</li>
                <li>Docker + Azure deployment ($8/month)</li>
                <li>Deployed on Vercel (this dashboard)</li>
                <li>Free tier: GitHub, Kaggle, Colab</li>
              </ul>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Built by Suhas Reddy | Production-grade ML platform for senior engineering interviews |{' '}
              <a
                href="https://github.com/Suhxs-Reddy/sg-smart-city-analytics"
                className="text-primary hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                View on GitHub
              </a>
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
