import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from 'react-leaflet';
import { Camera, Activity, AlertTriangle, CloudRain, Map as MapIcon } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import './App.css';

// Fix for default Leaflet icon in React
import L from 'leaflet';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

// 3-Tier ML Architecture Payload Structure (To be fetched from Serverless Backend)
const ML_PIPELINE_DATA = [
  { 
    id: '1701', 
    name: 'CTE/Braddell', 
    lat: 1.34355, lng: 103.86019, 
    status: 'Active', 
    l1_yolo_count: 42, 
    weather: 'Clear',
    l3_forecast_15m: 'Stable (+2%)',
    l2_vlm_anomaly: null
  },
  { 
    id: '1703', 
    name: 'ECP/Marina', 
    lat: 1.29367, lng: 103.86794, 
    status: 'Critical Alert', 
    l1_yolo_count: 0, 
    weather: 'Heavy Rain',
    l3_forecast_15m: 'Severe Cascade (+85%)',
    l2_vlm_anomaly: 'Florence-2: "A multi-vehicle collision blocking all 3 lanes."'
  },
  { 
    id: '1704', 
    name: 'AYE/Clementi', 
    lat: 1.31215, lng: 103.76345, 
    status: 'Active', 
    l1_yolo_count: 145, 
    weather: 'Clear',
    l3_forecast_15m: 'Increasing (+15%)',
    l2_vlm_anomaly: null
  },
  { 
    id: '1705', 
    name: 'KPE/Tampines', 
    lat: 1.36544, lng: 103.90422, 
    status: 'Active', 
    l1_yolo_count: 60, 
    weather: 'Clear',
    l3_forecast_15m: 'Clearing (-5%)',
    l2_vlm_anomaly: null
  },
];

function App() {
  const [cameras, setCameras] = useState(ML_PIPELINE_DATA);

  // Center of Singapore
  const position = [1.3521, 103.8198];

  return (
    <div className="app-container">
      {/* Sidebar Overlay */}
      <div className="sidebar">
        
        {/* Header Panel */}
        <header className="glass-panel header-panel">
          <h1>
            SG <span className="text-gradient">Smart City</span>
          </h1>
          <p>
            <span className="status-dot"></span>
            Live Analytics Pipeline
          </p>
        </header>

        {/* Live Tracking Stats */}
        <div className="stats-grid">
          <div className="glass-panel stat-card">
            <Activity className="stat-icon" size={20} />
            <span className="stat-value">90/90</span>
            <span className="stat-label">Active Cameras</span>
          </div>
          <div className="glass-panel stat-card">
            <Camera className="stat-icon" size={20} />
            <span className="stat-value">2,410</span>
            <span className="stat-label">Vehicles Tracked</span>
          </div>
          <div className="glass-panel stat-card">
            <AlertTriangle className="stat-icon" size={20} color="#ef4444" />
            <span className="stat-value">3</span>
            <span className="stat-label">Anomalies Detected</span>
          </div>
          <div className="glass-panel stat-card">
            <CloudRain className="stat-icon" size={20} color="#8b5cf6" />
            <span className="stat-value">Light</span>
            <span className="stat-label">Weather Impact</span>
          </div>
        </div>

        {/* Camera List */}
        <div className="glass-panel camera-list-panel">
          <div className="panel-header">
            CAMERA FEEDS (TOP CONGESTION)
          </div>
          <div className="camera-feed-list">
            {cameras.map((cam) => (
              <div key={cam.id} className="camera-item">
                <div className="cam-info">
                  <div className="cam-icon">
                    <Camera size={18} />
                  </div>
                  <div className="cam-details">
                    <h4>{cam.name}</h4>
                    <p>L1 Count: {cam.l1_yolo_count} | L3 Forecast: {cam.l3_forecast_15m}</p>
                    {cam.l2_vlm_anomaly && <p className="vlm-alert text-xs text-red-400 mt-1 italic">{cam.l2_vlm_anomaly}</p>}
                  </div>
                </div>
                <span className={`cam-status ${cam.status.includes('Alert') ? 'warning' : 'active'}`}>
                  {cam.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Full Screen Map Container */}
      <div className="map-container">
        <MapContainer center={position} zoom={12} style={{ height: '100%', width: '100%' }} zoomControl={false}>
          {/* Dark modern map tiles - CartoDB Dark Matter */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          
          {/* Map markers for each camera */}
          {cameras.map((cam) => (
            <CircleMarker 
              key={cam.id}
              center={[cam.lat, cam.lng]}
              radius={8}
              pathOptions={{
                color: cam.status === 'Warning' ? '#ef4444' : '#3b82f6',
                fillColor: cam.status === 'Warning' ? '#ef4444' : '#3b82f6',
                fillOpacity: 0.6,
                weight: 2
              }}
            >
              <Popup className="custom-popup">
                <div style={{ textAlign: 'center' }}>
                  <b style={{color: '#3b82f6', fontSize: '14px'}}>{cam.name}</b><br />
                  <span style={{fontWeight: 'bold'}}>YOLO Count: {cam.l1_yolo_count}</span>
                  <br />
                  <span style={{color: '#8b5cf6', fontSize: '12px'}}>T+15 Forecast: {cam.l3_forecast_15m}</span>
                  <br />
                  <span style={{color: '#94a3b8', fontSize: '11px'}}>Weather: {cam.weather}</span>
                  {cam.l2_vlm_anomaly && (
                    <div style={{marginTop: '4px', padding: '4px', backgroundColor: '#451a1a', borderRadius: '4px', fontSize: '11px', color: '#fca5a5'}}>
                      🚨 {cam.l2_vlm_anomaly}
                    </div>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
