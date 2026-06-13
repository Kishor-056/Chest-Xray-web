import React, { useState, useEffect } from 'react';
import { healthCheck } from '../services/api';
import '../styles/ConnectionStatus.css';

function ConnectionStatus() {
    const [status, setStatus] = useState('checking');
    const [lastCheck, setLastCheck] = useState(null);

    const checkConnection = async () => {
        setStatus('checking');
        try {
            const response = await healthCheck();
            if (response.status === 200) {
                setStatus('connected');
                setLastCheck(new Date());
            } else {
                setStatus('error');
            }
        } catch (error) {
            setStatus('disconnected');
            console.error('Backend connection failed:', error.message);
        }
    };

    useEffect(() => {
        checkConnection();
        // Check connection every 30 seconds
        const interval = setInterval(checkConnection, 30000);
        return () => clearInterval(interval);
    }, []);

    const getStatusColor = () => {
        switch (status) {
            case 'connected': return '#10b981';
            case 'disconnected': return '#ef4444';
            case 'checking': return '#f59e0b';
            default: return '#6b7280';
        }
    };

    const getStatusText = () => {
        switch (status) {
            case 'connected': return 'Connected';
            case 'disconnected': return 'Disconnected';
            case 'checking': return 'Checking...';
            default: return 'Unknown';
        }
    };

    return (
        <div className="connection-status">
            <div className="status-indicator" style={{ backgroundColor: getStatusColor() }}></div>
            <span className="status-text">{getStatusText()}</span>
            {lastCheck && (
                <span className="last-check">
                    Last checked: {lastCheck.toLocaleTimeString()}
                </span>
            )}
            <button className="refresh-btn" onClick={checkConnection} disabled={status === 'checking'}>
                ↻
            </button>
        </div>
    );
}

export default ConnectionStatus;
