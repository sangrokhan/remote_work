import React, { useState, useEffect, useRef } from 'react';

function App() {
    const [command, setCommand] = useState('');
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(false);
    const logsEndRef = useRef(null);

    const addLog = (msg) => {
        setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    // Auto-scroll to bottom
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    const pollTask = async (taskId) => {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`http://localhost:8080/api/task/${taskId}`);
                if (!res.ok) return;
                const data = await res.json();

                // If there are new logs in the task, we could show them, 
                // but for now we just show status updates to avoid dupes or complexity
                // simplified: just show status changes or completion

                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(interval);
                    setLoading(false);
                    addLog(`Task ${taskId} finished. Status: ${data.status}`);
                    if (data.result) addLog(`Result: ${data.result}`);
                    if (data.error) addLog(`Error: ${data.error}`);
                } else {
                    // Optionally enable this to see heartbeat
                    // addLog(`Task ${taskId} is ${data.status}...`);
                }
            } catch (err) {
                console.error("Polling error", err);
            }
        }, 2000);
    };

    const runRoutine = async () => {
        setLoading(true);
        addLog('Starting Daily Routine...');
        try {
            const res = await fetch('http://localhost:8080/api/run-daily-routine', {
                method: 'POST',
            });
            const data = await res.json();
            addLog(`Response: ${JSON.stringify(data)}`);

            if (data.task_id) {
                addLog(`Task ID received: ${data.task_id}. Polling for updates...`);
                pollTask(data.task_id);
            } else {
                setLoading(false);
            }
        } catch (err) {
            addLog(`Error: ${err.message}`);
            setLoading(false);
        }
    };

    const sendCommand = async (e) => {
        e.preventDefault();
        if (!command) return;

        setLoading(true);
        addLog(`Sending command: ${command}`);
        try {
            const res = await fetch('http://localhost:8080/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command }),
            });
            const data = await res.json();
            addLog(`Response: ${JSON.stringify(data)}`);

            if (data.task_id) {
                addLog(`Task ID received: ${data.task_id}. Polling for updates...`);
                pollTask(data.task_id);
            } else {
                setLoading(false);
            }

        } catch (err) {
            addLog(`Error: ${err.message}`);
            setLoading(false);
        } finally {
            setCommand('');
        }
    };

    return (
        <div style={{ fontFamily: 'Inter, sans-serif', maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
            <header style={{ marginBottom: '2rem' }}>
                <h1 style={{ fontSize: '2rem', fontWeight: 'bold' }}>🤖 A2A Agent Interface</h1>
                <p style={{ color: '#666' }}>Control your agent and view status.</p>
            </header>

            <div style={{ display: 'grid', gap: '1rem', marginBottom: '2rem' }}>
                <div style={{ padding: '1.5rem', border: '1px solid #eee', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                    <h2 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Quick Actions</h2>
                    <button
                        onClick={runRoutine}
                        disabled={loading}
                        style={{
                            padding: '0.8rem 1.5rem',
                            background: '#2563eb',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: loading ? 'not-allowed' : 'pointer',
                            opacity: loading ? 0.7 : 1
                        }}
                    >
                        {loading ? 'Running...' : '▶ Start Daily Routine'}
                    </button>
                </div>

                <div style={{ padding: '1.5rem', border: '1px solid #eee', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                    <h2 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Command Input</h2>
                    <form onSubmit={sendCommand} style={{ display: 'flex', gap: '0.5rem' }}>
                        <input
                            type="text"
                            value={command}
                            onChange={(e) => setCommand(e.target.value)}
                            placeholder="Enter a command (e.g., 'Check status')"
                            style={{ flex: 1, padding: '0.8rem', borderRadius: '6px', border: '1px solid #ddd' }}
                        />
                        <button
                            type="submit"
                            disabled={loading}
                            style={{
                                padding: '0.8rem 1.5rem',
                                background: '#10b981',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: loading ? 'not-allowed' : 'pointer'
                            }}
                        >
                            Send
                        </button>
                    </form>
                </div>
            </div>

            <div style={{ background: '#1e293b', color: '#f8fafc', padding: '1.5rem', borderRadius: '8px', height: '300px', overflowY: 'auto' }}>
                <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', color: '#94a3b8' }}>Activity Log</h3>
                {logs.length === 0 && <p style={{ color: '#64748b' }}>No activity yet.</p>}
                {logs.map((log, i) => (
                    <div key={i} style={{ marginBottom: '0.5rem', fontFamily: 'monospace' }}>{log}</div>
                ))}
                <div ref={logsEndRef} />
            </div>
        </div>
    );
}

export default App;
