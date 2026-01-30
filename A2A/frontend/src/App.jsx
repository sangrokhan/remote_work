import React, { useState, useEffect, useRef } from 'react';

function App() {
    const [command, setCommand] = useState('');
    const [parquetPath, setParquetPath] = useState('data/mock_data.parquet');
    const [trainDataPath, setTrainDataPath] = useState('data/mock_data.parquet');
    const [useRandomData, setUseRandomData] = useState(false);
    const [trainGoal, setTrainGoal] = useState('');
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(false);
    const logsEndRef = useRef(null);
    const pollingIntervalRef = useRef(null);

    const addLog = (msg, fromServer = false) => {
        const timestamp = new Date().toLocaleTimeString();
        const prefix = fromServer ? "SERVER" : "CLIENT";
        setLogs((prev) => [...prev, `[${timestamp}] [${prefix}] ${msg}`]);
    };

    // Auto-scroll to bottom
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    const pollTask = async (taskId) => {
        // Clear any existing polling
        if (pollingIntervalRef.current) {
            clearTimeout(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
        }

        // Use a local variable to track logs for THIS specific task poll
        let lastLogIdx = 0;
        let isPolling = true;

        const poll = async () => {
            if (!isPolling) return;

            try {
                const res = await fetch(`http://localhost:8080/api/task/${taskId}`);
                if (!res.ok) {
                    // Schedule next poll and return
                    if (isPolling) {
                        pollingIntervalRef.current = setTimeout(poll, 1000);
                    }
                    return;
                }
                const data = await res.json();

                // Append new server logs using local index
                if (data.logs && data.logs.length > lastLogIdx) {
                    const newLogs = data.logs.slice(lastLogIdx);
                    newLogs.forEach(log => addLog(log, true));
                    lastLogIdx = data.logs.length;
                }

                if (data.status === 'completed' || data.status === 'failed') {
                    isPolling = false;
                    pollingIntervalRef.current = null;
                    setLoading(false);
                    addLog(`Task ${taskId} finished. Status: ${data.status}`);
                    if (data.status === 'failed' && data.error) addLog(`Error: ${data.error}`);
                } else {
                    // Schedule next poll AFTER this one completes
                    if (isPolling) {
                        pollingIntervalRef.current = setTimeout(poll, 1000);
                    }
                }
            } catch (err) {
                console.error("Polling error", err);
                // Continue polling on error
                if (isPolling) {
                    pollingIntervalRef.current = setTimeout(poll, 1000);
                }
            }
        };

        // Start polling immediately
        poll();
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

    const analyzeParquet = async (e) => {
        e.preventDefault();
        if (!parquetPath) return;

        setLoading(true);
        addLog(`Requesting analysis for: ${parquetPath}`);
        try {
            const res = await fetch('http://localhost:8080/api/analyze-parquet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ files: [parquetPath] }),
            });
            const data = await res.json();

            if (data.status === 'success') {
                addLog(`Analysis Result:\n${data.analysis}`);
            } else {
                addLog(`Analysis/Server Error: ${JSON.stringify(data)}`);
            }
        } catch (err) {
            addLog(`Error: ${err.message}`);
        } finally {
            setLoading(false);
            // setParquetPath(''); // keep path for convenience
        }
    };

    const startAutoTrain = async (e) => {
        e.preventDefault();
        if (!trainDataPath && !useRandomData) return;

        setLoading(true);
        if (useRandomData) {
            addLog(`Starting AutoML Pipeline with RANDOM Synthetic Data...`);
        } else {
            addLog(`Starting AutoML Pipeline for: ${trainDataPath}`);
        }
        addLog(`Goal: ${trainGoal || "Default"}`);

        try {
            const res = await fetch('http://localhost:8080/api/auto-train-pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    files: useRandomData ? [] : [trainDataPath],
                    use_random_data: useRandomData,
                    goal: trainGoal
                }),
            });
            const data = await res.json();
            addLog(`Response: ${JSON.stringify(data)}`);

            if (data.task_id) {
                addLog(`Pipeline Task ID: ${data.task_id}. Polling...`);
                pollTask(data.task_id);
            }
        } catch (err) {
            addLog(`Pipeline Error: ${err.message}`);
            setLoading(false);
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

            <div style={{ display: 'grid', gap: '1rem', marginBottom: '2rem' }}>
                <div style={{ padding: '1.5rem', border: '1px solid #eee', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                    <h2 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Parquet Analysis</h2>
                    <form onSubmit={analyzeParquet} style={{ display: 'flex', gap: '0.5rem' }}>
                        <input
                            type="text"
                            value={parquetPath}
                            onChange={(e) => setParquetPath(e.target.value)}
                            placeholder="Enter absolute file path (e.g. /home/user/data.parquet)"
                            style={{ flex: 1, padding: '0.8rem', borderRadius: '6px', border: '1px solid #ddd' }}
                        />
                        <button
                            type="submit"
                            disabled={loading}
                            style={{
                                padding: '0.8rem 1.5rem',
                                background: '#8b5cf6',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: loading ? 'not-allowed' : 'pointer'
                            }}
                        >
                            Analyze
                        </button>
                    </form>
                </div>
            </div>



            <div style={{ display: 'grid', gap: '1rem', marginBottom: '2rem' }}>
                <div style={{ padding: '1.5rem', border: '1px solid #eee', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
                    <h2 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>🚀 AutoML Pipeline</h2>
                    <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '1rem' }}>
                        Fully automated: Analysis → Planning → Training
                    </p>
                    <form onSubmit={startAutoTrain} style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                        <div>
                            <div style={{ marginBottom: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <input
                                    type="checkbox"
                                    id="useRandom"
                                    checked={useRandomData}
                                    onChange={(e) => setUseRandomData(e.target.checked)}
                                />
                                <label htmlFor="useRandom" style={{ fontWeight: 600, fontSize: '0.9rem', cursor: 'pointer' }}>
                                    Use Random Synthetic Data (Test Triggering)
                                </label>
                            </div>

                            <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: 600, opacity: useRandomData ? 0.5 : 1 }}>
                                Dataset Path
                            </label>
                            <input
                                type="text"
                                value={trainDataPath}
                                disabled={useRandomData}
                                onChange={(e) => setTrainDataPath(e.target.value)}
                                placeholder="/data/dataset.parquet"
                                style={{ width: '100%', padding: '0.8rem', borderRadius: '6px', border: '1px solid #ddd', opacity: useRandomData ? 0.5 : 1 }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: 600 }}>Training Goal</label>
                            <input
                                type="text"
                                value={trainGoal}
                                onChange={(e) => setTrainGoal(e.target.value)}
                                placeholder="E.g., Train a high-accuracy classifier for this data"
                                style={{ width: '100%', padding: '0.8rem', borderRadius: '6px', border: '1px solid #ddd' }}
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={loading}
                            style={{
                                padding: '0.8rem 1.5rem',
                                background: '#f59e0b',
                                color: 'white',
                                border: 'none',
                                borderRadius: '6px',
                                cursor: loading ? 'not-allowed' : 'pointer',
                                fontWeight: 'bold'
                            }}
                        >
                            {loading ? 'Processing...' : '⚡ Start Auto-Train Pipeline'}
                        </button>
                    </form>
                </div>
            </div>

            <div style={{ background: '#1e293b', color: '#f8fafc', padding: '1.5rem', borderRadius: '8px', height: '300px', overflowY: 'auto' }}>
                <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', color: '#94a3b8' }}>Activity Log</h3>
                {logs.length === 0 && <p style={{ color: '#64748b' }}>No activity yet.</p>}
                {logs.map((log, i) => (
                    <div key={i} style={{ marginBottom: '0.5rem', fontFamily: 'monospace', whiteSpace: 'pre-wrap', borderBottom: '1px solid #334155', paddingBottom: '0.2rem' }}>{log}</div>
                ))}
                <div ref={logsEndRef} />
            </div>
        </div >
    );
}

export default App;
