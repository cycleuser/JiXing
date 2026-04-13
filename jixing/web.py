
from flask import Flask, jsonify, render_template, request

from .api import (
    delete_session,
    get_session,
    get_stats,
    query_sessions,
    run_moxing,
    run_ollama,
    search_messages,
)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_AS_ASCII"] = False

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/sessions", methods=["GET"])
    def api_sessions_list():
        provider = request.args.get("provider")
        model = request.args.get("model")
        limit = int(request.args.get("limit", 100))

        result = query_sessions(
            model_provider=provider,
            model_name=model,
            limit=limit,
        )
        return jsonify(result.to_dict())

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def api_session_get(session_id: str):
        result = get_session(session_id=session_id)
        return jsonify(result.to_dict())

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def api_session_delete(session_id: str):
        result = delete_session(session_id=session_id)
        return jsonify(result.to_dict())

    @app.route("/api/search", methods=["GET"])
    def api_search():
        query = request.args.get("q", "")
        session_id = request.args.get("session")
        provider = request.args.get("provider")
        limit = int(request.args.get("limit", 100))

        if not query:
            return jsonify({"success": False, "error": "Query required"}), 400

        result = search_messages(
            query=query,
            session_id=session_id,
            model_provider=provider,
            limit=limit,
        )
        return jsonify(result.to_dict())

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        result = get_stats()
        return jsonify(result.to_dict())

    @app.route("/api/ollama/run", methods=["POST"])
    def api_ollama_run():
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        model = data.get("model")
        prompt = data.get("prompt")
        session_id = data.get("session_id")

        if not model or not prompt:
            return jsonify({"success": False, "error": "model and prompt required"}), 400

        result = run_ollama(model=model, prompt=prompt, session_id=session_id)
        return jsonify(result.to_dict())

    @app.route("/api/moxing/run", methods=["POST"])
    def api_moxing_run():
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        model = data.get("model")
        prompt = data.get("prompt")
        session_id = data.get("session_id")

        if not model or not prompt:
            return jsonify({"success": False, "error": "model and prompt required"}), 400

        result = run_moxing(model=model, prompt=prompt, session_id=session_id)
        return jsonify(result.to_dict())

    return app


def create_template_files():
    from pathlib import Path

    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JiXing - Local AI Model Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        h1 { color: #00d9ff; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #16213e; border: none; color: #eee; cursor: pointer; border-radius: 5px; }
        .tab.active { background: #0f3460; color: #00d9ff; }
        .panel { display: none; background: #16213e; padding: 20px; border-radius: 10px; }
        .panel.active { display: block; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #aaa; }
        input, select, textarea { width: 100%; padding: 10px; background: #1a1a2e; border: 1px solid #333; color: #eee; border-radius: 5px; }
        textarea { min-height: 100px; resize: vertical; }
        button { padding: 10px 20px; background: #00d9ff; border: none; color: #1a1a2e; cursor: pointer; border-radius: 5px; font-weight: bold; }
        button:hover { background: #00b8d9; }
        .message { background: #1a1a2e; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 3px solid #00d9ff; }
        .message.user { border-left-color: #ff6b6b; }
        .message.assistant { border-left-color: #4ecdc4; }
        .meta { font-size: 12px; color: #666; margin-top: 5px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-card { background: #1a1a2e; padding: 20px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 32px; color: #00d9ff; }
        .stat-label { color: #aaa; margin-top: 5px; }
        .session-list { max-height: 400px; overflow-y: auto; }
        .session-item { padding: 10px; background: #1a1a2e; margin: 5px 0; border-radius: 5px; cursor: pointer; }
        .session-item:hover { background: #0f3460; }
        #chat-container { max-height: 500px; overflow-y: auto; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>JiXing - 本地AI模型助手</h1>
            <p>Local AI Model Assistant with Long-term Memory</p>
        </header>

        <div class="tabs">
            <button class="tab active" data-tab="chat">Chat</button>
            <button class="tab" data-tab="sessions">Sessions</button>
            <button class="tab" data-tab="search">Search</button>
            <button class="tab" data-tab="stats">Statistics</button>
        </div>

        <div id="chat" class="panel active">
            <div class="form-group">
                <label>Provider</label>
                <select id="chat-provider">
                    <option value="ollama">Ollama</option>
                    <option value="moxing">Moxing</option>
                </select>
            </div>
            <div class="form-group">
                <label>Model</label>
                <input type="text" id="chat-model" placeholder="e.g., gemma3:1b, llama2">
            </div>
            <div id="chat-container"></div>
            <div class="form-group">
                <textarea id="chat-prompt" placeholder="Enter your prompt..."></textarea>
            </div>
            <button onclick="sendChat()">Send</button>
            <input type="hidden" id="chat-session-id">
        </div>

        <div id="sessions" class="panel">
            <h2>Sessions</h2>
            <div class="form-group">
                <label>Filter by Provider</label>
                <select id="filter-provider">
                    <option value="">All</option>
                    <option value="ollama">Ollama</option>
                    <option value="moxing">Moxing</option>
                </select>
            </div>
            <div class="session-list" id="session-list"></div>
            <button onclick="loadSessions()">Refresh</button>
        </div>

        <div id="search" class="panel">
            <h2>Search Messages</h2>
            <div class="form-group">
                <input type="text" id="search-query" placeholder="Search query...">
            </div>
            <button onclick="doSearch()">Search</button>
            <div id="search-results"></div>
        </div>

        <div id="stats" class="panel">
            <h2>Statistics</h2>
            <div class="stats" id="stats-grid"></div>
            <button onclick="loadStats()">Refresh</button>
        </div>
    </div>

    <script>
        const API_BASE = '/api';

        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab).classList.add('active');

                if (tab.dataset.tab === 'sessions') loadSessions();
                if (tab.dataset.tab === 'stats') loadStats();
            });
        });

        async function sendChat() {
            const provider = document.getElementById('chat-provider').value;
            const model = document.getElementById('chat-model').value;
            const prompt = document.getElementById('chat-prompt').value;
            const sessionId = document.getElementById('chat-session-id').value;

            if (!model || !prompt) { alert('Model and prompt required'); return; }

            const container = document.getElementById('chat-container');
            container.innerHTML += `<div class="message user"><strong>You:</strong> ${prompt}</div>`;

            const endpoint = provider === 'ollama' ? '/api/ollama/run' : '/api/moxing/run';
            const body = { model, prompt };
            if (sessionId) body.session_id = sessionId;

            const res = await fetch(API_BASE + endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const result = await res.json();

            if (result.success) {
                document.getElementById('chat-session-id').value = result.data.session_id;
                const metrics = result.data.metrics;
                container.innerHTML += `<div class="message assistant"><strong>Assistant:</strong> ${result.data.response}<div class="meta">Tokens: ${metrics.eval_count || metrics.tokens || 'N/A'} | Time: ${metrics.wall_time_ms || 'N/A'}ms</div></div>`;
            } else {
                container.innerHTML += `<div class="message assistant"><strong>Error:</strong> ${result.error}</div>`;
            }

            document.getElementById('chat-prompt').value = '';
            container.scrollTop = container.scrollHeight;
        }

        async function loadSessions() {
            const provider = document.getElementById('filter-provider').value;
            let url = API_BASE + '/sessions?limit=100';
            if (provider) url += '&provider=' + provider;

            const res = await fetch(url);
            const result = await res.json();

            const list = document.getElementById('session-list');
            if (result.success) {
                list.innerHTML = result.data.map(s => `<div class="session-item" onclick="loadSession('${s.id}')">${s.id.substring(0,8)}... | ${s.model_provider}/${s.model_name} | ${s.created_at.substring(0,10)}</div>`).join('');
            } else {
                list.innerHTML = '<p>Error loading sessions</p>';
            }
        }

        async function loadSession(sessionId) {
            const res = await fetch(API_BASE + '/sessions/' + sessionId);
            const result = await res.json();
            if (result.success) {
                const s = result.data;
                alert(`Session: ${s.id}\\nProvider: ${s.model_provider}\\nModel: ${s.model_name}\\nMessages: ${s.messages.length}`);
            }
        }

        async function doSearch() {
            const query = document.getElementById('search-query').value;
            if (!query) return;

            const res = await fetch(API_BASE + '/search?q=' + encodeURIComponent(query));
            const result = await res.json();

            const container = document.getElementById('search-results');
            if (result.success) {
                container.innerHTML = result.data.map(m => `<div class="message"><strong>${m.role}</strong> (${m.model_provider}/${m.model_name}): ${m.content.substring(0,200)}...<div class="meta">${m.timestamp}</div></div>`).join('');
            } else {
                container.innerHTML = '<p>Error searching</p>';
            }
        }

        async function loadStats() {
            const res = await fetch(API_BASE + '/stats');
            const result = await res.json();

            const grid = document.getElementById('stats-grid');
            if (result.success) {
                const s = result.data;
                grid.innerHTML = `
                    <div class="stat-card"><div class="stat-value">${s.total_sessions}</div><div class="stat-label">Sessions</div></div>
                    <div class="stat-card"><div class="stat-value">${s.total_messages}</div><div class="stat-label">Messages</div></div>
                    <div class="stat-card"><div class="stat-value">${s.total_tokens}</div><div class="stat-label">Tokens</div></div>
                    <div class="stat-card"><div class="stat-value">${(s.total_duration_ms/1000).toFixed(1)}s</div><div class="stat-label">Duration</div></div>
                `;
            }
        }
    </script>
</body>
</html>
"""
    (templates_dir / "index.html").write_text(index_html)


if __name__ == "__main__":
    create_template_files()
    app = create_app()
    app.run(host="127.0.0.1", port=5000)
