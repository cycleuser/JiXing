import json
import time
import threading

import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from .api import (
    delete_session,
    get_session,
    get_stats,
    merge_sessions,
    query_sessions,
    run_moxing,
    search_messages,
)
from .core import SessionManager, ModelRunner, SystemInfo


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
        result = query_sessions(model_provider=provider, model_name=model, limit=limit)
        return jsonify(result.to_dict())

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def api_session_get(session_id: str):
        result = get_session(session_id=session_id)
        return jsonify(result.to_dict())

    @app.route("/api/sessions/<session_id>/messages", methods=["GET"])
    def api_session_messages(session_id: str):
        result = get_session(session_id=session_id)
        if not result.success:
            return jsonify(result.to_dict()), 404
        return jsonify(
            {
                "success": True,
                "data": {"session_id": session_id, "messages": result.data.get("messages", [])},
            }
        )

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def api_session_delete(session_id: str):
        result = delete_session(session_id=session_id)
        return jsonify(result.to_dict())

    @app.route("/api/sessions/merge", methods=["POST"])
    def api_sessions_merge():
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400
        session_ids = data.get("session_ids", [])
        merge_mode = data.get("merge_mode", "timeline")
        model_provider = data.get("model_provider")
        model_name = data.get("model_name")
        if not session_ids or len(session_ids) < 2:
            return jsonify({"success": False, "error": "At least 2 session_ids required"}), 400
        result = merge_sessions(
            session_ids=session_ids,
            merge_mode=merge_mode,
            model_provider=model_provider,
            model_name=model_name,
        )
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
            query=query, session_id=session_id, model_provider=provider, limit=limit
        )
        return jsonify(result.to_dict())

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        result = get_stats()
        return jsonify(result.to_dict())

    @app.route("/api/ollama/models", methods=["GET"])
    def api_ollama_models():
        base_url = request.args.get("base_url", "http://localhost:11434")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return jsonify({"success": True, "data": {"models": models}})
        except requests.exceptions.ConnectionError:
            return jsonify(
                {"success": False, "error": "Cannot connect to Ollama", "data": {"models": []}}
            ), 503
        except Exception as e:
            return jsonify({"success": False, "error": str(e), "data": {"models": []}}), 500

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

    @app.route("/api/ollama/stream", methods=["POST"])
    def api_ollama_stream():
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400
        model = data.get("model")
        prompt = data.get("prompt")
        session_id = data.get("session_id")
        if not model or not prompt:
            return jsonify({"success": False, "error": "model and prompt required"}), 400

        manager = SessionManager.get_instance()
        session = None
        if session_id:
            session = manager.get_session(session_id)
        if session is None:
            session = manager.create_session(
                model_provider="ollama",
                model_name=model,
                system_info=SystemInfo.collect().to_dict(),
            )

        session.add_message("user", prompt)

        def generate():
            full_response = ""
            metrics = {}
            start_time = time.time()
            url = "http://localhost:11434/api/generate"
            payload = {"model": model, "prompt": prompt, "stream": True}

            try:
                resp = requests.post(url, json=payload, stream=True, timeout=300)
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        full_response += token
                        if "done" in chunk and chunk["done"]:
                            metrics = {
                                "eval_count": chunk.get("eval_count", 0),
                                "eval_duration": chunk.get("eval_duration", 0),
                                "total_duration": chunk.get("total_duration", 0),
                                "wall_time_ms": int((time.time() - start_time) * 1000),
                            }
                        yield f"data: {json.dumps({'token': token, 'done': chunk.get('done', False)})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
                return

            session.add_message("assistant", full_response, metrics=metrics)
            manager._save_session(session)

            from .db import Database

            try:
                db = Database()
                db.save_session(session)
            except Exception:
                pass

            yield f"data: {json.dumps({'done': True, 'session_id': session.id, 'metrics': metrics})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

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


if __name__ == "__main__":
    create_template_files()
    app = create_app()
    app.run(host="127.0.0.1", port=5000)
