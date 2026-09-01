#!/usr/bin/env python3
"""Apply or roll back the credential-free OmniRoute local Ollama canary route."""

from __future__ import annotations

import argparse
import datetime
import json
import sqlite3

NODE_ID = "openai-compatible-chat-uec-ollama"
CONNECTION_ID = "uec-ollama-connection"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("apply", "rollback", "status"))
    parser.add_argument("database")
    args = parser.parse_args()
    database = sqlite3.connect(args.database)
    if args.action == "rollback":
        database.execute("delete from provider_connections where id = ?", (CONNECTION_ID,))
        database.execute("delete from provider_nodes where id = ?", (NODE_ID,))
        database.commit()
        print("ROLLED_BACK")
        return 0
    if args.action == "status":
        row = database.execute(
            "select base_url from provider_nodes where id = ?", (NODE_ID,)
        ).fetchone()
        print("CONFIGURED" if row == ("http://127.0.0.1:11434/v1",) else "NOT_CONFIGURED")
        return 0 if row else 1
    now = datetime.datetime.now(datetime.UTC).isoformat()
    provider_data = json.dumps(
        {
            "prefix": "local-ollama",
            "apiType": "chat",
            "baseUrl": "http://127.0.0.1:11434/v1",
            "nodeName": "UEC Local Ollama",
            "chatPath": "/chat/completions",
            "modelsPath": "/models",
        }
    )
    database.execute("delete from provider_connections where id = ?", (CONNECTION_ID,))
    database.execute("delete from provider_nodes where id = ?", (NODE_ID,))
    database.execute(
        "insert into provider_nodes (id,type,name,prefix,api_type,base_url,chat_path,models_path,created_at,updated_at) values (?,?,?,?,?,?,?,?,?,?)",
        (
            NODE_ID,
            "openai-compatible",
            "UEC Local Ollama",
            "local-ollama",
            "chat",
            "http://127.0.0.1:11434/v1",
            "/chat/completions",
            "/models",
            now,
            now,
        ),
    )
    database.execute(
        "insert into provider_connections (id,provider,auth_type,name,priority,is_active,test_status,provider_specific_data,default_model,created_at,updated_at) values (?,?,?,?,?,?,?,?,?,?,?)",
        (
            CONNECTION_ID,
            NODE_ID,
            "apikey",
            "UEC Local Ollama",
            1,
            1,
            "success",
            provider_data,
            "qwen25-7b-fast:latest",
            now,
            now,
        ),
    )
    database.commit()
    print("CONFIG_APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
